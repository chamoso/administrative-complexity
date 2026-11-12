"""Benchmark: a single, direct model judgment of administrative complexity.

Answers the question whether the fifteen-dimension instrument adds anything over
asking the same model for one global rating. Everything except the rubric is held
identical to the main assessment: model, system instruction, document excerpt,
decoding at temperature zero and output format.

    python hpc_score_global.py --input data/processed/validation_collections_scoring.parquet \\
        --out outputs/llm_scores --tag global --shard 0 --n-shards 2
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.features import llm as llm_module  # noqa: E402

GLOBAL_RUBRIC = """You will assess one official administrative document.

Rate the document on one scale from 1 to 10.

ADMINISTRATIVE_COMPLEXITY - overall, how demanding the document is for a member of
the intended audience who must act on it: working out what is required, obtaining
what is needed and completing the procedure it describes.
  1-2  Very simple: little or nothing is required, and it is obvious what to do.
  3-4  Simple: a few plain requirements.
  5-6  Moderate: several requirements, some conditions or documents.
  7-8  Demanding: many requirements, conditions, documents or prior knowledge.
  9-10 Very demanding: professional help is effectively required to comply.

Answer with exactly this JSON object and no other text:
{"administrative_complexity": <1-10>,
 "justification": "<one or two sentences citing what in the document drove the score>"}
"""
JSON_RE = re.compile(r"\\{.*\\}", re.S)


def parse(raw: str):
    m = re.search(r"\{.*\}", raw or "", re.S)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
        value = int(data["administrative_complexity"])
        return {"llm_global_complexity": value, "justification": str(data.get("justification", ""))} \
            if 1 <= value <= 10 else None
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tag", default="global")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-input-chars", type=int, default=12000)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260726)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    df = pd.read_parquet(args.input)
    df = df.iloc[args.shard :: args.n_shards].reset_index(drop=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    outfile = out / f"scores_{args.tag}_shard{args.shard}.jsonl"
    done = set()
    if outfile.exists():
        done = {json.loads(line)["doc_id"] for line in open(outfile, encoding="utf-8") if line.strip()}
    df = df[~df.doc_id.isin(done)]
    print(f"shard {args.shard}: {len(df)} documents to assess", flush=True)

    llm_module._RUBRIC = GLOBAL_RUBRIC  # same prompt layout, different rubric
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    engine = LLM(model=args.model, gpu_memory_utilization=0.90, max_model_len=8192, dtype="bfloat16")
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=200, seed=args.seed)

    start, written = time.time(), 0
    with open(outfile, "a", encoding="utf-8") as fh:
        for b in range(0, len(df), args.batch_size):
            batch = df.iloc[b : b + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": llm_module.SYSTEM_PROMPT},
                     {"role": "user", "content": llm_module.build_prompt(
                         r.title, r.text or "", getattr(r, "language", "en"), max_chars=args.max_input_chars)}],
                    tokenize=False, add_generation_prompt=True)
                for r in batch.itertuples()
            ]
            for r, o in zip(batch.itertuples(), engine.generate(prompts, sampling)):
                raw = o.outputs[0].text if o.outputs else ""
                parsed = parse(raw)
                fh.write(json.dumps({"doc_id": r.doc_id, "model": args.model, "tag": args.tag,
                                     "temperature": 0.0, "parsed_ok": parsed is not None,
                                     **(parsed or {"raw": raw[:500]})}, ensure_ascii=False) + "\n")
                written += 1
            print(f"[shard {args.shard}] {written}/{len(df)} ({written / (time.time() - start):.1f}/s)", flush=True)


if __name__ == "__main__":
    main()
