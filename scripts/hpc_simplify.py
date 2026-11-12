"""Experiments 5 and 6: what happens to the index when a procedure is simplified.

Two interventions are applied to the same documents.

  Removal   the requirements a procedure imposes are stripped out, leaving the
            rest of the text untouched. This is a mechanical edit, not a
            rewrite, so the change in the index is attributable to the
            requirements alone.

  Rewriting the procedure is rewritten in plain language, preserving every
            obligation. Nothing is removed; only the presentation changes.

The point of the pair is to separate two things public managers are often told
are the same thing. If the index falls under rewriting but not under removal,
complexity is presentational. If it falls under removal but not rewriting, the
burden is in what is demanded, and clearer writing will not touch it.

    python hpc_simplify.py --input corpus.parquet --out outputs/simplify \\
                           --sample 1500 --shard 0 --n-shards 7
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.features.llm import excerpt  # noqa: E402
from adminplex.features.simplify import INTERVENTIONS, SYSTEM_PROMPT  # noqa: E402


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--sample", type=int, default=1500,
                    help="documents to intervene on; the whole corpus is not needed for a contrast")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--max-input-chars", type=int, default=8000)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-words", type=int, default=200)
    ap.add_argument("--gpu-memory-utilisation", type=float, default=0.90)
    ap.add_argument("--seed", type=int, default=20260726)
    return ap.parse_args()


def load_shard(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.input, columns=["doc_id", "title", "text", "language", "country", "source"])
    df = df[df.text.str.split().str.len() >= args.min_words]
    # A fixed sample, drawn once from the whole corpus and then sharded, so the
    # same documents are intervened on however many GPUs are available.
    df = df.sample(n=min(args.sample, len(df)), random_state=args.seed).reset_index(drop=True)
    if args.n_shards > 1:
        df = df.iloc[args.shard :: args.n_shards].copy()
    return df.reset_index(drop=True)


def already_done(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                    done.add((rec["doc_id"], rec["intervention"]))
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"simplified_shard{args.shard}.jsonl"

    df = load_shard(args)
    done = already_done(outfile)
    jobs = [
        (row.doc_id, name, row)
        for row in df.itertuples()
        for name in INTERVENTIONS
        if (row.doc_id, name) not in done
    ]
    print(f"[shard {args.shard}] {len(df)} documents, {len(jobs)} interventions to run", flush=True)
    if not jobs:
        return

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_memory_utilisation,
        max_model_len=12288,
        dtype="bfloat16",
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_new_tokens, seed=args.seed)

    start = time.time()
    written = 0
    for begin in range(0, len(jobs), args.batch_size):
        batch = jobs[begin : begin + args.batch_size]
        prompts = []
        for _, name, row in batch:
            body = INTERVENTIONS[name].format(text=excerpt(row.text or "", args.max_input_chars))
            prompts.append(
                tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": body}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
        outputs = llm.generate(prompts, sampling)
        with open(outfile, "a", encoding="utf-8") as fh:
            for (doc_id, name, row), out in zip(batch, outputs):
                text = out.outputs[0].text.strip() if out.outputs else ""
                fh.write(json.dumps({
                    "doc_id": doc_id,
                    "intervention": name,
                    "model": args.model,
                    "language": row.language,
                    "country": row.country,
                    "source": row.source,
                    "title": row.title,
                    "original_words": len((row.text or "").split()),
                    "text": text,
                    "produced_words": len(text.split()),
                }, ensure_ascii=False) + "\n")
                written += 1
        rate = written / max(1e-9, time.time() - start)
        print(f"[shard {args.shard}] {written}/{len(jobs)} produced ({rate:.2f}/s)", flush=True)


if __name__ == "__main__":
    main()
