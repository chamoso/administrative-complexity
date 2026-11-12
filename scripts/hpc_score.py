"""Assess the three judgement-based dimensions on the compute cluster.

Runs one process per GPU. Each process takes a disjoint shard of the corpus,
writes its results incrementally, and can be restarted without repeating work.
The scoring model is an open-weights model held locally, so the assessment is
reproducible by anyone with the same corpus and rubric.

Usage:
    python hpc_score.py --input corpus.parquet --out scores --shard 0 --n-shards 7
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from adminplex.features.llm import (  # noqa: E402
    RUBRIC_VERSION,
    SYSTEM_PROMPT,
    build_prompt,
    parse_response,
)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--model", default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--n-shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-input-chars", type=int, default=12000)
    ap.add_argument("--max-new-tokens", type=int, default=320)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--variant-rubric", action="store_true", help="paraphrased rubric (robustness run)")
    ap.add_argument("--tag", default="main", help="label for this scoring run")
    ap.add_argument("--gpu-memory-utilisation", type=float, default=0.90)
    ap.add_argument(
        "--tensor-parallel", type=int, default=1,
        help="split one model across this many GPUs; needed for models that do not fit on one, "
             "and used instead of sharding the corpus rather than alongside it",
    )
    ap.add_argument(
        "--seed", type=int, default=20260726,
        help="sampling seed; repeated runs testing consistency must each use a different one, "
             "or the sampler reproduces the previous run exactly and the test is vacuous",
    )
    ap.add_argument(
        "--backend",
        choices=("vllm", "hf"),
        default="vllm",
        help="vllm is faster; hf uses plain transformers and depends on nothing beyond torch",
    )
    return ap.parse_args()


def load_shard(args: argparse.Namespace) -> pd.DataFrame:
    df = pd.read_parquet(args.input)
    if args.n_shards > 1:
        df = df.iloc[args.shard :: args.n_shards].copy()
    if args.limit:
        df = df.head(args.limit)
    return df


def already_done(path: Path) -> set[str]:
    done: set[str] = set()
    if path.exists():
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                try:
                    done.add(json.loads(line)["doc_id"])
                except (json.JSONDecodeError, KeyError):
                    continue
    return done


MAX_MODEL_LEN = 16384


def fit_to_context(tokenizer, build, budget: int, max_chars: int) -> str:
    """Build a prompt that fits, shortening the document until it does.

    A character budget is not a token budget. Dense legal text — citations,
    article numbers, accented words — can tokenise at under two characters per
    token, so an excerpt sized in characters can be twice the length assumed.
    Rather than guess a safe constant, the document is shortened until the
    assembled prompt measures small enough, which holds for any language and
    any tokeniser.
    """
    chars = max_chars
    for _ in range(6):
        prompt = build(chars)
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) <= budget:
            return prompt
        chars = int(chars * 0.6)
    return build(max(1000, chars))


def build_generator(args: argparse.Namespace, tokenizer):
    """Return a callable mapping a list of prompts to a list of completions.

    Two backends produce the same scores. vLLM is several times faster and is
    what the full corpus is scored with; the transformers backend needs nothing
    beyond torch, so the assessment can be reproduced on a machine where the
    inference server cannot be installed.
    """
    if args.backend == "vllm":
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=args.model,
            tensor_parallel_size=args.tensor_parallel,
            gpu_memory_utilization=args.gpu_memory_utilisation,
            max_model_len=MAX_MODEL_LEN,
            dtype="bfloat16",
            enforce_eager=False,
        )
        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=1.0 if args.temperature == 0 else 0.95,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )

        def run(prompts: list[str]) -> list[str]:
            outputs = llm.generate(prompts, sampling)
            return [o.outputs[0].text if o.outputs else "" for o in outputs]

        return run

    import torch
    from transformers import AutoModelForCausalLM

    torch.manual_seed(args.seed)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    # Decoder-only batching requires the padding on the left, or the generated
    # continuation starts from pad tokens rather than from the prompt.
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_budget = MAX_MODEL_LEN - args.max_new_tokens

    def run(prompts: list[str]) -> list[str]:
        enc = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=prompt_budget
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=0.95 if args.temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
            )
        return tokenizer.batch_decode(out[:, enc["input_ids"].shape[1] :], skip_special_tokens=True)

    return run


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"scores_{args.tag}_shard{args.shard}.jsonl"

    df = load_shard(args)
    done = already_done(outfile)
    todo = df[~df.doc_id.isin(done)].reset_index(drop=True)
    print(f"[shard {args.shard}] {len(df)} documents, {len(done)} already scored, {len(todo)} to do", flush=True)
    if todo.empty:
        return

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    generate = build_generator(args, tokenizer)

    start = time.time()
    written = 0
    for begin in range(0, len(todo), args.batch_size):
        batch = todo.iloc[begin : begin + args.batch_size]
        prompts = []
        budget = MAX_MODEL_LEN - args.max_new_tokens - 64
        for row in batch.itertuples():
            def build(chars: int, row=row) -> str:
                user = build_prompt(
                    getattr(row, "title", "") or "",
                    getattr(row, "text", "") or "",
                    getattr(row, "language", "en") or "en",
                    variant=args.variant_rubric,
                    max_chars=chars,
                )
                return tokenizer.apply_chat_template(
                    [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": user}],
                    tokenize=False,
                    add_generation_prompt=True,
                )

            prompts.append(fit_to_context(tokenizer, build, budget, args.max_input_chars))
        outputs = generate(prompts)
        with open(outfile, "a", encoding="utf-8") as fh:
            for row, raw in zip(batch.itertuples(), outputs):
                parsed = parse_response(raw)
                record = {
                    "doc_id": row.doc_id,
                    "model": args.model,
                    "rubric_version": RUBRIC_VERSION,
                    "rubric_variant": bool(args.variant_rubric),
                    "temperature": args.temperature,
                    "backend": args.backend,
                    "tag": args.tag,
                    "parsed_ok": parsed is not None,
                }
                if parsed:
                    record.update(
                        {
                            "llm_cognitive_load": parsed.get("cognitive_load"),
                            "llm_legal_knowledge": parsed.get("legal_knowledge"),
                            "llm_clarity": parsed.get("clarity"),
                            "llm_n_requirements": parsed.get("n_requirements"),
                            "llm_n_documents": parsed.get("n_documents"),
                            "justification": parsed.get("justification", ""),
                        }
                    )
                else:
                    record["raw_response"] = raw[:400]
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
        rate = written / max(1e-9, time.time() - start)
        print(
            f"[shard {args.shard}] {written}/{len(todo)} scored ({rate:.1f}/s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
