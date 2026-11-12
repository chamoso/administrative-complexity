#!/bin/bash
#SBATCH --job-name=aci-robust
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=$HOME/publicmoney/logs/robust_%j.out
#SBATCH --error=$HOME/publicmoney/logs/robust_%j.err

# Experiments 9 and 10. The assessment is repeated with the rubric paraphrased,
# with a different model, and three times unchanged at a temperature above zero.
# All runs are on the same subsample: the question is whether the scores move,
# and that is answered on a few thousand documents as well as on all of them.

set -euo pipefail

ROOT=$HOME/publicmoney
PY=${PY:-$ROOT/envs/vllm/bin/python}
OUTDIR=${OUTDIR:-$ROOT/outputs/llm_scores}
SUBSAMPLE=${SUBSAMPLE:-$ROOT/data/processed/robustness_sample.parquet}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
ALT_MODEL=${ALT_MODEL:-meta-llama/Llama-3.1-8B-Instruct}
NSHARDS=${NSHARDS:-2}
N=${N:-4000}

mkdir -p "$OUTDIR" "$ROOT/logs"
cd "$ROOT"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# One subsample, drawn once, used by every run below.
$ROOT/envs/aci/bin/python - "$SUBSAMPLE" "$N" <<'PY'
import sys, pandas as pd
out, n = sys.argv[1], int(sys.argv[2])
df = pd.read_parquet("data/processed/corpus.parquet")
df.sample(n=min(n, len(df)), random_state=20260726).to_parquet(out, index=False)
print(f"robustness subsample: {min(n, len(df))} documents -> {out}")
PY

run_pass () {
  local tag="$1"; shift
  local model="$1"; shift
  echo "=== pass ${tag} (model ${model}) ==="
  for shard in $(seq 0 $((NSHARDS - 1))); do
    CUDA_VISIBLE_DEVICES=$shard $PY $ROOT/scripts/hpc_score.py \
        --input "$SUBSAMPLE" --out "$OUTDIR" --model "$model" \
        --shard "$shard" --n-shards "$NSHARDS" --tag "$tag" "$@" \
        > "$ROOT/logs/score_${tag}_shard${shard}.log" 2>&1 &
    sleep 8
  done
  wait
  wc -l "$OUTDIR"/scores_${tag}_shard*.jsonl
}

# Experiment 9: the rubric, then the model.
run_pass variant   "$MODEL"     --variant-rubric
run_pass alt_model "$ALT_MODEL"

# Experiment 10: the identical configuration, three times, with sampling on.
# At temperature zero there is nothing to vary, so consistency is only a
# meaningful question above it.
for i in 1 2 3; do
  run_pass "repeat${i}" "$MODEL" --temperature 0.3 --seed $((20260726 + i))
done

echo "all robustness passes finished"
