#!/bin/bash
#SBATCH --job-name=aci-global
#SBATCH --partition=short
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=$HOME/publicmoney/logs/global_%j.out
#SBATCH --error=$HOME/publicmoney/logs/global_%j.err
# Benchmark: one global model rating of administrative complexity for the
# burden-validation collections, to compare with the fifteen dimensions.
set -euo pipefail
ROOT=$HOME/publicmoney
PY=${PY:-$ROOT/envs/vllm/bin/python}
INPUT=${INPUT:-$ROOT/data/processed/validation_collections_scoring.parquet}
OUTDIR=${OUTDIR:-$ROOT/outputs/llm_scores}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
cd "$ROOT"
echo "host=$(hostname)"
for shard in 0 1; do
  CUDA_VISIBLE_DEVICES=$shard $PY $ROOT/scripts/hpc_score_global.py \
      --input "$INPUT" --out "$OUTDIR" --tag global --shard "$shard" --n-shards 2 \
      > "$ROOT/logs/score_global_shard${shard}.log" 2>&1 &
  sleep 10
done
wait
wc -l "$OUTDIR"/scores_global_shard*.jsonl
