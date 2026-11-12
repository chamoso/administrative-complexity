#!/bin/bash
#SBATCH --job-name=aci-score
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --output=$HOME/publicmoney/logs/score_%j.out
#SBATCH --error=$HOME/publicmoney/logs/score_%j.err

# Assess cognitive load, required legal knowledge and clarity over the corpus.
# One process per GPU, disjoint shards, incremental output: the job can be
# cancelled and resubmitted without losing completed work.
#
# Two GPUs, not the whole node. The cluster is shared, and at roughly eighteen
# documents a second per GPU the corpus is scored in under half an hour on two;
# asking for all seven bought nothing and queued behind every other user's job.
# Raise it with --gres and NSHARDS when the node happens to be idle.

set -euo pipefail

ROOT=$HOME/publicmoney
# vLLM scores about twenty-five times faster than the transformers backend and
# the two agree; hf remains available as BACKEND=hf PY=$ROOT/envs/score/bin/python.
BACKEND=${BACKEND:-vllm}
PY=${PY:-$ROOT/envs/vllm/bin/python}
INPUT=${INPUT:-$ROOT/data/processed/corpus.parquet}
OUTDIR=${OUTDIR:-$ROOT/outputs/llm_scores}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
TAG=${TAG:-main}
NSHARDS=${NSHARDS:-2}
BATCH=${BATCH:-16}
EXTRA=${EXTRA:-}

mkdir -p "$OUTDIR" "$ROOT/logs"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "host=$(hostname) input=$INPUT model=$MODEL tag=$TAG shards=$NSHARDS backend=$BACKEND"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

for shard in $(seq 0 $((NSHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$shard $PY $ROOT/scripts/hpc_score.py \
      --input "$INPUT" --out "$OUTDIR" --model "$MODEL" --backend "$BACKEND" \
      --shard "$shard" --n-shards "$NSHARDS" --tag "$TAG" --batch-size "$BATCH" $EXTRA \
      > "$ROOT/logs/score_${TAG}_shard${shard}.log" 2>&1 &
  sleep 10   # stagger model loading so the shards do not contend for the cache
done

wait
echo "all shards finished"
wc -l "$OUTDIR"/scores_${TAG}_shard*.jsonl
