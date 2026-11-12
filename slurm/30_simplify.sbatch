#!/bin/bash
#SBATCH --job-name=aci-simplify
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --output=$HOME/publicmoney/logs/simplify_%j.out
#SBATCH --error=$HOME/publicmoney/logs/simplify_%j.err

# Experiments 5 and 6: strip the requirements out of a sample of procedures,
# and separately rewrite them in plain language without dropping any
# obligation. Both edits are produced for the same documents.

set -euo pipefail

ROOT=$HOME/publicmoney
PY=${PY:-$ROOT/envs/vllm/bin/python}
INPUT=${INPUT:-$ROOT/data/processed/corpus.parquet}
OUTDIR=${OUTDIR:-$ROOT/outputs/simplify}
MODEL=${MODEL:-Qwen/Qwen2.5-14B-Instruct}
SAMPLE=${SAMPLE:-1500}
NSHARDS=${NSHARDS:-2}

mkdir -p "$OUTDIR" "$ROOT/logs"
cd "$ROOT"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "host=$(hostname) sample=$SAMPLE model=$MODEL"

for shard in $(seq 0 $((NSHARDS - 1))); do
  CUDA_VISIBLE_DEVICES=$shard $PY $ROOT/scripts/hpc_simplify.py \
      --input "$INPUT" --out "$OUTDIR" --model "$MODEL" --sample "$SAMPLE" \
      --shard "$shard" --n-shards "$NSHARDS" \
      > "$ROOT/logs/simplify_shard${shard}.log" 2>&1 &
  sleep 10
done
wait

echo "--- edited texts produced ---"
wc -l "$OUTDIR"/simplified_shard*.jsonl

# The edits then go through exactly the same assessment as the originals.
$ROOT/envs/aci/bin/python $ROOT/scripts/analyse_simplification.py --write-scoring-input

echo "now run: INPUT=\$ROOT/data/processed/simplified_for_scoring.parquet \\"
echo "         OUTDIR=\$ROOT/outputs/llm_scores_simplified TAG=simplified \\"
echo "         sbatch slurm/20_score_dimensions.sbatch"
