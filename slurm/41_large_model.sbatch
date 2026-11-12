#!/bin/bash
#SBATCH --job-name=aci-large
#SBATCH --partition=gpu-small
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=200G
#SBATCH --time=06:00:00
#SBATCH --output=$HOME/publicmoney/logs/large_%j.out
#SBATCH --error=$HOME/publicmoney/logs/large_%j.err

# Re-run the assessment on the robustness subsample with a much larger model.
#
# The first robustness check compared a 14B model against an 8B one, which
# confounds two questions: whether the scores depend on the model, and whether
# the smaller model is simply worse at the task. A model five times the size,
# from a different family, separates them. It also settles what is wrong with
# the clarity scale: if a stronger model spreads its clarity ratings, the
# narrow distribution was a limitation of the scorer; if it does not, the
# construct does not discriminate between administrative documents.
#
# The model does not fit on one card, so it is split across four rather than
# the corpus being split across four processes.

set -euo pipefail

ROOT=$HOME/publicmoney
PY=${PY:-$ROOT/envs/vllm/bin/python}
OUTDIR=${OUTDIR:-$ROOT/outputs/llm_scores}
SUBSAMPLE=${SUBSAMPLE:-$ROOT/data/processed/robustness_sample.parquet}
MODEL=${MODEL:-meta-llama/Llama-3.1-70B-Instruct}
TAG=${TAG:-large_model}
TP=${TP:-4}

mkdir -p "$OUTDIR" "$ROOT/logs"
cd "$ROOT"
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
export TOKENIZERS_PARALLELISM=false
export VLLM_WORKER_MULTIPROC_METHOD=spawn

echo "host=$(hostname) model=$MODEL tag=$TAG tensor_parallel=$TP"
nvidia-smi --query-gpu=index,name,memory.total --format=csv

$PY "$ROOT/scripts/hpc_score.py" \
    --input "$SUBSAMPLE" --out "$OUTDIR" --model "$MODEL" --tag "$TAG" \
    --tensor-parallel "$TP" --n-shards 1 --batch-size 32 \
    --gpu-memory-utilisation 0.92

echo "done"
wc -l "$OUTDIR"/scores_${TAG}_shard0.jsonl
