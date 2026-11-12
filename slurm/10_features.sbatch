#!/bin/bash
#SBATCH --job-name=aci-features
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=48
#SBATCH --mem=180G
#SBATCH --time=04:00:00
#SBATCH --output=$HOME/publicmoney/logs/features_%j.out
#SBATCH --error=$HOME/publicmoney/logs/features_%j.err

# Measure the twelve rule-based dimensions and the readability comparators over
# the whole corpus. Pure CPU work, one process per core.

set -euo pipefail

ROOT=$HOME/publicmoney
PY=$ROOT/envs/aci/bin/python

cd "$ROOT"
export PYTHONPATH=$ROOT/src
export OMP_NUM_THREADS=1          # the parallelism is across documents, not within
export TOKENIZERS_PARALLELISM=false

echo "host=$(hostname) cores=${SLURM_CPUS_PER_TASK:-?}"
$PY -c "
import sys, pandas as pd
sys.path.insert(0, '$ROOT/src')
from adminplex.features import pipeline
f = pipeline.run()
print('features:', f.shape)
print(f.isna().mean().sort_values(ascending=False).head(10).to_string())
"
