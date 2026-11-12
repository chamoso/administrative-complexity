#!/bin/bash
#SBATCH --job-name=aci-collect
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=20:00:00
#SBATCH --output=$HOME/publicmoney/logs/collect_%j.out
#SBATCH --error=$HOME/publicmoney/logs/collect_%j.err

# Resume the sources whose crawl was cut short. Every collector skips the
# documents it has already stored, so this adds to the existing files rather
# than refetching them. The sources run one after another, not in parallel:
# these are public services and the crawl must not be the reason one of them
# slows down for anybody else.

set -euo pipefail

ROOT=$HOME/publicmoney
PY=$ROOT/envs/aci/bin/python
cd "$ROOT"
export PYTHONPATH=$ROOT/src

echo "host=$(hostname) start=$(date -Is)"

for module in ecfr eurlex sitemap; do
  echo "--- $module ---"
  $PY -m adminplex.collect.$module || echo "$module ended early; its partial output is kept"
done

echo "--- rebuilding corpus ---"
$PY -m adminplex.clean.normalise

$PY - <<'PY'
import pandas as pd
df = pd.read_parquet("data/processed/corpus.parquet")
print("corpus:", len(df), "documents")
print(df.groupby(["country", "source"]).size().to_string())
PY

echo "end=$(date -Is)"
