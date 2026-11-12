#!/usr/bin/env bash
# Everything after collection, in order, on the cluster.
#
# Each stage is skipped if its output already exists, so the script can be run
# again after a failure without repeating work. Pass --force to rebuild.
#
#   ./scripts/run_all.sh            # corpus -> features -> scoring -> results
#   ./scripts/run_all.sh --force    # ignore existing outputs

set -euo pipefail

ROOT=$HOME/publicmoney
cd "$ROOT"
PY=$ROOT/envs/aci/bin/python
export PYTHONPATH=$ROOT/src

FORCE=${1:-}
INDEX=${INDEX:-pca}

have () { [ -s "$1" ] && [ "$FORCE" != "--force" ]; }

wait_for () {   # wait_for <jobid>
  local job="$1"
  echo "waiting for job $job"
  while squeue -h -j "$job" -o "%T" 2>/dev/null | grep -qE "PENDING|RUNNING|CONFIGURING"; do
    sleep 60
  done
  echo "job $job finished"
}

echo "=============== 0. measurement tests ==============="
# The measures produce every reported number, so they are checked before a
# run rather than after it. A failure here stops the pipeline: it is cheaper to
# fix a measure now than to discover afterwards that a column was wrong.
if ! $PY -m pytest tests/ -q; then
  echo "measurement tests failed; stopping before anything is computed" >&2
  exit 1
fi

echo "=============== 1. corpus ==============="
if have data/processed/corpus.parquet; then
  echo "corpus present, skipping"
else
  $PY -m adminplex.clean.normalise
fi
$PY - <<'PY'
import pandas as pd
df = pd.read_parquet("data/processed/corpus.parquet", columns=["doc_id","country","source"])
print(f"{len(df):,} documents")
print(df.groupby(["country","source"]).size().to_string())
PY

echo "=============== 2. features ==============="
if have data/interim/features.parquet; then
  echo "features present, skipping"
else
  wait_for "$(sbatch --parsable slurm/10_features.sbatch)"
fi

echo "=============== 3. validation frames ==============="
if have data/processed/validation_parts.parquet; then
  echo "validation frames present, skipping"
else
  $PY scripts/prepare_validation.py
fi

echo "=============== 4. assessment ==============="
if compgen -G "outputs/llm_scores/scores_main_shard*.jsonl" >/dev/null && [ "$FORCE" != "--force" ]; then
  echo "main assessment present, skipping"
else
  wait_for "$(sbatch --parsable slurm/20_score_dimensions.sbatch)"
fi

# The burden-validation collections are not part of the corpus and need the
# same assessment before they can be compared with anything.
if compgen -G "outputs/llm_scores/scores_collections_shard*.jsonl" >/dev/null && [ "$FORCE" != "--force" ]; then
  echo "collection-level assessment present, skipping"
elif [ -s data/processed/validation_collections_scoring.parquet ]; then
  wait_for "$(INPUT=$ROOT/data/processed/validation_collections_scoring.parquet TAG=collections \
              sbatch --parsable slurm/20_score_dimensions.sbatch)"
fi

echo "=============== 5. index ==============="
$PY scripts/build_index.py --tag main

echo "=============== 6. burden validation ==============="
$PY scripts/validate_burden.py --level parts --index "$INDEX" || echo "part-level validation unavailable"
$PY scripts/validate_burden.py --level collections --index "$INDEX" || echo "collection-level validation unavailable"

echo "=============== 7. experiments 2,3,4,7,8 ==============="
$PY scripts/run_experiments.py --index "$INDEX"

echo "=============== 7b. typology ==============="

echo "=============== 8. experiments 5,6 ==============="
if compgen -G "outputs/simplify/simplified_shard*.jsonl" >/dev/null && [ "$FORCE" != "--force" ]; then
  echo "edited texts present, skipping generation"
else
  wait_for "$(sbatch --parsable slurm/30_simplify.sbatch)"
fi
if compgen -G "outputs/llm_scores_simplified/scores_simplified_shard*.jsonl" >/dev/null && [ "$FORCE" != "--force" ]; then
  echo "edited texts already assessed, skipping"
elif [ -s data/processed/simplified_for_scoring.parquet ]; then
  wait_for "$(INPUT=$ROOT/data/processed/simplified_for_scoring.parquet \
              OUTDIR=$ROOT/outputs/llm_scores_simplified TAG=simplified \
              sbatch --parsable slurm/20_score_dimensions.sbatch)"
fi
$PY scripts/analyse_simplification.py --index "$INDEX" || echo "simplification analysis unavailable"

echo "=============== 9. experiments 9,10 ==============="
if compgen -G "outputs/llm_scores/scores_variant_shard*.jsonl" >/dev/null && [ "$FORCE" != "--force" ]; then
  echo "robustness passes present, skipping"
else
  wait_for "$(sbatch --parsable slurm/40_robustness.sbatch)"
fi
$PY scripts/analyse_robustness.py || echo "robustness analysis unavailable"

echo "=============== 10. tables and figures ==============="
$PY scripts/make_outputs.py --index "$INDEX"

echo
echo "=============== done ==============="
ls -1 outputs/tables/*.csv 2>/dev/null | wc -l | xargs echo "tables:"
ls -1 outputs/figures/*.pdf 2>/dev/null | wc -l | xargs echo "figures:"
