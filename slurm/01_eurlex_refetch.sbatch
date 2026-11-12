#!/bin/bash
#SBATCH --job-name=aci-eurlex
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=$HOME/publicmoney/logs/eurlex_%j.out
#SBATCH --error=$HOME/publicmoney/logs/eurlex_%j.err

# Refetch the EU instruments through the Publications Office repository.
#
# The earlier attempt went through the EUR-Lex website, which now answers
# automated requests with 202 and an empty body: five thousand documents were
# requested and none arrived, without an error. The collector now uses content
# negotiation on Cellar, which is the route intended for machine access.

set -euo pipefail

ROOT=$HOME/publicmoney
cd "$ROOT"
export PYTHONPATH=$ROOT/src

echo "host=$(hostname) start=$(date -Is)"
./envs/aci/bin/python -m adminplex.collect.eurlex
echo "end=$(date -Is)"
