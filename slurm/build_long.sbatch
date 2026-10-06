#!/bin/bash
#SBATCH --job-name=mg_lcond
#SBATCH --output=logs/lcond_%j.out
#SBATCH --error=logs/lcond_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=all,class
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
# Condition specs (C*) and action-trace specs incl. A2b/A3/A5 for a follow-up run.
# Usage: MG_DATASET=... MG_RUN=... sbatch --mail-user=<you> slurm/build_long.sbatch <N>
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export OMP_THREAD_LIMIT=1
: "${MG_RUN:?set MG_RUN}"
cd $PROJ; echo "$(date) $SLURMD_NODENAME dataset=$MG_DATASET run=$MG_RUN"
python -m src.conditions --n $1
python -m src.action_trace --n $1 --long
echo "$(date) done"
