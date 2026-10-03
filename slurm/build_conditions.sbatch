#!/bin/bash
#SBATCH --job-name=mg_cond
#SBATCH --output=logs/cond_%j.out
#SBATCH --error=logs/cond_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --partition=all,class
# Usage: sbatch slurm/build_conditions.sbatch <N>
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export OMP_THREAD_LIMIT=1
cd $PROJ; echo "$(date) $SLURMD_NODENAME"
python -m src.conditions --n ${1:-300}
echo "$(date) done"
