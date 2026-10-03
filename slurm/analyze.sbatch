#!/bin/bash
#SBATCH --job-name=mg_analyze
#SBATCH --output=logs/analyze_%j.out
#SBATCH --error=logs/analyze_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --partition=all,class
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
cd $PROJ; echo "$(date) $SLURMD_NODENAME"
python -m src.analyze --n ${1:-300} ${2:+--tag $2}
echo "$(date) done"
