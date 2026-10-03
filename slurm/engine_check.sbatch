#!/bin/bash
#SBATCH --job-name=mg_engchk
#SBATCH --output=logs/engchk_%j.out
#SBATCH --error=logs/engchk_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
cd $PROJ; echo "$(date) $SLURMD_NODENAME"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python -m scripts.engine_check ${1:-4096}
