#!/bin/bash
#SBATCH --job-name=mg_bb
#SBATCH --output=logs/bb_%A_%a.out
#SBATCH --error=logs/bb_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80|a40"
#SBATCH --array=0-3
# Usage: sbatch slurm/backbone.sbatch <N> [conds]   (resumable; finished episodes are skipped)
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_THREAD_LIMIT=1
cd $PROJ; echo "$(date) $SLURMD_NODENAME"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.backbone --n ${1:-300} --shard $SLURM_ARRAY_TASK_ID --nshards 4 --conds ${2:-C0,C1,C2,C3,C4,C5,C6,C7,C8,C8t,C8c} --batch_eps 8
echo "$(date) done"
