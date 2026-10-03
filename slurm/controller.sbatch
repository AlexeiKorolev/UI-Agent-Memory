#!/bin/bash
#SBATCH --job-name=mg_ctrl
#SBATCH --output=logs/ctrl_%A_%a.out
#SBATCH --error=logs/ctrl_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80|a40"
#SBATCH --array=0-3
# Usage: sbatch slurm/controller.sbatch <N>    (resumable; finished episodes are skipped)
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_THREAD_LIMIT=1
cd $PROJ; echo "$(date) $SLURMD_NODENAME"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.controller --n ${1:-300} --shard $SLURM_ARRAY_TASK_ID --nshards 4
echo "$(date) done"
