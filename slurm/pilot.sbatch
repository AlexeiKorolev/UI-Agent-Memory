#!/bin/bash
#SBATCH --job-name=mg_pilot
#SBATCH --output=logs/pilot_%j.out
#SBATCH --error=logs/pilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80|a40"
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_THREAD_LIMIT=1
cd $PROJ
echo "$(date) $SLURMD_NODENAME"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import torch;print('cuda ok', torch.cuda.is_available(), torch.version.cuda)"
# pilot = first 10 episodes of the stratified order
python -m src.controller --n 10 && \
python -m src.conditions --n 10 --md results/md_steps_n300.parquet --tag _pilot && \
python -m src.backbone --n 10 --batch_eps 10
echo "$(date) done"
