#!/bin/bash
#SBATCH --job-name=mg_lbb
#SBATCH --output=logs/lbb_%A_%a.out
#SBATCH --error=logs/lbb_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80|a40"
#SBATCH --array=0-3
# Backbone on a follow-up run: C* specs -> results/raw, action-trace specs -> results/raw_at. Resumable per episode.
# Usage: MG_DATASET=... MG_RUN=... NSHARDS=<k> sbatch --mail-user=<you> --array=0-<k-1> slurm/backbone_long.sbatch <N>
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_THREAD_LIMIT=1
: "${MG_RUN:?set MG_RUN}"
R=$PROJ/runs/$MG_RUN/results; NS=${NSHARDS:-4}
cd $PROJ; echo "$(date) $SLURMD_NODENAME dataset=$MG_DATASET run=$MG_RUN shard $SLURM_ARRAY_TASK_ID/$NS"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.backbone --n $1 --shard $SLURM_ARRAY_TASK_ID --nshards $NS --batch_eps 8 \
  --conds C0,C1,C2,C2b,C2w,C3,C4,C5,C7,C8,C8t,C8c
python -m src.backbone --n $1 --shard $SLURM_ARRAY_TASK_ID --nshards $NS --batch_eps 8 \
  --conds A0,A1,A2,A2b,A3,A5,A7,A8 --specs $R/specs_at --raw $R/raw_at
echo "$(date) done"
