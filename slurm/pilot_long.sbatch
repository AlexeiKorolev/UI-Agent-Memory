#!/bin/bash
#SBATCH --job-name=mg_lpilot
#SBATCH --output=logs/lpilot_%j.out
#SBATCH --error=logs/lpilot_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=03:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --constraint="gpu80|a40"
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
# Long-horizon follow-up pilot (whole pipeline on the first N episodes of runs/$MG_RUN/episodes.txt).
# Usage: MG_DATASET=odyssey|memgui MG_RUN=<name>_pilot sbatch --mail-user=<you> slurm/pilot_long.sbatch [N=10]
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_THREAD_LIMIT=1
: "${MG_RUN:?set MG_RUN}"
N=${1:-10}; R=$PROJ/runs/$MG_RUN/results
cd $PROJ; echo "$(date) $SLURMD_NODENAME dataset=$MG_DATASET run=$MG_RUN"; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -m src.ocr --files runs/$MG_RUN/shots.txt --procs 8
python -m src.md_labels --n $N
python -m src.controller --n $N
python -m src.conditions --n $N
python -m src.action_trace --n $N --long
python -m src.backbone --n $N --conds C0,C1,C2,C2b,C2w,C3,C4,C5,C7,C8,C8t,C8c --batch_eps $N
python -m src.backbone --n $N --conds A0,A1,A2,A2b,A3,A5,A7,A8 --specs $R/specs_at --raw $R/raw_at --batch_eps $N
python -m src.analyze --n $N --raw $R/raw,$R/raw_at
echo "$(date) done"
