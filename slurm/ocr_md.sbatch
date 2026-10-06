#!/bin/bash
#SBATCH --job-name=mg_ocrmd
#SBATCH --output=logs/ocrmd_%j.out
#SBATCH --error=logs/ocrmd_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=all,class
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
# OCR every screenshot of a follow-up run, then MD labels. Resumable (cached OCR JSONs are skipped).
# Usage: MG_DATASET=... MG_RUN=... sbatch --mail-user=<you> slurm/ocr_md.sbatch <N>
set -eo pipefail
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export OMP_THREAD_LIMIT=1
: "${MG_RUN:?set MG_RUN}"
cd $PROJ; echo "$(date) $SLURMD_NODENAME dataset=$MG_DATASET run=$MG_RUN"
srun python -m src.ocr --files runs/$MG_RUN/shots.txt --procs 32
python -m src.md_labels --n $1
echo "$(date) done"
