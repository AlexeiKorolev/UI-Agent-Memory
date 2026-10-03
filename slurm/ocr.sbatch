#!/bin/bash
#SBATCH --job-name=mg_ocr
#SBATCH --output=logs/ocr_%j.out
#SBATCH --error=logs/ocr_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --partition=all,class
#SBATCH --mail-type=END,FAIL,TIME_LIMIT
# Usage: sbatch slurm/ocr.sbatch <file-list>   (resumable: cached JSONs are skipped)
source "$SLURM_SUBMIT_DIR/env.sh"   # submit from the repo root
export OMP_THREAD_LIMIT=1
echo "$(date) $SLURMD_NODENAME"
cd $PROJ
srun python -m src.ocr --files ${1:-data/shots_600.txt} --procs 32
echo "$(date) done"
# MD labels (cheap once OCR is cached)
python -m src.md_labels --n 300
python -m src.md_labels --n 600
echo "$(date) md labels done"
