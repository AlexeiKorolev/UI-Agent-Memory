#!/bin/bash
# Run on the LOGIN node (needs internet). Creates the conda env in the project dir.
set -e
PROJ=${MG_PROJ:-$(cd "$(dirname "$0")/.." && pwd)}
export PIP_CACHE_DIR=$PROJ/.pip_cache CONDA_PKGS_DIRS=$PROJ/.conda_pkgs XDG_CACHE_HOME=$PROJ/.cache
module purge; module load ${MG_ANACONDA_MODULE:-anaconda3/2025.12}
source $(conda info --base)/etc/profile.d/conda.sh
conda create -y -p $PROJ/env python=3.11
conda activate $PROJ/env
pip install -U pip
pip install vllm==0.30.0            # pulls torch 2.13.0+cu130, transformers 5.18.0
pip install qwen-vl-utils datasets pillow pandas pyarrow scipy statsmodels rapidfuzz pytesseract matplotlib \
            huggingface_hub hf_transfer remotezip pytest opencv-python-headless
# OCR uses the system tesseract 4.1.1 (/usr/bin/tesseract, eng); nothing to install.
