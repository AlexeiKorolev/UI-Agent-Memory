# Source this in every shell / job (from anywhere):  source <repo>/env.sh
# Cluster-specific knobs (override before sourcing): MG_PROJ, MG_ANACONDA_MODULE
export PROJ=${MG_PROJ:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}
export MG_PROJ=$PROJ
export HF_HOME=$PROJ/hf_cache
export HF_HUB_CACHE=$PROJ/hf_cache/hub
export PIP_CACHE_DIR=$PROJ/.pip_cache
export CONDA_PKGS_DIRS=$PROJ/.conda_pkgs
export XDG_CACHE_HOME=$PROJ/.cache
export VLLM_CACHE_ROOT=$PROJ/.cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$PROJ/.cache/inductor
export TRITON_CACHE_DIR=$PROJ/.cache/triton
export PYTHONPATH=$PROJ:$PYTHONPATH
# vLLM/FlashInfer: greedy decoding needs no FlashInfer sampler (its JIT build needs curand.h and failed on Adroit);
# keep any FlashInfer JIT cache inside the project, never in $HOME.
export VLLM_USE_FLASHINFER_SAMPLER=0
export FLASHINFER_WORKSPACE_BASE=$PROJ
module purge
module load ${MG_ANACONDA_MODULE:-anaconda3/2025.12}
module load ${MG_CUDA_MODULE:-cudatoolkit/13.0} 2>/dev/null || true   # headers for any JIT kernels
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $PROJ/env 2>/dev/null || true
