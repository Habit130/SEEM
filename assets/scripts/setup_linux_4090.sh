#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_NAME="seem-4090"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH" >&2
  exit 1
fi

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc not found in PATH" >&2
  exit 1
fi

if [ -z "${CUDA_HOME:-}" ]; then
  export CUDA_HOME
  CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$(command -v nvcc)")")")"
fi

export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_INPUT=1
export MKL_INTERFACE_LAYER="${MKL_INTERFACE_LAYER:-LP64}"

source "$(conda info --base)/etc/profile.d/conda.sh"

conda deactivate >/dev/null 2>&1 || true
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda env create -f "${REPO_ROOT}/environment.linux.4090.yml"
set +u
conda activate "${ENV_NAME}"
set -u

python -m pip install --upgrade pip
python -m pip install setuptools==68.2.2 wheel packaging "numpy<2"
python -m pip install -r "${REPO_ROOT}/assets/requirements/requirements.linux.4090.txt"
python -m pip install --no-deps --no-build-isolation git+https://github.com/MaureenZOU/detectron2-xyz.git
python -m pip install --no-deps git+https://github.com/openai/whisper.git
python -m pip install --upgrade "numpy<2" "wandb==0.15.12"

cd "${REPO_ROOT}/modeling/vision/encoder/ops"
python setup.py build install

cd "${REPO_ROOT}"
python -c "import numpy; import torch; import detectron2; import wandb; import whisper; import MultiScaleDeformableAttention; assert int(numpy.__version__.split('.')[0]) < 2, numpy.__version__; print('environment_ok', {'numpy': numpy.__version__, 'torch': torch.__version__, 'wandb': wandb.__version__})"
