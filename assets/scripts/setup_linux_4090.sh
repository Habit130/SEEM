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

source "$(conda info --base)/etc/profile.d/conda.sh"

conda deactivate >/dev/null 2>&1 || true
conda env remove -n "${ENV_NAME}" -y >/dev/null 2>&1 || true
conda env create -f "${REPO_ROOT}/environment.linux.4090.yml"
conda activate "${ENV_NAME}"

python -m pip install --upgrade pip
python -m pip install setuptools==68.2.2 wheel packaging
python -m pip install -r "${REPO_ROOT}/assets/requirements/requirements.linux.4090.txt"
python -m pip install --no-build-isolation git+https://github.com/MaureenZOU/detectron2-xyz.git
python -m pip install git+https://github.com/openai/whisper.git

cd "${REPO_ROOT}/modeling/vision/encoder/ops"
python setup.py build install

cd "${REPO_ROOT}"
python -c "import torch; import detectron2; import wandb; import whisper; import MultiScaleDeformableAttention; print('environment_ok', torch.__version__)"
