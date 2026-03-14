#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEIGHTS_DIR="${REPO_ROOT}/weights"
WEIGHT_NAME="xdecoder_focalt_last.pt"
WEIGHT_PATH="${WEIGHTS_DIR}/${WEIGHT_NAME}"
WEIGHT_URL="https://huggingface.co/xdecoder/X-Decoder/resolve/main/${WEIGHT_NAME}"

mkdir -p "${WEIGHTS_DIR}"

if [ -s "${WEIGHT_PATH}" ]; then
  echo "Using existing init weight: ${WEIGHT_PATH}"
  exit 0
fi

TMP_PATH="${WEIGHT_PATH}.download"
rm -f "${TMP_PATH}"

download_with_curl() {
  curl -L --fail --output "${TMP_PATH}" "${WEIGHT_URL}"
}

download_with_wget() {
  wget -O "${TMP_PATH}" "${WEIGHT_URL}"
}

download_with_python() {
  local python_cmd=""
  if command -v python3 >/dev/null 2>&1; then
    python_cmd="python3"
  elif command -v python >/dev/null 2>&1; then
    python_cmd="python"
  else
    return 1
  fi

  "${python_cmd}" -c "import pathlib, sys, urllib.request; urllib.request.urlretrieve(sys.argv[1], sys.argv[2]); pathlib.Path(sys.argv[2]).stat()" "${WEIGHT_URL}" "${TMP_PATH}"
}

cleanup() {
  rm -f "${TMP_PATH}"
}

trap cleanup EXIT

if command -v curl >/dev/null 2>&1; then
  download_with_curl
elif command -v wget >/dev/null 2>&1; then
  download_with_wget
else
  download_with_python
fi

if [ ! -s "${TMP_PATH}" ]; then
  echo "Failed to prepare init weight: ${WEIGHT_URL}" >&2
  exit 1
fi

mv "${TMP_PATH}" "${WEIGHT_PATH}"
trap - EXIT
echo "Prepared init weight: ${WEIGHT_PATH}"
