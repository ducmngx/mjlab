#!/usr/bin/env bash
# One-time setup of this clone on a cluster: installs uv if missing, builds
# the project's .venv from uv.lock (Python 3.13 + CUDA 12.8 PyTorch), and
# checks that the GPU driver can run it. Run from anywhere inside the repo.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if ! command -v uv >/dev/null 2>&1; then
  echo "[setup] Installing uv to ~/.local/bin"
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

# Keep uv's package cache off a small home quota when scratch exists.
if [[ -n "${SCRATCH:-}" && -z "${UV_CACHE_DIR:-}" ]]; then
  export UV_CACHE_DIR="$SCRATCH/.cache/uv"
  echo "[setup] UV_CACHE_DIR=$UV_CACHE_DIR (add this to your shell profile)"
fi

echo "[setup] uv sync --extra cu128"
uv sync --extra cu128

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
else
  echo "[setup] nvidia-smi not found: run this on a GPU node to check CUDA."
fi

uv run python - <<'PY'
import torch
import warp as wp

print(f"[setup] torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
  print(f"[setup] GPU 0: {torch.cuda.get_device_name(0)}")
wp.init()
print(f"[setup] warp {wp.config.version}, CUDA devices: {wp.get_cuda_device_count()}")
PY

echo "[setup] Done. Next: 'uv run wandb login', then"
echo "        'sbatch go2_him/cluster/train.sbatch'."
