# Paths for go2_him on a cluster: everything that gets written goes under
# DATA_ROOT on scratch, not into the clone or $HOME. Sourced by setup.sh and
# train.sbatch; source it yourself before interactive `uv run` commands:
#
#   source go2_him/cluster/env.sh
#
# Each variable keeps any value already set, so you can override any of them.

export PATH="$HOME/.local/bin:$PATH"

export DATA_ROOT="${DATA_ROOT:-/scratch/$USER/go2_him}"
# Checkpoints, ONNX, params: passed to the trainer as --log-root.
export LOG_ROOT="${LOG_ROOT:-$DATA_ROOT/rsl_rl}"
# W&B writes its local run files to $WANDB_DIR/wandb/.
export WANDB_DIR="${WANDB_DIR:-$DATA_ROOT}"
# uv's package cache and the project virtualenv (several GB with CUDA torch).
export UV_CACHE_DIR="${UV_CACHE_DIR:-$DATA_ROOT/.cache/uv}"
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$DATA_ROOT/venv}"

if ! mkdir -p "$DATA_ROOT" "$LOG_ROOT" "$WANDB_DIR" "$UV_CACHE_DIR" 2>/dev/null; then
  echo "[env] Cannot create $DATA_ROOT; set DATA_ROOT to a writable scratch path." >&2
  return 1 2>/dev/null || exit 1
fi
