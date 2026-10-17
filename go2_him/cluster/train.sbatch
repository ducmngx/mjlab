#!/usr/bin/env bash
# Train a Go2 HIMLoco policy on one GPU. Submit from the repo root:
#
#   sbatch go2_him/cluster/train.sbatch
#   TASK=Go2-HIM-Rough NUM_ENVS=4096 sbatch go2_him/cluster/train.sbatch
#   sbatch --array=1-3 go2_him/cluster/train.sbatch   # one seed per task
#
# Settings (environment variables, with defaults):
#   TASK        Go2-HIM-Rough-Original (or Go2-HIM-Rough)
#   NUM_ENVS    4096
#   MAX_ITERS   20000
#   SEED        $SLURM_ARRAY_TASK_ID, else 42
#   RUN_NAME    <task>_s<seed>
#   LOGGER      wandb (needs 'uv run wandb login' or WANDB_API_KEY)
#   EXTRA_ARGS  extra flags for mjlab's trainer, e.g. "--agent.resume True"
#
# Edit the resource lines below for your cluster.
#SBATCH --job-name=go2_him
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/slurm/%x-%j.out
##SBATCH --partition=<partition>
##SBATCH --account=<account>
set -euo pipefail

# sbatch runs a copy of this script, so find the repo from the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
mkdir -p logs/slurm
export PATH="$HOME/.local/bin:$PATH"
export MUJOCO_GL=egl

TASK="${TASK:-Go2-HIM-Rough-Original}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-20000}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-42}}"
RUN_NAME="${RUN_NAME:-${TASK}_s${SEED}}"
LOGGER="${LOGGER:-wandb}"

echo "[train] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086  # EXTRA_ARGS is split into flags on purpose.
uv run python -m go2_him.scripts.train "$TASK" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.seed "$SEED" \
  --agent.run-name "$RUN_NAME" \
  --agent.logger "$LOGGER" \
  --agent.wandb-project go2_him \
  ${EXTRA_ARGS:-}
