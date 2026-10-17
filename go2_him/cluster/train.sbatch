#!/usr/bin/env bash
# Train a Go2 HIMLoco policy on one A100 80GB of the Hopper cluster. Submit
# from the repo root after `bash go2_him/cluster/setup.sh`:
#
#   sbatch go2_him/cluster/train.sbatch
#   TASK=Go2-HIM-Rough sbatch go2_him/cluster/train.sbatch
#   sbatch --array=1-3 go2_him/cluster/train.sbatch   # one seed per task
#
# Settings (environment variables, with defaults):
#   TASK        Go2-HIM-Rough-Original (or Go2-HIM-Rough)
#   NUM_ENVS    4096
#   MAX_ITERS   20000
#   SEED        $SLURM_ARRAY_TASK_ID, else 42
#   RUN_NAME    <task>_s<seed>
#   LOGGER      wandb (needs `uv run wandb login` once, or WANDB_API_KEY)
#   EXTRA_ARGS  extra flags for mjlab's trainer, e.g. to resume:
#               "--agent.resume True --agent.load-run <run dir name>"
#
# Everything the job writes goes to scratch: the job log below, and under
# DATA_ROOT (default /scratch/$USER/go2_him, see env.sh) the virtualenv,
# uv cache, checkpoints/ONNX/params (rsl_rl/) and W&B files (wandb/).
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=go2_him
#SBATCH --output=/scratch/%u/logs/go2_him-%j.out
#SBATCH --error=/scratch/%u/logs/go2_him-%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
##SBATCH --mail-user=<you>@gmu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --time=3-00:00:00
set -euo pipefail

# sbatch runs a copy of this script, so find the repo from the submit dir.
cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source go2_him/cluster/env.sh
export MUJOCO_GL=egl

TASK="${TASK:-Go2-HIM-Rough-Original}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERS="${MAX_ITERS:-20000}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-42}}"
RUN_NAME="${RUN_NAME:-${TASK}_s${SEED}}"
LOGGER="${LOGGER:-wandb}"

echo "[train] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
echo "[train] DATA_ROOT=$DATA_ROOT  runs -> $LOG_ROOT/go2_him/"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# shellcheck disable=SC2086  # EXTRA_ARGS is split into flags on purpose.
uv run python -m go2_him.scripts.train "$TASK" \
  --log-root "$LOG_ROOT" \
  --env.scene.num-envs "$NUM_ENVS" \
  --agent.max-iterations "$MAX_ITERS" \
  --agent.seed "$SEED" \
  --agent.run-name "$RUN_NAME" \
  --agent.logger "$LOGGER" \
  --agent.wandb-project go2_him \
  ${EXTRA_ARGS:-}
