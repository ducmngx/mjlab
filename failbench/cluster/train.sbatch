#!/usr/bin/env bash
# Train the FailBench impact model on one GPU of the Hopper cluster, on the
# samples generate.sbatch built. Submit from the repo root:
#
#   sbatch failbench/cluster/train.sbatch                          # phase 1
#   sbatch --export=ALL,PHASE=2 failbench/cluster/train.sbatch     # phase 2
#
# Phase 1 learns from the clean map; phase 2 starts from phase 1's best.pt
# and fine-tunes on the LiDAR-realistic (fused, noisy) map. Computes
# stats.json first if it is missing. Logs to W&B (auth from ~/.netrc or
# `uv run wandb login`, never a key in this file) and tensorboard.
#
# Settings (environment variables, with defaults):
#   PHASE       1
#   RUN         failbench_impact (output under $FAILBENCH_ROOT/models/$RUN)
#   EPOCHS      10 (phase 1), 5 (phase 2)
#   BATCH       256
#   EXTRA       extra train.py arguments, e.g. "--lr 1e-4"
#
# The model is small (~3M parameters); a MIG slice is enough:
#   sbatch --gres=gpu:3g.40gb:1 failbench/cluster/train.sbatch
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=failbench_train
#SBATCH --output=/scratch/%u/logs/failbench_train-%j.out
#SBATCH --error=/scratch/%u/logs/failbench_train-%j.out
#SBATCH --mail-type=FAIL
##SBATCH --mail-user=<you>@gmu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --time=12:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source go2_him/cluster/env.sh
# Print step lines to the log as they happen, not when Python's buffer fills.
export PYTHONUNBUFFERED=1

FAILBENCH_ROOT="${FAILBENCH_ROOT:-/scratch/$USER/failbench}"
SAMPLES="$FAILBENCH_ROOT/samples"
PHASE="${PHASE:-1}"
RUN="${RUN:-failbench_impact}"
BATCH="${BATCH:-256}"
OUT="$FAILBENCH_ROOT/models/$RUN"

echo "[failbench] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
echo "[failbench] phase=$PHASE run=$RUN samples=$SAMPLES -> $OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

if [[ ! -f "$SAMPLES/stats.json" ]]; then
  uv run python -m failbench.data.stats --root "$SAMPLES"
fi

if [[ "$PHASE" == 1 ]]; then
  args=(--map-mode clean --epochs "${EPOCHS:-10}" --output-dir "$OUT/phase1")
else
  args=(--map-mode fused --epochs "${EPOCHS:-5}" --lr 1e-4 --warmup-steps 200
        --init-from "$OUT/phase1/best.pt" --output-dir "$OUT/phase2")
fi

# shellcheck disable=SC2086
uv run python -m failbench.scripts.train \
  --root "$SAMPLES" \
  --batch-size "$BATCH" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  --wandb --run-name "${RUN}_phase${PHASE}" \
  "${args[@]}" ${EXTRA:-}
echo "[failbench] $(date) done"
