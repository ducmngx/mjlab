#!/usr/bin/env bash
# Evaluate a trained FailBench impact model on the held-out splits (val,
# test_terrain, test_fault) on one GPU of the Hopper cluster. Submit from the
# repo root:
#
#   sbatch failbench/cluster/evaluate.sbatch                       # phase 1
#   sbatch --export=ALL,PHASE=2 failbench/cluster/evaluate.sbatch  # phase 2
#
# Settings (environment variables, with defaults):
#   PHASE       1
#   RUN         failbench_impact (model under $FAILBENCH_ROOT/models/$RUN)
#   CHECKPOINT  $FAILBENCH_ROOT/models/$RUN/phase$PHASE/best.pt
#   EXTRA       extra evaluate.py arguments, e.g. "--map-mode fused"
#
# Writes eval.json and reliability.png next to the checkpoint, in eval/.
# Inference only; a MIG slice is plenty:
#   sbatch --gres=gpu:3g.40gb:1 failbench/cluster/evaluate.sbatch
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=failbench_eval
#SBATCH --output=/scratch/%u/logs/failbench_eval-%j.out
#SBATCH --error=/scratch/%u/logs/failbench_eval-%j.out
#SBATCH --mail-type=FAIL
##SBATCH --mail-user=<you>@gmu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=64G
#SBATCH --export=ALL
#SBATCH --time=02:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source go2_him/cluster/env.sh
export PYTHONUNBUFFERED=1

FAILBENCH_ROOT="${FAILBENCH_ROOT:-/scratch/$USER/failbench}"
PHASE="${PHASE:-1}"
RUN="${RUN:-failbench_impact}"
CHECKPOINT="${CHECKPOINT:-$FAILBENCH_ROOT/models/$RUN/phase$PHASE/best.pt}"
OUT="$(dirname "$CHECKPOINT")/eval"

echo "[failbench] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
echo "[failbench] evaluating $CHECKPOINT -> $OUT"

# shellcheck disable=SC2086
uv run python -m failbench.scripts.evaluate \
  --checkpoint "$CHECKPOINT" \
  --root "$FAILBENCH_ROOT/samples" \
  --output-dir "$OUT" \
  --num-workers "${SLURM_CPUS_PER_TASK:-8}" \
  ${EXTRA:-}
echo "[failbench] $(date) done"
