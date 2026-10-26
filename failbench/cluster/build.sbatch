#!/usr/bin/env bash
# Rebuild FailBench sample shards from existing raw shards (no simulation),
# one terrain seed per array task, on a MIG slice of the Hopper cluster.
# Submit from the repo root:
#
#   sbatch --array=0-59 failbench/cluster/build.sbatch
#   sbatch --array=0 failbench/cluster/build.sbatch  # smoke
#
# Task i rebuilds raw/<terrain>_s<i>_r*.npz, with the terrain picked as in
# generate.sbatch.
#
# Settings (environment variables, with defaults):
#   SAMPLES_DIR  output under FAILBENCH_ROOT, default samples_v2 (the
#                originals in samples/ are kept)
#   SEED         $SLURM_ARRAY_TASK_ID, else 0
#   EXTRA        extra build.py arguments, e.g. "--held-out" (no pairs held
#                out: every fault in train, for the planner's model)
#
# Afterwards, compute the normalisation stats once (CPU is fine):
#
#   uv run python -m failbench.data.stats --root /scratch/$USER/failbench/samples_v2
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=failbench_build
#SBATCH --output=/scratch/%u/logs/failbench_build-%A_%a.out
#SBATCH --error=/scratch/%u/logs/failbench_build-%A_%a.out
##SBATCH --mail-user=<you>@gmu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=48G
#SBATCH --export=ALL
#SBATCH --time=02:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source go2_him/cluster/env.sh

FAILBENCH_ROOT="${FAILBENCH_ROOT:-/scratch/$USER/failbench}"
SAMPLES_DIR="${SAMPLES_DIR:-samples_v2}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
TERRAINS=(rough stairs stairs_up course stairs_tall stairs_up_tall)
TERRAIN="${TERRAIN:-${TERRAINS[$(( (SEED / 10) % 6 ))]}}"

echo "[failbench] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
shopt -s nullglob
raw=("$FAILBENCH_ROOT/raw/${TERRAIN}_s${SEED}_r"*.npz)
if [ ${#raw[@]} -eq 0 ]; then
  echo "[failbench] no raw shards for ${TERRAIN} seed ${SEED}" >&2
  exit 1
fi
# shellcheck disable=SC2086
uv run python -m failbench.data.build --raw "${raw[@]}" \
  --output-dir "$FAILBENCH_ROOT/$SAMPLES_DIR" ${EXTRA:-}
echo "[failbench] $(date) done"
