#!/usr/bin/env bash
# Generate and build FailBench data on one A100 80GB of the Hopper cluster,
# one terrain seed per array task. Submit from the repo root after
# `bash go2_him/cluster/setup.sh` (same virtualenv):
#
#   sbatch --array=0-59 failbench/cluster/generate.sbatch
#   sbatch --array=0 --export=ALL,ROUNDS=1 failbench/cluster/generate.sbatch  # smoke
#
# Task i uses terrain seed i and terrain preset TERRAINS[(i / 10) % 6]
# (rough, stairs, stairs_up, course, stairs_tall, stairs_up_tall), so every block of ten seeds has one
# test-terrain seed (i % 10 == 0) and one val seed (i % 10 == 1) per preset.
#
# Settings (environment variables, with defaults):
#   CHECKPOINT  the frozen HIM policy; default: model_7200.pt of the
#               Go2-HIM-Rough-Original run under $LOG_ROOT (see env.sh)
#   NUM_ENVS    4096 trials per round
#   ROUNDS      4 rounds (one raw shard each)
#   SEED        $SLURM_ARRAY_TASK_ID, else 0
#
# Writes under FAILBENCH_ROOT (default /scratch/$USER/failbench): raw shards
# to raw/, built sample shards to samples/. Afterwards, compute the
# normalisation stats once (CPU is fine):
#
#   uv run python -m failbench.data.stats --root /scratch/$USER/failbench/samples
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=failbench_gen
#SBATCH --output=/scratch/%u/logs/failbench_gen-%A_%a.out
#SBATCH --error=/scratch/%u/logs/failbench_gen-%A_%a.out
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
export MUJOCO_GL=egl

FAILBENCH_ROOT="${FAILBENCH_ROOT:-/scratch/$USER/failbench}"
CHECKPOINT="${CHECKPOINT:-$LOG_ROOT/go2_him/2026-10-17_20-44-47_Go2-HIM-Rough-Original_s42/model_7200.pt}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROUNDS="${ROUNDS:-4}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
TERRAINS=(rough stairs stairs_up course stairs_tall stairs_up_tall)
TERRAIN="${TERRAIN:-${TERRAINS[$(( (SEED / 10) % 6 ))]}}"

echo "[failbench] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
echo "[failbench] terrain=$TERRAIN seed=$SEED envs=$NUM_ENVS rounds=$ROUNDS -> $FAILBENCH_ROOT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -m failbench.scripts.generate \
  --checkpoint-file "$CHECKPOINT" \
  --output-dir "$FAILBENCH_ROOT/raw" \
  --terrain "$TERRAIN" \
  --terrain-seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --rounds "$ROUNDS"

shopt -s nullglob
raw=("$FAILBENCH_ROOT/raw/${TERRAIN}_s${SEED}_r"*.npz)
uv run python -m failbench.data.build --raw "${raw[@]}" --output-dir "$FAILBENCH_ROOT/samples"
echo "[failbench] $(date) done"
