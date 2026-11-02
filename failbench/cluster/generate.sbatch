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
#   TASK        the policy's task: Go2-HIM-Rough-Original (default) or
#               Go2-Plain-Rough-Original (the no-internal-model control)
#   CHECKPOINT  the frozen policy; default: HIM's model_7200.pt of the
#               Go2-HIM-Rough-Original run under $LOG_ROOT (see env.sh)
#   NUM_ENVS    4096 trials per round
#   ROUNDS      4 rounds (one raw shard each)
#   SEED        $SLURM_ARRAY_TASK_ID, else 0
#   PREFAULT_MIX  shares of the pre-fault modes walk, stand, start, slow,
#               stopped (failbench/commands.py), default "1 0 0 0 0"
#   SAMPLES     built shards, default $FAILBENCH_ROOT/samples
#   RAW         raw shards, default $FAILBENCH_ROOT/raw
#   DRIVER      planner-driven data (failbench/driver.py): nav2, dwb, or mix
#               (dwb for seeds with SEED % 5 == 4, else nav2); empty (the
#               default) for the goal follower. Driven data runs a 3.5 s
#               window, one snapshot at the fault, builds 3 s samples, on
#               the obstacle-free presets DRIVEN_TERRAINS[(i / 10) % 5]
#               (rough, stairs, stairs_up, stairs_tall, stairs_up_tall).
#   DRIVE_EXTRA driven data: more --driver.* options (failbench/driver.py
#               PlannerDriveCfg), e.g. a path library, driving styles and
#               random commits; empty (the default) is the stock planner
#               straight at the goal around random phantoms
#
# The planner-driven v3 data (mostly MPPI, some DWB), seeds 0-49:
#
#   sbatch --array=0-49 --export=ALL,ROUNDS=1,DRIVER=mix,PREFAULT_MIX="0.6 0.1 0.1 0.1 0.1",RAW=/scratch/$USER/failbench/raw_v3,SAMPLES=/scratch/$USER/failbench/samples_v3 failbench/cluster/generate.sbatch
#
# The v5 data: planners tracking stretches of planned routes on phantom
# maps (failbench/scripts/plan_paths.py; the library copied to scratch
# first), half starting shortly before a corner, per-trial driving styles,
# some random commits; new terrain seeds 50-249, in blocks of 40 (the QOS
# submit limit):
#
#   sbatch --array=50-89 --export=ALL,ROUNDS=1,DRIVER=mix,PREFAULT_MIX="0.6 0.1 0.1 0.1 0.1",DRIVE_EXTRA="--driver.library /scratch/$USER/failbench/paths_v1.npz --driver.corner-prob 0.5 --driver.style-range 0.5 2.0 --driver.random-prob 0.2",RAW=/scratch/$USER/failbench/raw_v5,SAMPLES=/scratch/$USER/failbench/samples_driven_v5 failbench/cluster/generate.sbatch
#
# The low-speed shards (faults at rest, setting off, slow, just stopped),
# seeds 60-119:
#
#   sbatch --array=60-119 --export=ALL,ROUNDS=1,PREFAULT_MIX="0.2 0.2 0.2 0.2 0.2",SAMPLES=/scratch/$USER/failbench/samples_lowspeed failbench/cluster/generate.sbatch
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
TASK="${TASK:-Go2-HIM-Rough-Original}"
NUM_ENVS="${NUM_ENVS:-4096}"
ROUNDS="${ROUNDS:-4}"
SEED="${SEED:-${SLURM_ARRAY_TASK_ID:-0}}"
read -r -a mix <<< "${PREFAULT_MIX:-1 0 0 0 0}"
SAMPLES="${SAMPLES:-$FAILBENCH_ROOT/samples}"
RAW="${RAW:-$FAILBENCH_ROOT/raw}"
DRIVER="${DRIVER:-}"
driven=()
build=()
if [ -n "$DRIVER" ]; then
  if [ "$DRIVER" = mix ]; then
    DRIVER=$([ $(( SEED % 5 )) -eq 4 ] && echo dwb || echo nav2)
  fi
  DRIVEN_TERRAINS=(rough stairs stairs_up stairs_tall stairs_up_tall)
  TERRAIN="${TERRAIN:-${DRIVEN_TERRAINS[$(( (SEED / 10) % 5 ))]}}"
  driven=(--horizon-s 3.5 --snapshot-s 0 driver:planner-drive-cfg --driver.planner "$DRIVER")
  read -r -a drive_extra <<< "${DRIVE_EXTRA:-}"
  driven+=("${drive_extra[@]}")
  build=(--horizon-s 3.0)
fi
TERRAINS=(rough stairs stairs_up course stairs_tall stairs_up_tall)
TERRAIN="${TERRAIN:-${TERRAINS[$(( (SEED / 10) % 6 ))]}}"

echo "[failbench] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
echo "[failbench] terrain=$TERRAIN seed=$SEED envs=$NUM_ENVS rounds=$ROUNDS prefault=${mix[*]} driver=${DRIVER:-none} extra=${DRIVE_EXTRA:-none} -> $SAMPLES"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

uv run python -m failbench.scripts.generate \
  --checkpoint-file "$CHECKPOINT" \
  --task "$TASK" \
  --output-dir "$RAW" \
  --terrain "$TERRAIN" \
  --terrain-seed "$SEED" \
  --num-envs "$NUM_ENVS" \
  --rounds "$ROUNDS" \
  --prefault.mix "${mix[@]}" \
  "${driven[@]}"

shopt -s nullglob
raw=("$RAW/${TERRAIN}_s${SEED}_r"*.npz)
uv run python -m failbench.data.build --raw "${raw[@]}" --output-dir "$SAMPLES" "${build[@]}"
echo "[failbench] $(date) done"
