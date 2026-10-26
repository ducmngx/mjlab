#!/usr/bin/env bash
# Paired fault-risk MPPI evaluation: every (course, arm, seed) one array task,
# on a MIG slice of the Hopper cluster. Submit from the repo root:
#
#   sbatch --array=0-29 failbench/cluster/mppi.sbatch
#
# Task i runs course COURSES[i / (A * SEEDS)], arm ARMS[(i / SEEDS) % A] and
# seed i % SEEDS, for A arms. Every arm of a seed gets the same robots,
# faults and fault times, so they compare pairwise. Arms (planners):
#   none  no fault risk                       -> <course>_p0_s<seed>
#   avg   expected severity at RISK_P         -> <course>_p<RISK_P>_s<seed>
#   cvar  CVaR of the worst CVAR_ALPHA        -> <course>_cvar<CVAR_ALPHA>_s<seed>
#   max   the worst fault outcome             -> <course>_max_s<seed>
#
#   sbatch --array=0-29 --export=ALL,ARMS="cvar max" failbench/cluster/mppi.sbatch
#
# Settings (environment variables, with defaults):
#   POLICY    the frozen HIM policy (model.pt; binaries stay out of git),
#             default $FAILBENCH_ROOT/policies/go2_him_rough_original_7200/model.pt
#   MODEL     impact model checkpoint, default the all-faults model
#   OUT       output directory, default $FAILBENCH_ROOT/mppi/grid
#   COURSES   "severity severity_swap cliff"
#   SEEDS     5
#   ENVS      64 robots per run
#   ARMS      "none avg"
#   RISK_P    0.1 (fault-risk prior of avg, cvar and max)
#   CVAR_ALPHA  0.025
#   EXTRA     extra run_mppi.py arguments
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=failbench_mppi
#SBATCH --output=/scratch/%u/logs/failbench_mppi-%A_%a.out
#SBATCH --error=/scratch/%u/logs/failbench_mppi-%A_%a.out
##SBATCH --mail-user=<you>@gmu.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:3g.40gb:1
#SBATCH --mem=48G
#SBATCH --export=ALL
#SBATCH --time=03:00:00
set -euo pipefail

cd "${SLURM_SUBMIT_DIR:-$(git rev-parse --show-toplevel)}"
source go2_him/cluster/env.sh

FAILBENCH_ROOT="${FAILBENCH_ROOT:-/scratch/$USER/failbench}"
POLICY="${POLICY:-$FAILBENCH_ROOT/policies/go2_him_rough_original_7200/model.pt}"
MODEL="${MODEL:-$FAILBENCH_ROOT/models/failbench_all_ground_20ep/phase1/best.pt}"
OUT="${OUT:-$FAILBENCH_ROOT/mppi/grid}"
read -r -a courses <<< "${COURSES:-severity severity_swap cliff}"
read -r -a arms <<< "${ARMS:-none avg}"
SEEDS="${SEEDS:-5}"
i="${SLURM_ARRAY_TASK_ID:-0}"
course="${courses[$(( i / (${#arms[@]} * SEEDS) ))]}"
arm="${arms[$(( (i / SEEDS) % ${#arms[@]} ))]}"
seed=$(( i % SEEDS ))
p="${RISK_P:-0.1}"
alpha="${CVAR_ALPHA:-0.025}"
case "$arm" in
  none) risk=(--mppi.fault-risk.p 0); tag="p0" ;;
  avg) risk=(--mppi.fault-risk.p "$p"); tag="p$p" ;;
  cvar) risk=(--mppi.fault-risk.p "$p" --mppi.fault-risk.measure cvar
              --mppi.fault-risk.cvar-alpha "$alpha"); tag="cvar$alpha" ;;
  max) risk=(--mppi.fault-risk.p "$p" --mppi.fault-risk.measure max); tag="max" ;;
  *) echo "[failbench] unknown arm $arm" >&2; exit 1 ;;
esac
name="${course}_${tag}_s${seed}"

echo "[failbench] $(date) host=$(hostname) job=${SLURM_JOB_ID:-local} commit=$(git rev-parse --short HEAD)"
echo "[failbench] $name -> $OUT"
mkdir -p "$OUT"
# shellcheck disable=SC2086
uv run python -m failbench.scripts.run_mppi \
  --checkpoint-file "$POLICY" \
  --model "$MODEL" \
  --output "$OUT/$name" \
  --terrain "$course" \
  --num-envs "${ENVS:-64}" \
  --duration-s 25 \
  --family priced \
  --sweep-map oracle \
  --seed "$seed" \
  --map-log-every 0 \
  --mppi.gap-quantile 0.1 \
  "${risk[@]}" \
  --mppi.weights.fault-risk 5 \
  ${EXTRA:-}
echo "[failbench] $(date) done"
