#!/usr/bin/env bash
# The household fault experiment (failbench/experiments/house_faults.md): one
# task per (layout, arm), each running every seed in FAULT_SEEDS. Submit from
# the repo root on Hopper, one array task per layout x arm:
#
#   ARMS="normal ours_w0.5 ours_w1" sbatch --array=0-23 failbench/cluster/house_faults.sbatch
#
# A new baseline reuses the existing arms' results: run only its own arm,
#
#   ARMS="inflate_0.3" sbatch --array=0-7 failbench/cluster/house_faults.sbatch
#
# Every arm shares the planner (Nav2 MPPI, loose_path), the map, the seeds
# and therefore the faults and their places. Results land in
# $R/mppi/house_faults/<arm>/<layout>_s<seed>.{json,npz}; existing ones are
# skipped.
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=fb_hfault
#SBATCH --output=/scratch/%u/logs/fb_hfault-%A_%a.out
#SBATCH --error=/scratch/%u/logs/fb_hfault-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=32G
#SBATCH --export=ALL
#SBATCH --time=06:00:00
set -euo pipefail
cd ~/mjlab
source go2_him/cluster/env.sh
export GIT_PYTHON_REFRESH=quiet MUJOCO_GL=egl
R=/scratch/$USER/failbench
OUT=$R/mppi/${FAULT_DIR:-house_faults}
layouts=(apartment loft garden townhouse lobby crowd hall side_crowd)
read -r -a arms <<< "${ARMS:-normal ours_w0.5 ours_w1}"
i=${SLURM_ARRAY_TASK_ID:-0}
l=${layouts[$(( i % ${#layouts[@]} ))]}
arm=${arms[$(( i / ${#layouts[@]} ))]}
SEEDS=${FAULT_SEEDS:-1 2 3 4}
FAULTS=(FL_hip_joint:limp FR_hip_joint:limp FL_calf_joint:lock FR_calf_joint:lock
        RL_thigh_joint:limp RR_thigh_joint:limp RL_thigh_joint:lock RR_thigh_joint:lock)
# Every arm builds the critic so its throws are logged; only ours prices them.
critic=(--risk-critic --risk.chunk 32768 --risk.faults "${FAULTS[@]}"
        --model "$R/models/impact_v5/phase1/best.pt")
case "$arm" in
  normal) extra=(--risk.weight 0) ;;
  ours_w*) extra=(--risk.weight "${arm#ours_w}") ;;
  inflate_*) extra=(--risk.weight 0 --nav2.costmap.margin "${arm#inflate_}") ;;
  semantic_*) extra=(--risk.weight 0 --nav2.costmap.margin "${arm#semantic_}"
                     --nav2.costmap.severity-ref 25) ;;
  *) echo "unknown arm $arm"; exit 2 ;;
esac
mkdir -p "$OUT/$arm"
for s in $SEEDS; do
  name="$OUT/$arm/${l}_s${s}"
  echo "[hfault] $(date) host=$(hostname) $arm $l seed $s"
  [ -f "$name.json" ] && { echo "exists"; continue; }
  uv run python -m failbench.scripts.run_mppi \
    --checkpoint-file "$R/policies/go2_him_rough_original_7200/model.pt" \
    --output "$name" --planner nav2 --nav2-profile loose_path --sweep-map oracle \
    "${critic[@]}" "${extra[@]}" --no-fault-aware \
    --family priced --fault-place-frac 0.15 0.85 --fault-timeout-s 30 \
    --fault-window-s 3.0 --post-fault replan \
    --terrain "house_$l" --seed "$s" --num-envs 64 --duration-s 40 --map-log-every 10
done
echo "[hfault] $(date) done"
