#!/usr/bin/env bash
# Record the risk critic's labels for the distilled critic
# (failbench/distill/dump.py): fault-free run_mppi on a layout with the
# teacher pricing every MPPI sample each cycle. Each array task is one
# layout, both drivers; submit from the repo root on Hopper:
#
#   sbatch --array=0-27 failbench/cluster/distill_labels.sbatch           # train
#   SET=val sbatch --array=0-4 failbench/cluster/distill_labels.sbatch    # held out
#   SET=val_new sbatch --array=0-4 failbench/cluster/distill_labels.sbatch
#
# Drivers: the teacher itself at weight 1 (``ours``) and stock MPPI
# (``stock``, the critic labels at weight 0), so the labels cover the
# states both visit. Train layouts relabel half the objects' severity per
# robot (``--distill-relabel``) to severities from the training grid;
# held-out layouts (household lobby and side_crowd, the probes) keep the true
# classes (``val``) or relabel half the objects to severities never drawn in
# training (``val_new``: new entities). Shards land in
# $R/distill/labels/<set>/<driver>_<layout>_s<seed>.labels_<i>.npz (beside the run's own .json/.npz).
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=fb_dlabel
#SBATCH --output=/scratch/%u/logs/fb_dlabel-%A_%a.out
#SBATCH --error=/scratch/%u/logs/fb_dlabel-%A_%a.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=48G
#SBATCH --export=ALL
#SBATCH --time=06:00:00
set -euo pipefail
cd ~/mjlab
source go2_him/cluster/env.sh
export GIT_PYTHON_REFRESH=quiet MUJOCO_GL=egl
R=/scratch/$USER/failbench
SET=${SET:-train}
SEED=${SEED:-1}
values=train
case "$SET" in
  train)
    layouts=(severity severity_swap severity_wall severity_wall_swap severity_person
             severity_person_swap severity_person_side severity_person_one
             cluster cluster_swap gap gap_swap pass pass_swap clutter clutter_swap
             choice choice_swap choice_narrow choice_narrow_swap lone lone_mirror
             house_apartment house_loft house_garden house_townhouse house_crowd
             house_hall)
    relabel=0.5 ;;
  val)
    layouts=(house_lobby house_side_crowd house_probe_mixed house_probe_corridor
             house_probe_door)
    relabel=0.0 ;;
  val_new)
    layouts=(house_lobby house_side_crowd house_probe_mixed house_probe_corridor
             house_probe_door)
    relabel=0.5; values=held_out ;;
  *) echo "unknown set $SET"; exit 2 ;;
esac
layout=${layouts[${SLURM_ARRAY_TASK_ID:-0}]}
# House layouts follow their own routes; the scenes track a global path.
route=(--global-route)
[[ "$layout" == house_* ]] && route=()
FAULTS=(FL_hip_joint:limp FR_hip_joint:limp FL_calf_joint:lock FR_calf_joint:lock
        RL_thigh_joint:limp RR_thigh_joint:limp RL_thigh_joint:lock RR_thigh_joint:lock)
OUT=$R/distill/labels/$SET
mkdir -p "$OUT"
for driver in ours stock; do
  case "$driver" in
    ours) weight=1 ;;
    stock) weight=0 ;;
  esac
  name=$OUT/${driver}_${layout}_s${SEED}
  echo "[dlabel] $(date) host=$(hostname) $SET $driver $layout seed $SEED"
  [ -f "$name.json" ] && { echo "exists"; continue; }
  uv run python -m failbench.scripts.run_mppi \
    --checkpoint-file "$R/policies/go2_him_rough_original_7200/model.pt" \
    --output "$name" --planner nav2 --nav2-profile loose_path --sweep-map oracle \
    "${route[@]}" --risk-critic --risk.chunk 32768 --risk.faults "${FAULTS[@]}" \
    --model "$R/models/impact_v5/phase1/best.pt" --risk.weight "$weight" \
    --no-fault-aware --family nominal --duration-s 60 --terrain "$layout" \
    --seed "$SEED" --num-envs 64 --map-log-every 0 \
    --distill-dump "$name.labels" --distill-relabel "$relabel" \
    --distill-relabel-set "$values"
done
echo "[dlabel] $(date) done"
