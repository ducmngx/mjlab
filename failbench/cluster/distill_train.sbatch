#!/usr/bin/env bash
# Train the distilled risk critic (failbench/scripts/train_distill.py) on
# the labels distill_labels.sbatch recorded. Submit from the repo root:
#
#   sbatch failbench/cluster/distill_train.sbatch
#
# Logs to W&B (auth from ~/.netrc, never a key in this file) and
# tensorboard. Settings (environment variables, with defaults):
#   RUN     distill_v1 (output under $R/distill/models/$RUN)
#   EPOCHS  20
#   EXTRA   extra train_distill.py arguments
#SBATCH --partition=gpuq
#SBATCH --qos=gpu
#SBATCH --job-name=fb_dtrain
#SBATCH --output=/scratch/%u/logs/fb_dtrain-%j.out
#SBATCH --error=/scratch/%u/logs/fb_dtrain-%j.out
#SBATCH --mail-type=FAIL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:A100.80gb:1
#SBATCH --mem=96G
#SBATCH --export=ALL
#SBATCH --time=24:00:00
set -euo pipefail
cd ~/mjlab
source go2_him/cluster/env.sh
export GIT_PYTHON_REFRESH=quiet
R=/scratch/$USER/failbench
RUN=${RUN:-distill_v1}
L=$R/distill/labels
uv run python -m failbench.scripts.train_distill \
  --train "$L/train/*.labels_*.npz" --val "$L/val/*.labels_*.npz" \
  --impact-model "$R/models/impact_v5/phase1/best.pt" \
  --output-dir "$R/distill/models/$RUN" --epochs "${EPOCHS:-20}" \
  --wandb --run-name "$RUN" ${EXTRA:-}
