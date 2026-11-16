# Failure demo videos

Kinematic replays of `run_mppi.py` logs from the household failure experiment
(`failbench/experiments/house_faults.md`). Everything drawn is logged.

- `make_video.py`: the comparison reel, `fault_demo_full.mp4`, in three parts:
  1. Nominal MPPI vs ours on the same failure at the same place (`CASES`);
  2. a 2x2 grid with CBF-0.2 and Severity-CBF-0.2 (`BASELINE_CASES`);
  3. fault-free runs where Severity-CBF-0.2's margin stalls the robot, with its
     keep-out zone (`STRICT_CASES`).
- `explain_video.py`: a single-scene explainer (default `garden_s1:16`). Time
  freezes before the real failure, the critic's eight imagined failures are
  drawn as ghost robots for Nominal's plan and ours, and then the failure
  happens.

The case lists, and the rules used to pick them, are at the top of each script.

## Logs

The failure runs' `.npz` logs are only on Hopper. Copy the ones the cases use
(the fault-free runs' logs are already in `benchmark_results/failbench/house_clean`):

```sh
mkdir -p ~/fault_logs
cat > /tmp/fault_logs.txt <<'EOF'
normal/garden_s3.npz
ours_w1/garden_s3.npz
normal/loft_s1.npz
ours_w1/loft_s1.npz
normal/apartment_s2.npz
ours_w1/apartment_s2.npz
normal/garden_s1.npz
ours_w1/garden_s1.npz
normal/crowd_s4.npz
cbf_0.2/crowd_s4.npz
cbfsem_0.2/crowd_s4.npz
ours_w1/crowd_s4.npz
normal/loft_s3.npz
cbf_0.2/loft_s3.npz
cbfsem_0.2/loft_s3.npz
ours_w1/loft_s3.npz
normal/crowd_s1.npz
cbf_0.2/crowd_s1.npz
cbfsem_0.2/crowd_s1.npz
ours_w1/crowd_s1.npz
EOF
rsync -a --files-from=/tmp/fault_logs.txt \
  hopper:/scratch/mnguy21/failbench/mppi/house_faults/ ~/fault_logs/
```

About 1 GB. A new case `<layout>_s<seed>:<env>` needs `<arm>/<layout>_s<seed>.npz`
for every arm it shows.

## Running

From the repo root:

```sh
MUJOCO_GL=egl uv run python failbench/paper_figures/fault_video/make_video.py \
  ~/fault_logs benchmark_results/failbench/house_faults \
  benchmark_results/failbench/house_clean ~/fault_video_out [main|baselines|strict ...]

MUJOCO_GL=egl uv run python failbench/paper_figures/fault_video/explain_video.py \
  ~/fault_logs benchmark_results/failbench/house_faults ~/fault_video_out/explain.mp4 \
  [<layout>_s<seed>:<env> [freeze_s]]
```

The reel takes 25-35 min, the explainer about 10.

- `make_video.py`: naming parts renders only those (and the reel holds only
  those). `REUSE=1` keeps clips already in the output folder and only
  re-stitches the reel; don't use it after changing the drawing code.
  `AZ`, `EL` and `ZOOM` move the camera.
- `explain_video.py`: `PREVIEW=1` writes the two freeze stills
  (`<out>_A.png`, `<out>_B.png`) instead of the video, for trying other scenes.

## Caveats for anyone presenting these

- The cases are hand-picked examples. Across the 2,048 paired failures,
  Nominal hits a person where ours doesn't in 63 pairs, the reverse in 26;
  show the overall table beside the videos.
- A logged "hit" is contact with a person's collision box, which is wider
  than the drawn figure; the struck box is drawn red so the contact is visible.
- In the explainer, Plan A is Nominal's chosen plan from the paired run (a few
  cm away at the same moment), not a candidate from ours' own sample batch.
- In fault-free runs, Severity-CBF-0.2 and ours fail about equally often
  overall (16 vs 14 routes); the stall examples are real but not typical.
