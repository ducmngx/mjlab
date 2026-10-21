# Policies

Curated checkpoints, copied out of `logs/rsl_rl/` (which holds every
intermediate save). One folder per policy:

```
policies/<name>/
  model.pt         rsl_rl checkpoint (--checkpoint-file)
  policy.onnx      exported actor (observation history -> joint targets)
  params/          env.yaml / agent.yaml the run was trained with
```

`model.pt` and `policy.onnx` are gitignored (6-7 MB each); `params/` and this
README are tracked. Copy the binaries between machines by hand, or pull them
from W&B for cluster runs.

| Name | Task | Robot | Observations | Iterations | Source run |
|---|---|---|---|---|---|
| `go2_him_rough_2000` | `Go2-HIM-Rough` | Go2 (unitree_mujoco MJCF, capsules) | blind: 6-frame proprio history, HIM estimator | 2000 | `logs/rsl_rl/go2_him/2026-10-17_09-44-11_local_2000` (`model_1999.pt`), commit 9b0fc57 |
| `go2_him_rough_original_7200` | `Go2-HIM-Rough-Original` | Go2 (original collision shapes, HIMLoco gains Kp 40 / Kd 1, 500 CCD iterations) | as `go2_him_rough_2000` | 7200 (hit the 12 h job limit) | Hopper job, `/scratch/mnguy21/go2_him/rsl_rl/go2_him/2026-10-17_20-44-47_Go2-HIM-Rough-Original_s42` (`model_7200.pt`), seed 42, 4096 envs |
| `go1_mjlab_rough_10000` | `Mjlab-Velocity-Rough-Unitree-Go1` | Go1 (mjlab) | true base velocity + height scan | 10000 | `logs/rsl_rl/go1_velocity/2026-09-22_22-12-37` (`model_9999.pt`), commit c2e1e06 |

## Usage

```sh
# Go2 HIMLoco: viewer, and the fault impact pipeline.
uv run python -m go2_him.scripts.play Go2-HIM-Rough \
  --checkpoint-file policies/go2_him_rough_2000/model.pt
uv run python -m heat_bench.scripts.fault_impact --task Go2-HIM-Rough \
  --checkpoint-file policies/go2_him_rough_2000/model.pt --terrain course

# mjlab's Go1 baseline (heat_bench's default task).
uv run python -m heat_bench.scripts.fault_impact \
  --checkpoint-file policies/go1_mjlab_rough_10000/model.pt --terrain course
```

## Evaluation (2026-10-17)

Deterministic, no noise, pushes or domain randomization; straight commands
held for 6 s; every terrain type x 10 difficulty rows; a fall is a tilt over
60 degrees. "vx" is the measured forward speed while upright.

| | Falls, HIM terrains (0.5/1.0/1.5 m/s) | Falls, mjlab terrains | vx, HIM terrains | Course, no fault: fell / median drift |
|---|---|---|---|---|
| `go2_him_rough_2000` | 1 / 0 / 2 % | 0 / 0 / 2 % | 0.38 / 0.80 / 1.18 | 6 % / 0.41 m |
| `go2_him_rough_original_7200` | 0 / 0 / 2 % | 0 / 1 / 0 % | 0.41 / 0.93 / 1.36 | 4 % / 0.15 m |
| `go1_mjlab_rough_10000` | 2 / 3 / 8 % | 1 / 1 / 7 % | 0.34 / 0.80 / 1.33 | 7 % / 0.63 m |

`go2_him_rough_2000`'s velocity estimate is within 0.04-0.12 m/s of the true
base velocity. Its weak spot is stairs up (0.47 m/s at a 1.0 m/s command),
where the Go1 baseline, which sees the steps, reaches 0.61 m/s.
`go2_him_rough_original_7200` (evaluated with 2 instead of 4 repeats per
tile, for GPU memory) reaches 0.85 m/s there and is the best policy overall;
its velocity estimate is within 0.08-0.17 m/s. Run it with
`--task Go2-HIM-Rough-Original`, since it was trained with that task's
collision shapes and gains.

The same run's `model_2000.pt` (not kept here) matches `go2_him_rough_2000`
closely: 0 / 1 / 2 % falls and 0.33 / 0.83 / 1.24 m/s on HIM terrains,
0.42 m/s on stairs up, 2 % / 0.26 m on the course. So most of the gain comes
from training longer, not from the original collision shapes and gains.
