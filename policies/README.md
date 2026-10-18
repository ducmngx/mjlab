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
| `go1_mjlab_rough_10000` | 2 / 3 / 8 % | 1 / 1 / 7 % | 0.34 / 0.80 / 1.33 | 7 % / 0.63 m |

`go2_him_rough_2000`'s velocity estimate is within 0.04-0.12 m/s of the true
base velocity. Its weak spot is stairs up (0.47 m/s at a 1.0 m/s command),
where the Go1 baseline, which sees the steps, reaches 0.61 m/s.
