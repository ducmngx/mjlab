# failbench

The data pipeline for FailBench: a dataset for learning the impact of a joint
failure on a quadruped. Each sample pairs four inputs with five seconds of
outcomes.

Inputs:
- a robot-centred elevation map;
- the trunk state;
- a command sequence;
- a joint fault vector.

Outcomes:
- the velocity deviation from the command;
- the path;
- whether and when the robot falls, and when its trunk or head hits
  something;
- where its body (anything but the feet) touches the terrain.

The robot is a Go2 running the frozen HIMLoco policy from `go2_him/`
(`policies/go2_him_rough_original_7200`). The policy is never retrained. An
impact model trained on this data is meant to sit inside an MPPI planner that
picks `cmd_vel` for the policy after a fault.

## Pipeline

```
generate.py  (sim, GPU)    raw shard per round: 50 Hz logs, clean maps, contacts
build.py     (GPU)         sample shards: 4 samples per trial, 10 Hz targets,
                           LiDAR visibility masks
stats.py     (CPU)         stats.json over the train split
dataset.py                 torch Dataset: noisy LiDAR map, contact grids
```

| File | Contents |
|---|---|
| `faults.py` | the 24-d fault vector, its sampler and `inject_kinodynamic_fault` |
| `goals.py` | per-trial goal by terrain, and a spawn heading facing it |
| `commands.py` | `GoalVelocityCommand`: goal follower before the fault, then the plan |
| `plans.py` | `SegmentPlan`: open-loop post-fault command plans (shared with the planner later) |
| `env_cfg.py` | the data env: HIM task (play, no randomization), faults, commands, sensors |
| `sensing.py` | head-LiDAR visibility (occlusion, field of view, self-occlusion), map noise |
| `scripts/generate.py` | simulate trials into raw shards |
| `data/build.py` | raw shards to sample shards |
| `data/stats.py` | normalisation statistics |
| `data/dataset.py` | `FailBenchDataset` |
| `scripts/inspect_samples.py` | render samples for checking by eye |
| `scripts/play_trial.py` | replay a logged trial as a video, with the goal, plan and drift drawn in |
| `cluster/generate.sbatch` | Hopper array job: generate, then build |

## Trials

Each trial runs this way:
- **Terrain.** heat_bench's `rough`, `stairs`, `stairs_up` and `course`,
  or FailBench's `stairs_tall` and `stairs_up_tall`: ten 0.25 m treads with
  8–18 cm risers down the difficulty rows (18–36°, 0.8–1.8 m flights), past
  the 10 cm the policy trained on.
- **Goal.** The robot spawns facing a goal that fits the terrain. On the
  stair tiles (pyramids) it is the outer edge, so the robot climbs out of
  the pit on `stairs_up` and walks down off the top platform on `stairs`.
  On the course it is the lane end; on rough, 5 m away in a random
  direction.
- **Before the fault.** The robot stands for 1 s, then a goal follower
  drives it (closed loop): turn toward the goal, walk at a cruise speed
  drawn in 0.5–1 m/s.
- **The fault.** On rough and the course it fires at a time drawn in
  3–5 s. On the stair tiles it fires when the robot reaches a sampled
  distance from the pyramid centre, from just before the first step to the
  last, and no earlier than 2.5 s, so nearly every stair fault hits on the
  flight. The fault is known the moment it fires; there is no detection
  step.
- **After the fault.** An open-loop plan drives the robot (below).
- Nothing resets.

**Fault vector.** There are 2 parameters per joint:

| Parameter | Effect |
|---|---|
| `torque` | scales the force limit (1 = nominal, 0 = limp) |
| `lock` | 1 = the joint jams at its fault-time angle |

**Families.** Each trial draws one, and at most one joint fails:

| Family | Share | What it does |
|---|---|---|
| nominal | 15% | no fault |
| lock | 40% | one joint jammed, like a seized gearbox |
| weak | 45% | one joint's torque limit scaled into [0, 0.5], and to exactly 0 (limp) a quarter of the time |

Gain, damping and friction faults were tried and dropped: HIM absorbed them,
and their drift was barely above the nominal noise floor. The network gets
the vector as is (`encode_fault`), where a healthy joint is `(1, 0)`.

**Post-fault plans** (`plans.py`). A plan is drawn at the fault from the
command the robot had then (`c0`) and its pose, and is never adjusted after
that, like a candidate plan MPPI scores:
- hold (30%): `c0` for the whole window, the "nobody reacts" plan;
- segments (70%): segments of exponential length (mean 1.5 s), each with a
  target twist. vx stops, keeps the previous speed or takes a new one in
  0.2–1.2 m/s, and never goes negative. wz pulls toward the goal plus
  noise, and vy is usually 0. vx ramps at up to 1 m/s²; vy and wz ramp over
  0.5 s.

On the course, plans follow the lane instead: the feasible path is the lane
centre (obstacles stand 0.55–1.0 m to the sides), so the turn rate is
recomputed every step to pull the plan's predicted pose back onto the centre
line, plus a small noise, and obstacle contacts come from the fault's drift
rather than from the plan. Nothing in the stack avoids obstacles: HIM's actor
is blind, and the planner that does (MPPI) comes later.

Elsewhere the goal pull steers toward a fixed bearing (fault position to goal), so a
plan that passes the goal doesn't circle back. It compares that bearing with
the plan's own integrated yaw, never the robot's, so a plan can't encode the
drift the model predicts. The segment generator
follows ZLIK's random-walk driver (Pokhrel et al., arXiv:2602.12385).
Planar commands of 0.2 m/s or less are zeroed, as in HIMLoco's sampler.

## Samples

A trial gives one sample at the fault and one each at +0.5 s, +1 s and +2 s.
Each covers 5 s at 10 Hz (50 bins).

**Inputs:**

| Field | Shape | Contents |
|---|---|---|
| `map_clean` | 64×64 | terrain height relative to the trunk; 0.05 m cells, trunk-centred, yaw-aligned |
| `known_single`, `known_fused` | 64×64 | what the head LiDAR could see (see below) |
| `state` | 33 | HIM's velocity estimate (hardware has no true velocity), angular velocity, projected gravity, joint positions and velocities |
| `state_true_vel` | 3 | true base velocity, for an ablation |
| `him_latent` | 16 | HIM's latent |
| `history` | 20×36 | the last 2 s at 10 Hz, oldest first: `state` plus the command |
| `cmd_seq` | 50×3 | the plan: the command for each bin |
| `fault` | 24 | the encoded fault vector |

**Targets:**

| Field | Shape | Contents |
|---|---|---|
| `dv` | 50×3 | (vx, vy, wz) minus the command, body frame |
| `pose` | 50×3 | (x, y, yaw) in the start frame |
| `cmd_path` | 50×3 | the plan's commanded path in the same frame; drift = `pose − cmd_path` |
| `upright` | 50 | not fallen yet: no tilt over 60° and the trunk at least 0.15 m above the ground under it |
| `collision` | 50 | the trunk or head touched terrain or an obstacle in that bin, fallen or not |
| contacts | per bin | non-foot contacts on the same 64×64 grid, stored as sparse rows and rasterised by the Dataset |
| `contact_offmap` | 50 | a contact landed off the map in that bin |

**Splits** (`split`):

| Split | What goes there |
|---|---|
| val | terrain seeds ending in 1 |
| test_terrain | terrain seeds ending in 0 |
| test_fault | a locked `RL_calf_joint`, never trained on |

## Sensor realism

The simulator's map is cast straight down from above the trunk, so it sees
everything. A Go2 builds its map from the Unitree L1 LiDAR on its head.
`sensing.py` keeps only the cells that LiDAR could return:
- **Terrain occlusion.** The map is ray-marched along the line of sight, so
  the treads behind a step edge going down are hidden.
- **Field of view.** A hemisphere about the sensor axis.
- **Self-occlusion.** Ground under the trunk is hidden.

There are two variants:
- **Single.** The current pose only.
- **Fused.** The union over the current pose and the poses 0.5 s, 1 s and
  1.5 s earlier, the way an elevation mapper accumulates scans. This is the
  default.

The Dataset then adds 2 cm height noise, a 3 cm per-map offset and 5% cell
dropout. This happens afresh each time a sample is loaded, unless a seed is
given. The map the network sees has 2 channels: height (0 where unknown) and
the known mask. `map_mode="clean"` gives the full map without noise, as an
ablation.

> The mount pose comes from Unitree's `go2_description` URDF (`radar_joint`),
> and the 90° half-angle field of view is an approximation of the L1's
> 360°×90°. Check both against Unitree's datasheet before quoting them.

## Running

Locally, as a smoke test:

```sh
uv run python -m failbench.scripts.generate \
  --checkpoint-file policies/go2_him_rough_original_7200/model.pt \
  --output-dir /tmp/fb/raw --num-envs 64 --terrain stairs --terrain-seed 2
uv run python -m failbench.data.build --raw /tmp/fb/raw/stairs_s2_r0.npz \
  --output-dir /tmp/fb/samples
uv run python -m failbench.data.stats --root /tmp/fb/samples
uv run python -m failbench.scripts.inspect_samples --root /tmp/fb/samples \
  --output-dir /tmp/fb/inspect
```

On Hopper (see `go2_him/README.md` for the one-time setup):

```sh
sbatch --array=0 --export=ALL,ROUNDS=1 failbench/cluster/generate.sbatch  # smoke
sbatch --array=0-39 failbench/cluster/generate.sbatch
```

Everything goes under `/scratch/$USER/failbench`.

**Size.** A built sample takes about 19 KB, so 1M samples is about 19 GB. A
raw shard of 4096 trials is about 420 MB.
