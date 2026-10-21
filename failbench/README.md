# failbench

The data pipeline for FailBench: a dataset for learning the impact of a joint
failure on a quadruped. Each sample pairs four inputs with five seconds of
outcomes.

Inputs:
- a robot-centred elevation map;
- the trunk state;
- a command sequence;
- a kinodynamic fault vector.

Outcomes:
- the velocity deviation from the command;
- the path;
- whether and when the robot falls;
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
| `faults.py` | the 72-d fault vector, its sampler and `inject_kinodynamic_fault` |
| `commands.py` | `KnotVelocityCommand`: pre-fault command, then 1 s knots |
| `env_cfg.py` | the data env: HIM task (play, no randomization), faults, commands, sensors |
| `sensing.py` | head-LiDAR visibility (occlusion, field of view, self-occlusion), map noise |
| `scripts/generate.py` | simulate trials into raw shards |
| `data/build.py` | raw shards to sample shards |
| `data/stats.py` | normalisation statistics |
| `data/dataset.py` | `FailBenchDataset` |
| `scripts/inspect_samples.py` | render samples for checking by eye |
| `cluster/generate.sbatch` | Hopper array job: generate, then build |

## Trials

Each trial runs this way:
- The env walks a constant pre-fault command on a heat_bench fault terrain:
  `rough`, `stairs`, `stairs_up` or `course`.
- At a time sampled in 3–5 s, the fault fires and the command switches to a
  sequence of 1 s knots, with 7 knots covering the 7 s logged after the
  fault.
- Nothing resets.

**Fault vector.** There are 6 parameters per joint, each relative to nominal:

| Parameter | Effect |
|---|---|
| `torque` | scales the force limit (0 = limp) |
| `damping`, `friction` | scale joint damping and friction loss |
| `kp`, `kd` | scale the PD gains |
| `range` | half-width of the allowed range around the fault-time angle (0 = locked) |

**Families.** Each trial draws one:

| Family | Share | What it does |
|---|---|---|
| nominal | 15% | no fault |
| severe | 35% | one joint locked, limp, or with a torque limit of at most 0.3 |
| continuous | 30% | one joint, each parameter perturbed with probability 0.5 |
| multi-mild | 20% | 2–4 joints, mildly perturbed |

The network gets the vector through `encode_fault`, where a healthy joint is
`(1, 0, 0, 1, 1, 1)`.

**Command styles.** The post-fault knots follow one of:
- stop (30%), where every knot is zero;
- hold the pre-fault command (20%);
- random knots (50%), with vx in [-1, 1.5] and vy, wz in [-1, 1].

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
| `cmd_seq` | 50×3 | the command for each bin |
| `fault` | 72 | the encoded fault vector |

**Targets:**

| Field | Shape | Contents |
|---|---|---|
| `dv` | 50×3 | (vx, vy, wz) minus the command, body frame |
| `pose` | 50×3 | (x, y, yaw) in the start frame |
| `upright` | 50 | no trunk or head contact and no tilt over 60° yet |
| contacts | per bin | non-foot contacts on the same 64×64 grid, stored as sparse rows and rasterised by the Dataset |
| `contact_offmap` | 50 | a contact landed off the map in that bin |

**Splits** (`split`):

| Split | What goes there |
|---|---|
| val | terrain seeds ending in 1 |
| test_terrain | terrain seeds ending in 0 |
| test_fault | every severe fault on `RL_calf_joint`, never trained on |

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
