# go2_him

A port of **HIMLoco** (Hybrid Internal Model) to mjlab, trained on Unitree's
**Go2**. It is meant to be the baseline locomotion policy for the fault-impact
study. HIM's internal model predicts the body velocity, so after a joint
fault you can compare the actual motion with what the policy expected.

There are no thermal, battery or current models here. This is a plain Go2
velocity task.

## What HIM is

The policy is blind: the actor never sees terrain.

- **Estimator.** Reads the last 6 steps of 45-d proprioception: command,
  angular velocity, projected gravity, joint position, joint velocity and
  last action. It outputs:
  - an estimate of the base velocity (3);
  - a unit-norm 16-d latent of how the robot is responding.
- **Actor.** Acts on the current observation plus those two estimates.
- **Training the estimator.**
  - The velocity part is trained with an MSE loss.
  - The latent is trained with a SwAV-style swapped-prediction loss (32
    prototypes, Sinkhorn) against an embedding of the *next* step's
    privileged observation.
  - It has its own Adam optimizer and never gets PPO gradients.
- **Critic.** Sees the 45-d proprioception without noise, the true base
  velocity, the external force on the base, and 187 height points (238-d).

| File | Contents |
|---|---|
| `robot/` | Unitree's Go2 MJCF and its adaptation to mjlab |
| `him/modules.py` | `HIMEstimator`, `HIMActor` (rsl_rl model), `sinkhorn` |
| `him/algorithm.py` | `HIMPPO` (rsl_rl `PPO` + estimator updates) |
| `him/env_cfg.py` | the task config: observations, rewards, randomization, curricula |
| `him/commands.py` | HIMLoco's velocity command sampling and command curriculum |
| `him/terrains.py` | HIMLoco's terrain mix |
| `him/mdp.py` | HIMLoco reward, observation, event and curriculum terms mjlab lacks |
| `him/rl_cfg.py` | network sizes and PPO settings |
| `him/tasks.py` | task registration |

## Tasks

Both tasks use HIMLoco's observations, rewards, terrain, commands,
curricula, resets, randomization, disturbances and training procedure. They
differ only in three settings that trade fidelity for GPU memory or concern
the robot itself:

| | `Go2-HIM-Rough` | `Go2-HIM-Rough-Original` |
|---|---|---|
| Go2 collision shapes | capsules/spheres | the MJCF's cylinders and boxes |
| `ccd_iterations` | 100 | 500 (mjlab's Go1 rough setting) |
| PD gains | Kp 20, Kd 0.5 (Unitree's Go2 configs) | Kp 40, Kd 1 (HIMLoco's Go1 config) |
| GPU | 2048 envs in 8 GB, about 13.5 s per iteration | needs a large GPU; 128 envs take about 10.7 s per iteration on an 8 GB card |

`Go2-Plain-Rough-Original` is `Go2-HIM-Rough-Original` without the
internal model: the same environment, critic and PPO settings, with a plain
MLP actor on the same 6-frame observation history and no velocity or latent
estimator. It is the control for what HIM itself contributes (FailBench
compares the two). Train it the same way, e.g.
`TASK=Go2-Plain-Rough-Original MAX_ITERS=7200 sbatch go2_him/cluster/train.sbatch`.

Other combinations come from
`go2_him_env_cfg(collision=..., gains=..., ccd_iterations=...)`.
`--env.sim.mujoco.ccd-iterations N` overrides the iteration count from the
command line.

## Train and play

```sh
# Full run (cluster), as close to HIMLoco as possible. The wrappers register
# the tasks, then hand off to mjlab's train/play, so every mjlab flag works.
uv run python -m go2_him.scripts.train Go2-HIM-Rough-Original \
  --env.scene.num-envs 4096

# Watch a checkpoint.
uv run python -m go2_him.scripts.play Go2-HIM-Rough \
  --checkpoint-file logs/rsl_rl/go2_him/<run>/model_<N>.pt
```

- **Logs** go to `logs/rsl_rl/go2_him/`. They include the estimator's
  `estimation` and `swap` losses, plus an ONNX policy on every save. The
  ONNX input is the flat `actor` observation, so the estimator is part of
  the exported model.
- **Logger.** The default logger is mjlab's, Weights & Biases. Add
  `--agent.logger tensorboard` to keep logs local.

## Training on a SLURM cluster

The scripts in `go2_him/cluster/` are set up for GMU's Hopper cluster:
- partition `gpuq` and QOS `gpu`;
- one `A100.80gb` GPU and up to 3 days per job;
- job logs in `/scratch/$USER/logs/`.

No hand-made virtual environment is needed. uv builds one from the
committed `uv.lock`, including Python 3.13 and the CUDA 12.8 PyTorch wheels.

**Everything that gets written goes to scratch.** The clone holds only
code and meshes, about 60 MB, so it can stay in `$HOME`. `env.sh` sets the
paths; each can be overridden:

| Path | Contents |
|---|---|
| `/scratch/$USER/logs/go2_him-<jobid>.out` | job stdout and stderr |
| `$DATA_ROOT` (default `/scratch/$USER/go2_him`) | root for everything below |
| `$DATA_ROOT/venv` | the virtualenv (`UV_PROJECT_ENVIRONMENT`) |
| `$DATA_ROOT/.cache/uv` | uv's package cache |
| `$DATA_ROOT/rsl_rl/go2_him/<date>_<run>/` | checkpoints `model_<N>.pt` (every 100 iterations, about 6.5 MB each), ONNX policy, the exact configs and the git state |
| `$DATA_ROOT/wandb/` | W&B's local run files (curves and checkpoints also upload to the W&B project `go2_him`) |

```sh
git clone git@github.com:ducmngx/mjlab.git ~/mjlab && cd ~/mjlab
git checkout heat-bench-actuator-health
bash go2_him/cluster/setup.sh     # installs uv if needed, builds the venv on
                                  # scratch, CUDA check
source go2_him/cluster/env.sh     # in any shell where you run `uv run ...`
uv run wandb login                # once
sbatch go2_him/cluster/train.sbatch
```

- **Script settings.** `train.sbatch` trains `Go2-HIM-Rough-Original` with
  4096 envs for 20000 iterations. Override with environment variables
  (`TASK`, `NUM_ENVS`, `MAX_ITERS`, `SEED`, `RUN_NAME`, `LOGGER`,
  `EXTRA_ARGS`). To get job email at your own address, uncomment
  `--mail-user` in the script.

  ```sh
  TASK=Go2-HIM-Rough sbatch go2_him/cluster/train.sbatch    # lighter task
  sbatch --array=1-3 go2_him/cluster/train.sbatch           # seeds 1, 2, 3
  ```

- **Resuming after the 3-day limit.** Submit again with
  `--agent.resume True` and the run directory name:

  ```sh
  EXTRA_ARGS="--agent.resume True --agent.load-run <date>_<run>" \
    sbatch go2_him/cluster/train.sbatch
  ```

- **Watching a checkpoint.** Copy a `model_<N>.pt` back and play it with
  `go2_him.scripts.play`.
- **Scratch may be purged.** Copy checkpoints you want to keep off scratch;
  W&B also keeps them.
- **Out of memory.** If `Go2-HIM-Rough-Original` at 4096 envs runs out of
  GPU memory, lower `NUM_ENVS`, or pass
  `EXTRA_ARGS="--env.sim.mujoco.ccd-iterations 200"`.
- **One GPU per job.** mjlab's multi-GPU mode (`--gpu-ids all`) runs
  worker processes that have not been tested with the `go2_him` task
  registration.

## Fault-impact data

`heat_bench/scripts/fault_impact.py` runs joint-fault trials (lock, free,
or no fault) with a trained Go2 HIM checkpoint. Pass the task it was trained
on:

```sh
uv run python -m heat_bench.scripts.fault_impact --task Go2-HIM-Rough-Original \
  --checkpoint-file <run>/model_<N>.pt --terrain course --map-size-m 8 \
  --num-envs 512 --output-file go2_impact.npz
```

- **Environment:** the play variant of the task, with the robot's randomized
  properties off (payload, COM, Kp/Kd, foot friction) so the fault labels
  are clean.
- **Commands:** HIMLoco's command sampler is swapped for one constant
  command per trial.
- **FREE faults:** no thermal stack is needed. FREE zeroes the failed
  joint's motor force range directly; LOCK and LIMIT narrow the joint's
  range as before.
- **HIM fields** in the `.npz`, alongside the usual velocity error, path,
  contact and impact-map data:
  - `dense_him_vel_est`: the estimator's base velocity, in m/s;
  - `dense_him_latent`: its 16-d latent;
  - `dense_dv_est_b`: actual velocity minus the previous step's estimate.
    HIM predicts the velocity after the action taken on that history, so
    it is compared with the next step.

## Settings

Values come from HIMLoco's `legged_robot_config.py` and `go1_config.py`
unless noted.

**Observations**

| Group | Contents |
|---|---|
| Actor | 45-d one-step proprioception × 6 frames (270-d), with noise: angular velocity ±0.2, gravity ±0.05, joint position ±0.01, joint velocity ±1.5 |
| Critic | the same 45 terms with noise (HIMLoco's privileged observation reuses the noisy proprioception), then noise-free base linear velocity ×2, body-frame base disturbance force (N), and height points `clip(z_base − 0.5 − z_terrain, ±1) × 5` with ±0.1 m noise; layout in `env_cfg.CRITIC_SLICES` |

Scales: angular velocity ×0.25, joint velocity ×0.05, command ×(2, 2, 0.25).

**Rewards** (scaled by dt, like legged_gym)

| Term | Weight |
|---|---|
| tracking_lin_vel (σ 0.25) | 1.0 |
| tracking_ang_vel (σ 0.25) | 0.5 |
| lin_vel_z | −2 |
| orientation | −0.2 |
| base_height (target 0.30 m) | −1 |
| ang_vel_xy | −0.05 |
| foot_clearance (target −0.20 m in the base frame) | −0.01 |
| action_rate | −0.01 |
| smoothness | −0.01 |
| dof_acc | −2.5e−7 |
| joint_power | −2e−5 |

`dof_acc` uses the finite difference of joint velocity over a control step,
as legged_gym does. The episode ends when the trunk or head touches the
ground (over 1 N), or after 20 s.

**Terrain:** 8 m tiles, 10 difficulty rows, with a curriculum.

| Type | Share | Row 0 → row 9 |
|---|---|---|
| smooth slope | 0.1 | 0 → 0.4 rise per run, half of them going down |
| rough slope | 0.2 | the same, with ±5 cm noise |
| stairs up | 0.3 | riser 0.05 → 0.23 m, 0.31 m tread |
| stairs down | 0.3 | same risers, descending |
| discrete blocks | 0.1 | 20 blocks of 1–2 m, 0.05 → 0.25 m high |

**Resets, randomization and disturbances** (training only)

| Item | Setting |
|---|---|
| Base reset | ±1 m in x/y, yaw unchanged, base linear and angular velocity ±0.5 per axis |
| Joint reset | default angles ×0.5–1.5, zero velocity |
| Foot friction | 0.2–1.25 |
| Payload | −1 to +2 kg |
| Base COM | ±5 cm |
| Kp, Kd | ×0.9–1.1, redrawn at every reset |
| Motor strength | none: HIMLoco samples it but never applies it |
| Action delay | each control step, the first 0–3 physics substeps still use the previous action |
| Base force | every 8 control steps, all envs together: ±30 N per axis in the body frame |
| Pushes | every 16 s, all envs together: world x/y base velocity set to U(−1, 1) m/s |

**Commands**
- The first 20% of envs draw forward speed from the curriculum range; the
  rest draw ±1 m/s. Fast envs (|vx| ≥ 1) get no sideways command.
- Every env draws sideways speed ±1 m/s and a heading; yaw rate follows the
  heading, clipped to ±1 rad/s.
- Planar commands of 0.2 m/s or less are zeroed. Commands are resampled
  every 10 s.
- **Command curriculum:** once per episode length, if both groups track
  forward speed at over 80% of the maximum reward, the curriculum range
  widens by 0.2 m/s each side, up to ±2 m/s.
- **Terrain curriculum:** a robot moves up a level after walking more than
  half a tile (4 m). It moves down after walking less than half its
  commanded distance. Past the top level it goes to a random level.

**PPO:**
- networks: actor and critic 512-256-128 ELU, no input normalization;
- learning rate 1e-3, adaptive (KL 0.01); 5 epochs, 4 minibatches; 100
  steps per env;
- entropy 0.01; γ 0.99; λ 0.95.

## Differences from HIMLoco

These remain in both tasks:

- **Robot.** HIMLoco has no Go2 config; the gains are a task setting (see
  Tasks).
- **Disturbance force duration.** Isaac Gym applies HIMLoco's force for
  one 5 ms physics substep. mjlab events can only hold a force for a whole
  control step, so the force is divided by the decimation, giving the same
  impulse over 20 ms. The critic still sees the full sampled force.
- **Base-height reward.** It averages terrain height over the 187-point
  height scan. HIMLoco uses its own 63-point base-height pattern.
- **Separate noise draws.** The critic's proprioception gets its own noise
  draw. In HIMLoco it is the same noisy vector the actor sees.
- **Terrain grid.** mjlab's curriculum layout gives each terrain type one
  column; HIMLoco spreads 20 columns by proportion. The share of robots per
  type is the same.
- **History layout.** mjlab stacks history term by term, oldest first.
  `HIMActor` reorders it to HIMLoco's frame-by-frame, newest-first layout
  (`modules.frame_major_index`). A test checks that the newest frame
  equals the critic's noise-free proprioception in a running env.

Matched to HIMLoco after first being approximated:
- terminal observations for the estimator, saved before mjlab resets an
  env;
- estimator steps interleaved with the PPO minibatches;
- the performance-based command curriculum and the high-velocity group;
- the legged_gym terrain curriculum;
- the reset, push and delay semantics;
- no motor-strength randomization;
- noisy critic observations;
- finite-difference `dof_acc`.

## The Go2 model

`robot/go2.xml` and `robot/assets/` are copied unchanged from Unitree's
[unitree_mujoco](https://github.com/unitreerobotics/unitree_mujoco)
(`unitree_robots/go2`, commit 1eb6642, BSD-3-Clause, `robot/LICENSE`).
`robot/go2_constants.py:get_spec` adapts it in code:
- drops the XML's motors, sensors, keyframe and solver options;
- zeroes the geom margins;
- renames `base_link` to `trunk`;
- names the collision geoms in mjlab's Go1 style (`trunk_collision`,
  `{leg}_thigh_collision`, `{leg}_foot_collision`, …);
- with `collision="capsule"`, converts the head and hip cylinders to
  spheres and the thigh boxes and calf cylinders to capsules of the same
  radius and length. mujoco_warp needs much less memory for these against
  heightfields, and gives them more than one contact point.
- adds a `{leg}` site at each foot and the IMU sensors the velocity task
  reads.

The joint order is the MJCF body order: FL, FR, RL, RR. The Unitree SDK's
motor order is FR, FL, RR, RL (`go2_constants.LEGS`), so deploying on the
real robot needs that index map.

## Reference

Junfeng Long, Zirui Wang, Quanyi Li, Jiawei Gao, Liu Cao, Jiangmiao Pang.
*Hybrid Internal Model: Learning Agile Legged Locomotion with Simulated
Robot Response.* ICLR 2024. arXiv:2312.11460.
Code: github.com/InternRobotics/HIMLoco (BSD-3-Clause).
