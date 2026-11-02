# FailBench stage 1: DWB port and planner-driven fault data — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A batched port of Nav2's DWB local planner beside the existing Nav2 MPPI port, and a data-generation mode where a real planner (MPPI or DWB) drives each trial before the fault, around phantom costmap obstacles, and at the fault one of that cycle's candidates runs open loop for 3 s.

**Architecture:** `failbench/mppi/dwb.py` mirrors `failbench/mppi/nav2.py`'s planner interface (`step(costmap, resolution, goal, speed, context)` returning the same keys) and shares its motion model. `failbench/phantoms.py` samples per-trial world boxes and rasterises them into robot-centred height maps that only the planner sees. `failbench/driver.py` wraps a planner + phantoms into a command source that `GoalVelocityCommand` uses instead of its goal follower and segment plan when `driver` is set; `generate.py` / `build.py` gain the fields and a 3 s window.

**Tech Stack:** Python, PyTorch (batched, GPU), mjlab (`ManagerBasedRlEnv`), tyro CLIs, pytest.

**Spec:** `/home/aaron/.claude/plans/help-me-build-this-dreamy-toucan.md` (approved design). This plan covers its run-order step 1 only. Stage 2 (model 3 s / 10 Hz), 3 (Hopper data + training), 4 (risk critic) and 5 (scenes, evaluation) get their own plans.

## Global Constraints

- Always `uv run`; `make check` with `PYRIGHT_PYTHON_GLOBAL_NODE=0` before declaring a task done.
- 2-space indentation, 88 columns, no local imports unless circular; tests are functions, no classes.
- **No commits** unless the user asks; each task ends with a checkpoint (`make check` + its tests), not a commit.
- Never commit data or binaries; nothing logged to W&B from local runs; cluster writes only under `/scratch/mnguy21`.
- Stock planners: DWB's critics and default scales are Nav2's `nav2_params.yaml` values; only Go2 limits and the shared motion model differ (documented in the config docstring, as `Nav2MppiCfg` does).
- Go2 limits (from `Nav2MppiCfg`): vx 0–1.2 m/s, |vy| ≤ 0.5 m/s, |wz| ≤ 1.0 rad/s; the motion model's `vy_gain` 0.7 and `deadband` 0.2.
- Planning cycle 10 Hz (`model_dt` 0.1, 5 control steps at 50 Hz); candidate horizon 3 s = 30 steps.
- Phantom obstacles exist only in the planner's costmap, never in physics.
- Existing behaviour unchanged when `driver` is None (default): old data mode, seeded draws and all current tests stay as they are.

## Review Focus

1. A goal behind the robot or a phantom across the straight line: DWB must still produce a valid command (turn in place / stop), not NaN or an exception — test in Task 2.
2. Every candidate colliding (robot boxed in by phantoms): both planners stop (`cmd` 0, `nav2_fail` True) rather than send a lethal command — test in Task 2 (DWB); MPPI already does.
3. A fault firing between planning cycles: the commit must happen on the fault step itself and the first committed command must run on that step — test in Task 5.
4. Pre-fault "stand"/"stopped" modes with the driver: the robot must stand (command 0) and the planner must not wind up a warm start it then fires on resume — test in Task 5.
5. Phantoms on top of the spawn or goal: never within 0.6 m of either, so trials are not born colliding or unreachable — test in Task 4.

---

### Task 1: Shared motion model and DWB trajectory generation

**Files:**
- Modify: `failbench/mppi/nav2.py` (module-level motion model; `Nav2Mppi._predict` / `_integrate` delegate; `step()` also returns `"cv"`)
- Create: `failbench/mppi/dwb.py`
- Test: `failbench/tests/test_dwb.py`

**Interfaces:**
- Produces (nav2.py): `accel_limited(cmds: Tensor[n,K,T,3], v0: Tensor[n,3], cfg: Nav2MppiCfg) -> Tensor[n,K,T,3]` (body of today's `_predict`), `integrate(v: Tensor[n,K,T,3], pose0: Tensor[n,3], cfg: Nav2MppiCfg) -> Tensor[n,K,T,3]` (body of `_integrate`). `Nav2Mppi.step` output gains `"cv": Tensor[n,K,T,3]` (the sampled command sequences).
- Produces (dwb.py): `DwbCfg` (frozen dataclass), `velocity_grid(v0: Tensor[n,3], cfg: DwbCfg) -> Tensor[n,S,3]`, `rollout(twist: Tensor[n,S,3], v0: Tensor[n,3], cfg: DwbCfg) -> tuple[Tensor[n,S,T,3] traj, Tensor[n,S,T,3] v]` with `T = round(sim_time / model_dt)`.

`DwbCfg` fields and values (Nav2 `nav2_params.yaml` DWB block; Go2 limits where marked):

| field | value | note |
|---|---|---|
| `vx_min, vx_max` | 0.0, 1.2 | Go2 |
| `vy_max` | 0.5 | Go2 (Nav2 diff-drive: 0) |
| `wz_max` | 1.0 | |
| `ax, ay, az` (acc) | 2.5, 2.5, 3.2 | ay Go2 (Nav2 0) |
| `dx, dy, dz` (decel) | -2.5, -2.5, -3.2 | |
| `vx_samples, vy_samples, wz_samples` | 20, 5, 20 | |
| `sim_time` | 1.7 | |
| `sim_period` | 0.05 | 1 / Nav2 controller_frequency 20 |
| `model_dt` | 0.1 | FailBench's 10 Hz cycle |
| `min_speed_xy`, `min_speed_theta` | 0.0, 0.0 | |
| `xy_goal_tolerance`, `trans_stopped_velocity` | 0.25, 0.25 | |
| critic scales | BaseObstacle 0.02, PathAlign 32.0, GoalAlign 24.0, PathDist 32.0, GoalDist 24.0, RotateToGoal 32.0 | |
| `forward_point_distance` | 0.1 | PathAlign, GoalAlign |
| `rotate_slowing_factor`, `rotate_lookahead_time` | 5.0, -1.0 | RotateToGoal |
| `oscillation_reset_dist`, `oscillation_reset_angle` | 0.05, 0.2 | |
| `vy_gain`, `deadband` | 0.7, 0.2 | shared motion model |
| `costmap` | `CostmapCfg()` | same as Nav2 MPPI |
| `plan_steps` | 50 | as `Nav2MppiCfg` |
| `commit_steps` | 30 | 3 s of a twist held, the risk critic's / data's candidate |

- [ ] **Step 1: Write the failing tests** in `failbench/tests/test_dwb.py`:

```python
def test_the_grid_is_the_dynamic_window_around_the_current_speed():
  cfg = DwbCfg()
  v0 = torch.tensor([[0.6, 0.0, 0.0]])
  g = velocity_grid(v0, cfg)
  assert g.shape == (1, 20 * 5 * 20, 3)
  # One control period of acceleration either side, inside the limits.
  assert torch.isclose(g[0, :, 0].max(), torch.tensor(0.6 + 2.5 * 0.05))
  assert torch.isclose(g[0, :, 0].min(), torch.tensor(0.6 - 2.5 * 0.05))
  assert g[0, :, 1].abs().max() <= 2.5 * 0.05 + 1e-6
  stop = velocity_grid(torch.zeros(1, 3), cfg)
  assert (stop[0, :, 0] >= 0).all()


def test_a_rollout_ramps_to_its_twist_within_the_acceleration_limit():
  cfg = DwbCfg()
  traj, v = rollout(torch.tensor([[[1.2, 0.0, 0.0]]]), torch.zeros(1, 3), cfg)
  assert v.shape[2] == round(1.7 / 0.1)
  assert (v[0, 0, 1:, 0].diff() <= 2.5 * 0.1 + 1e-6).all()
  assert torch.isclose(v[0, 0, -1, 0], torch.tensor(1.2))
  assert traj[0, 0, -1, 0] > 1.0 and traj[0, 0, -1, 1].abs() < 1e-6
```

- [ ] **Step 2: Run** `uv run pytest failbench/tests/test_dwb.py -v` — expected FAIL (module missing).
- [ ] **Step 3: Move the motion model** out of `Nav2Mppi` into `accel_limited` / `integrate` (module level, same maths, reading `ax_*`, `ay_*`, `az_max`, `model_dt`, `vy_gain`, `deadband` from any cfg with those attributes); `_predict` / `_integrate` call them. Add `"cv": cv` to `step()`'s return dict.
- [ ] **Step 4: Implement `DwbCfg`, `velocity_grid`, `rollout`** in `failbench/mppi/dwb.py`. Window per axis: `[v0 + decel * sim_period, v0 + acc * sim_period]` clipped to the limits, `linspace` with the sample counts, Cartesian product. `rollout` holds the twist and reaches it under the per-step acceleration limit (`acc * model_dt`), then `integrate` from pose (0, 0, 0). Module docstring: what is Nav2's (`dwb_core::DWBLocalPlanner`, `StandardTrajectoryGenerator`, `XYThetaIterator`) and what differs, like `nav2.py`'s.
- [ ] **Step 5: Run** `uv run pytest failbench/tests/test_dwb.py failbench/tests/test_nav2.py -v` — expected PASS (Nav2 tests unchanged by the refactor).
- [ ] **Step 6: Checkpoint** `PYRIGHT_PYTHON_GLOBAL_NODE=0 make check`.

### Task 2: DWB critics and planner

**Files:**
- Modify: `failbench/mppi/dwb.py`
- Modify: `failbench/tests/test_nav2.py` (generalise `_drive` to take a planner factory; keep its current call sites working)
- Test: `failbench/tests/test_dwb.py`

**Interfaces:**
- Consumes: Task 1; `inflate`, `nav2_costmap`, `_lookup`, `LETHAL`, `INSCRIBED`, `NO_INFORMATION`, `Nav2Mppi._footprint` logic (move `_footprint` to a module-level `footprint_points(pts, cfg: CostmapCfg)` in nav2.py if needed by both).
- Produces: `class Dwb` with `__init__(cfg: DwbCfg, n: int, device, extra_critics=())`, the runner interface of `Nav2Mppi` (`reset(ids, cmd=None)`, `track`, `advance() -> Tensor[n,3]`, `branch_names()`, attributes `substeps = 5`, `plan [n, plan_steps, 3]`), and `step(costmap, resolution, goal, speed, context=None) -> dict` with exactly `Nav2Mppi.step`'s keys; `"weight"` is one-hot on the chosen twist, `"cv"` is each twist held for `commit_steps` (`[n, S, 30, 3]`), `"traj"` the `sim_time` rollouts. Extra critics receive the same `data` dict keys as in Nav2 (`traj`, `v`, `cv`, `pose0`, `path`, `goal`, `costmap`, `resolution`, plus `context`).

Critics (each `scale * raw`, summed; DWB's `ObstacleFootprint` behaviour folded into BaseObstacle with the Go2 rectangle):
- BaseObstacle: sum of costmap cost at the rollout poses; any footprint point at `LETHAL` (or `NO_INFORMATION`) makes the twist invalid (excluded).
- PathDist / GoalDist: distance (in cells, `/ resolution`) from the last pose to the nearest point of the pruned straight-line path (`Nav2Mppi._path` logic) / to the path's end.
- PathAlign / GoalAlign: the same, for the point `forward_point_distance` ahead of the last pose along its heading; PathAlign is 0 within `xy_goal_tolerance` of the goal.
- RotateToGoal: 0 unless within `xy_goal_tolerance`; there, twists with planar speed above `trans_stopped_velocity` are invalid and the cost is the |yaw error| to the goal heading at the last pose.
- Oscillation: a twist that flips the sign of vx, vy or wz against the last sent command is invalid until the robot has moved `oscillation_reset_dist` or turned `oscillation_reset_angle` since the flip was armed (Nav2's `OscillationCritic`).
- All invalid: `cmd` 0, `nav2_fail` True, plan zeros.

- [ ] **Step 1: Write the failing tests** (reuse `_local_heights` and a generalised `_drive(boxes, goal_w, cycles, n, seed, make=...)` from `test_nav2.py`):

```python
def test_dwb_walks_to_a_goal_in_the_open_within_its_limits():
  poses, cmds, hit = _drive([], (4.0, 0.0), cycles=70, make=_dwb)
  assert not hit
  assert ((poses[-1, :, :2] - torch.tensor([4.0, 0.0])).norm(dim=-1) < 0.5).all()
  assert (cmds[..., 0] >= -1e-6).all() and (cmds[..., 0] <= 1.2 + 1e-6).all()
  assert (cmds[..., 1].abs() <= 0.5 + 1e-6).all()


def test_dwb_threads_an_opening_without_touching_it():
  wall = [(2.0, 2.3, 0.45, 3.0), (2.0, 2.3, -3.0, -0.45)]
  poses, _, hit = _drive(wall, (5.0, 0.0), cycles=80, make=_dwb)
  assert not hit
  assert ((poses[-1, :, :2] - torch.tensor([5.0, 0.0])).norm(dim=-1) < 0.5).all()


def test_dwb_turns_toward_a_goal_behind_it():
  poses, cmds, _ = _drive([], (-3.0, 0.0), cycles=40, make=_dwb)
  assert torch.isfinite(cmds).all()
  assert (poses[-1, :, 0] < -1.0).all()


def test_dwb_stops_when_every_twist_collides():
  box = [
    (-1.0, 1.0, 0.25, 0.6),
    (-1.0, 1.0, -0.6, -0.25),
    (0.4, 0.8, -0.6, 0.6),
    (-0.8, -0.4, -0.6, 0.6),
  ]
  cfg = DwbCfg()
  planner = Dwb(cfg, 1, "cpu")
  cmap = nav2_costmap(_local_heights(box, torch.zeros(1, 3)), RES, cfg.costmap)
  out = planner.step(cmap, RES, torch.tensor([[3.0, 0.0]]), torch.zeros(1, 3))
  assert bool(out["nav2_fail"][0]) and (out["cmd"] == 0).all()


def test_dwb_returns_each_twist_held_for_three_seconds_to_extra_critics():
  seen = {}

  def probe(data):
    seen["cv"] = data["cv"]
    return torch.zeros(data["cv"].shape[:2])

  planner = Dwb(DwbCfg(), 1, "cpu", extra_critics=[probe])
  cmap = torch.zeros(1, 160, 160)
  planner.step(cmap, RES, torch.tensor([[3.0, 0.0]]), torch.zeros(1, 3))
  cv = seen["cv"]
  assert cv.shape[2] == 30 and torch.equal(cv[:, :, 0], cv[:, :, -1])
```

- [ ] **Step 2: Run** `uv run pytest failbench/tests/test_dwb.py -v` — expected FAIL.
- [ ] **Step 3: Implement `Dwb`** per the critic list above; the robot-frame conventions, `pose0` latency step and path are `Nav2Mppi.step`'s. `plan`: the chosen twist held for `plan_steps`; `path`: its rollout integrated.
- [ ] **Step 4: Run** `uv run pytest failbench/tests/test_dwb.py failbench/tests/test_nav2.py -v` — expected PASS.
- [ ] **Step 5: Checkpoint** `make check`.

### Task 3: DWB as a `run_mppi` planner

**Files:**
- Modify: `failbench/scripts/run_mppi.py` (`planner: Literal["legacy", "nav2", "dwb"]`, `dwb: DwbCfg` field; the Nav2 branch takes either class; extra critics wire the same way)

**Interfaces:**
- Consumes: Task 2's `Dwb`, `DwbCfg`.

- [ ] **Step 1: Add the option**; every `isinstance(planner, Nav2Mppi)` check becomes `isinstance(planner, (Nav2Mppi, Dwb))`.
- [ ] **Step 2: Smoke run locally** (fault-free, 8 robots, severity corridor):
  `MUJOCO_GL=egl PYTHONPATH=$PWD uv run python failbench/scripts/run_mppi.py --checkpoint-file policies/go2_him_rough_original_7200/model.pt --output <scratchpad>/dwb_smoke --planner dwb --terrain severity --family nominal --num-envs 8 --duration-s 25`
  Expected: finishes; JSON summary shows most robots reached, no NaN commands.
- [ ] **Step 3: Checkpoint** `make check`.

### Task 4: Phantom obstacles

**Files:**
- Create: `failbench/phantoms.py`
- Test: `failbench/tests/test_driver.py`

**Interfaces:**
- Produces: `PhantomCfg` (frozen): `count: tuple[int, int] = (0, 6)`, `none_prob: float = 0.2`, `size_m: tuple[float, float] = (0.2, 1.2)` (each side), `lateral_m: float = 1.5` (centre within this of the spawn→goal line), `clear_m: float = 0.6` (from spawn and goal), `height_m: float = 0.8`, `max_count: int = 6`.
  `sample_phantoms(start_w: Tensor[n,2], goal_w: Tensor[n,2], cfg: PhantomCfg) -> Tensor[n, max_count, 5]` (cx, cy, half x, half y, yaw; unused slots have half sizes 0).
  `phantom_height(boxes: Tensor[n,M,5], pose_w: Tensor[n,3], cells: int, resolution: float, height_m: float) -> Tensor[n,cells,cells]` (robot-centred, yaw-aligned like `failbench.sensing.cell_centres`; 0 outside boxes).

- [ ] **Step 1: Write the failing tests:**

```python
def test_a_phantom_box_lands_where_it_is_in_the_robot_frame():
  boxes = torch.tensor([[[1.0, 0.0, 0.2, 0.2, 0.0]]])
  facing_left = torch.tensor([[0.0, 0.0, math.pi / 2]])
  h = phantom_height(boxes, facing_left, 80, 0.05, 0.8)
  xy = cell_centres(80, 0.05, "cpu")[h[0] > 0]
  # World +x is the robot's right (-y) when it faces +y.
  assert torch.allclose(xy.mean(0), torch.tensor([0.0, -1.0]), atol=0.05)


def test_phantoms_keep_clear_of_spawn_and_goal_and_near_the_line():
  torch.manual_seed(0)
  n = 500
  start, goal = torch.zeros(n, 2), torch.tensor([[6.0, 0.0]]).expand(n, 2)
  b = sample_phantoms(start, goal, PhantomCfg())
  real = b[..., 2] > 0
  for end in (start, goal):
    h = phantom_height(b, torch.cat([end, torch.zeros(n, 1)], -1), 32, 0.05, 0.8)
    near = cell_centres(32, 0.05, "cpu").norm(dim=-1) < 0.6
    assert (h[:, near] == 0).all()
  assert (b[..., 1][real].abs() <= 1.5 + 1e-6).all()
  assert 0.1 < (~real.any(-1)).float().mean() < 0.3
```

- [ ] **Step 2: Run** `uv run pytest failbench/tests/test_driver.py -v` — expected FAIL.
- [ ] **Step 3: Implement**; clearance is enforced by rejection (redraw up to 20 times, then drop the box).
- [ ] **Step 4: Run** — expected PASS. **Step 5: Checkpoint** `make check`.

### Task 5: Planner driver in the command term

**Files:**
- Create: `failbench/driver.py`
- Modify: `failbench/commands.py` (`GoalVelocityCommandCfg.driver: PlannerDriveCfg | None = None`; when set, the driver replaces `follow()` before the fault and `SegmentPlan` after it)
- Test: `failbench/tests/test_driver.py`

**Interfaces:**
- Consumes: `Nav2Mppi`/`Nav2MppiCfg`, `Dwb`/`DwbCfg`, `nav2_costmap`, Task 4.
- Produces: `PlannerDriveCfg` (frozen): `planner: Literal["nav2", "dwb"] = "nav2"`, `nav2: Nav2MppiCfg = Nav2MppiCfg(batch_size=256)`, `dwb: DwbCfg = DwbCfg()`, `phantoms: PhantomCfg`, `map_cells: int = 120`, `sample_prob: float = 0.8` (commit a uniformly random batch candidate; else the planner's own plan), `commit_steps: int = 30`.
  `class PlannerDriver`: `__init__(n, cfg, device)`; attributes `planner` (the `Nav2Mppi` or `Dwb`), `boxes [n, max_count, 5]` (this trial's phantoms, world frame), `last` (the last `step()` output dict); `reset(ids, start_w: Tensor[m,2], goal_w: Tensor[m,2])` (new phantoms, planner reset); `cycle(pose_w: Tensor[n,3], speed_b: Tensor[n,3], goal_w: Tensor[n,2], still: Tensor[n] bool) -> Tensor[n,3]` (one planning cycle for all envs; envs `still` are reset and get 0); `commit(ids) -> Tensor[m, commit_steps, 3]` (from the last cycle's `cv`: MPPI sample `k` → `cat(cv[k, 1:], cv[k, -1:])`; MPPI own → `plan[:, :30]`; DWB → the twist held), and `source: Tensor[n] long` (0 MPPI sample, 1 MPPI plan, 2 DWB grid twist, 3 DWB best).
  `GoalVelocityCommand` gains `committed [n, 30, 3]`, `commit_step [n]`; `dense` logging in `generate.py` already records `command.command`.
- Behaviour: cycles run every 5 control steps on each env's own clock and additionally on the fault step (Review Focus 3): on the step where `fault.faulted & ~started` the term runs `cycle` for those envs, `commit`s, and sends `committed[:, 0]`; afterwards it sends `committed[:, j]` with `j = (step - commit_step) // 5`, holding the last after 30. `prefault_still` envs stand and are reset each cycle (Review Focus 4). Speed input: `root_link_lin_vel_b[:, :2]` and `root_link_ang_vel_b[:, 2]` (true velocities; data, not odometry).

- [ ] **Step 1: Write the failing tests** (pure torch; the planner driven by its own motion model as in `test_nav2._drive`):

```python
def _walk(driver, cycles, goal_w, still=None):
  """Drive ``driver`` from the origin facing +x, the robot moved one cycle
  at a time by ``accel_limited`` + ``integrate`` (Task 1) on the command
  sent; returns poses [cycles + 1, n, 3] and commands [cycles, n, 3]."""


def test_the_driver_sidesteps_a_phantom_on_the_line():
  torch.manual_seed(0)
  driver = PlannerDriver(4, PlannerDriveCfg(), "cpu")
  goal = torch.tensor([[6.0, 0.0]]).expand(4, 2)
  driver.reset(torch.arange(4), torch.zeros(4, 2), goal)
  driver.boxes[:] = 0.0
  driver.boxes[:, 0] = torch.tensor([2.5, 0.0, 0.3, 0.4, 0.0])  # On the line.
  poses, cmds = _walk(driver, 60, goal)
  lateral = poses[..., 1][(poses[..., 0] > 2.0) & (poses[..., 0] < 3.0)]
  assert lateral.abs().min() > 0.3  # Went round it, never through.
  assert cmds[..., 1].abs().max() > 0.05 or cmds[..., 2].abs().max() > 0.2


def test_a_commit_is_one_of_the_cycles_candidates_and_holds_three_seconds():
  torch.manual_seed(0)
  driver = PlannerDriver(2, PlannerDriveCfg(sample_prob=1.0), "cpu")
  goal = torch.tensor([[6.0, 0.0]]).expand(2, 2)
  driver.reset(torch.arange(2), torch.zeros(2, 2), goal)
  _walk(driver, 5, goal)
  seq = driver.commit(torch.tensor([0, 1]))
  assert seq.shape == (2, 30, 3) and (driver.source == 0).all()
  cv = driver.last["cv"]  # [n, K, T, 3] of the last cycle
  rows = torch.cat([cv[:, :, 1:], cv[:, :, -1:]], 2)
  assert all((rows[b] == seq[b]).all(-1).all(-1).any() for b in range(2))


def test_still_trials_stand_and_their_planner_starts_from_rest():
  torch.manual_seed(0)
  driver = PlannerDriver(2, PlannerDriveCfg(), "cpu")
  goal = torch.tensor([[6.0, 0.0]]).expand(2, 2)
  driver.reset(torch.arange(2), torch.zeros(2, 2), goal)
  still = torch.tensor([True, False])
  _, cmds = _walk(driver, 10, goal, still=still)
  assert (cmds[:, 0] == 0).all() and (cmds[-1, 1, 0] > 0.3)
  assert (driver.planner.seq[0] == 0).all()
```

  plus, `@pytest.mark.slow`, in `test_failbench.py` beside `test_fault_writes_and_plan_starts_in_real_env`:

```python
def test_the_driver_commits_on_the_fault_step_in_real_env():
  # failbench_env_cfg(..., driver=PlannerDriveCfg()) with fault_time_s=(0.23, 0.23)
  # (between cycles); on the first step faulted is True:
  assert command.started.all()
  torch.testing.assert_close(
    command.command, deadband(command.committed[:, 0], 0.2)
  )  # failbench.mppi.inputs
```

- [ ] **Step 2: Run** `uv run pytest failbench/tests/test_driver.py -v` — expected FAIL.
- [ ] **Step 3: Implement `PlannerDriver`** in `failbench/driver.py`, then the `driver` branch in `GoalVelocityCommand._update_command` / `_resample_command` (reset phantoms with the spawn and goal at resample); thread `driver` through `failbench_env_cfg` (`failbench/env_cfg.py`) as `prefault` is.
- [ ] **Step 4: Run** `uv run pytest failbench/tests/test_driver.py failbench/tests/test_failbench.py -v -m "not slow"` then the slow test — expected PASS; existing tests unchanged.
- [ ] **Step 5: Checkpoint** `make check`.

### Task 6: Generate and build 3 s planner-driven samples

**Files:**
- Modify: `failbench/scripts/generate.py` (`GenerateConfig.driver: PlannerDriveCfg | None = None`; with it: `horizon_s` 3.5, `snapshot_s` (0.0,); per trial also `plan_source`, and config `driver_planner`)
- Modify: `failbench/data/build.py` (pass `plan_source` through to samples; `BuildConfig.horizon_s` 3.0 used for these shards)
- Test: `failbench/tests/test_failbench.py` (build on a synthetic raw shard, as the existing build tests do)

**Interfaces:**
- Consumes: Task 5 (`command.driver.source`).
- Produces: raw shard key `plan_source [B] uint8`; sample key `plan_source [N]`; samples with `horizon_s` 3 s = 30 bins at 10 Hz.

- [ ] **Step 1: Write the failing test** `test_a_three_second_build_keeps_the_plan_source`: a synthetic raw shard with `snapshot_steps = [0]`, 3.5 s horizon and `plan_source`; `build(..., BuildConfig(horizon_s=3.0))` gives `cmd_seq.shape[1] == 30` and the sources in order.
- [ ] **Step 2: Run** — expected FAIL. **Step 3: Implement.** **Step 4: Run** — expected PASS.
- [ ] **Step 5: Small local shard** (rough, 64 envs, MPPI driver, then DWB):
  `MUJOCO_GL=egl PYTHONPATH=$PWD uv run python failbench/scripts/generate.py --checkpoint-file policies/go2_him_rough_original_7200/model.pt --output-dir <scratchpad>/v3_raw --terrain rough --terrain-seed 0 --num-envs 64 --driver.planner nav2` (and `--driver.planner dwb --prefix rough_s0_dwb`). Expected: most trials kept; time per round printed.
- [ ] **Step 6: Throughput check:** time one planning cycle at 1024 and 4096 envs on the local GPU; record which `num_envs` keeps a round under ~10 min (feeds the stage 3 Hopper plan).
- [ ] **Step 7: Command coverage check** (scratchpad script, not repo): histograms of pre- and post-fault (vx, vy, wz) and per-cycle changes from the new shards against `dense_cmd` of the existing Nav2 runs (`<scratchpad>/sev`, `<scratchpad>/steer`). Expected: every planner command inside the data's range; |vy| > 0.1 in a sizeable share (> 15%) of pre-fault steps. Report the numbers.
- [ ] **Step 8: Inspect** two trials per planner with `failbench/scripts/play_trial.py` (phantoms drawn if easy; otherwise the commanded path) and show the user.
- [ ] **Step 9: Checkpoint** `make check`; full `uv run pytest failbench/tests -m "not slow"`.

---

Builder notes: confirm `ROUGH_TERRAINS_CFG` includes flat and slope sub-terrains before stage 3 picks the terrain list; DWB's PathDist/GoalDist use Euclidean distance to the straight-line path because FailBench has no global planner (Nav2 grows a wavefront over the costmap; on a straight path in free space they agree) — say so in `dwb.py`'s docstring.
