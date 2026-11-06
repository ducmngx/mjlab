"""Tests for the Nav2 MPPI port: its costmap, a critic, and closed loop on
small worlds of boxes, the robot moved by the planner's own motion model."""

import math

import torch
from failbench.faults import NOMINAL
from failbench.mppi.costmap import obstacle_cells
from failbench.mppi.fault_critic import FaultRiskCritic
from failbench.mppi.nav2 import (
  INSCRIBED,
  LETHAL,
  CostmapCfg,
  Nav2Mppi,
  Nav2MppiCfg,
  _lookup,
  accel_limited,
  footprint_max_cost,
  footprint_points,
  inflate,
  integrate,
  nav2_costmap,
)
from failbench.mppi.planner import FaultRiskCfg, Mppi, MppiCfg
from failbench.sensing import cell_centres
from failbench.tests.test_mppi import _Thrown

RES = 0.05
CELLS = 160


def _local_heights(boxes, pose):
  """Robot-centred, yaw-aligned height maps ``[n, H, W]`` of world boxes
  ``(x0, x1, y0, y1)`` (0.8 m tall) around robot poses ``[n, 3]``."""
  grid = cell_centres(CELLS, RES, "cpu")  # [H, W, 2], robot frame
  c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
  gx, gy = grid[..., 0][None], grid[..., 1][None]
  wx = pose[:, 0, None, None] + c[:, None, None] * gx - s[:, None, None] * gy
  wy = pose[:, 1, None, None] + s[:, None, None] * gx + c[:, None, None] * gy
  h = torch.zeros(len(pose), CELLS, CELLS)
  for x0, x1, y0, y1 in boxes:
    h[(wx >= x0) & (wx <= x1) & (wy >= y0) & (wy <= y1)] = 0.8
  return h


def _drive(boxes, goal_w, cycles=80, n=4, seed=0, risk_weight=0.0, make=None):
  """Run the planner closed loop from the origin facing +x; returns the
  poses ``[cycles + 1, n, 3]``, commands ``[cycles, n, 3]`` and whether the
  footprint ever overlapped a box. ``risk_weight`` > 0 adds the fault-risk
  critic, every fault throwing the robot right (``_Thrown``), the boxes
  glass. ``make(n)`` builds another planner (e.g. DWB) instead."""
  torch.manual_seed(seed)
  cfg = Nav2MppiCfg(batch_size=400)
  critics, engine = [], None
  if risk_weight > 0:
    risk = FaultRiskCfg(p=0.1, modes=("free",))
    engine = Mppi(MppiCfg(fault_risk=risk), _Thrown(side=-1.0), n, "cpu")
    critics.append(FaultRiskCritic(engine, risk_weight))
  planner = Nav2Mppi(cfg, n, "cpu", extra_critics=critics)
  if make is not None:
    planner = make(n)
  cfg = planner.cfg
  pose = torch.zeros(n, 3)
  speed = torch.zeros(n, 3)
  poses, cmds, hit = [pose.clone()], [], False
  goal_w = torch.tensor(goal_w).expand(n, 2)
  for _ in range(cycles):
    cmap = nav2_costmap(_local_heights(boxes, pose), RES, cfg.costmap)
    rel = goal_w - pose[:, :2]
    c, s = torch.cos(pose[:, 2]), torch.sin(pose[:, 2])
    goal = torch.stack(
      [c * rel[:, 0] + s * rel[:, 1], -s * rel[:, 0] + c * rel[:, 1]], -1
    )
    context = None
    if engine is not None:
      h = _local_heights(boxes, pose)
      obst = obstacle_cells(h, cfg.costmap.wall_jump_m)
      context = {
        "inputs": {"fault": torch.tensor(NOMINAL).repeat(12)[None].expand(n, -1)},
        "risk_fields": engine._risk_fields(
          obst, obst & False, torch.full_like(h, 25.0), RES
        ),
      }
    out = planner.step(cmap, RES, goal, speed, context)
    cmd = out["cmd"]
    cmds.append(cmd)
    # The robot walks the command for one cycle, as the motion model has it.
    v = accel_limited(cmd[:, None, None].expand(-1, 1, 2, -1), speed, cfg)[:, 0, 1]
    step = integrate(v[:, None, None], pose, cfg)[:, 0, 0]
    pose, speed = step, v
    poses.append(pose.clone())
    fp = footprint_points(pose[:, None, None], cfg.costmap)[:, 0, 0]  # [n, F, 2]
    for x0, x1, y0, y1 in boxes:
      inside = (
        (fp[..., 0] > x0) & (fp[..., 0] < x1) & (fp[..., 1] > y0) & (fp[..., 1] < y1)
      )
      hit |= bool(inside.any())
  return torch.stack(poses), torch.stack(cmds), hit


def test_footprint_max_cost_matches_the_outline_lookup():
  cm = CostmapCfg()
  g = torch.Generator().manual_seed(0)
  costmap = torch.randint(0, 255, (3, 40, 40), generator=g).float()
  pts = torch.randn(3, 5, 7, 3, generator=g)
  full = _lookup(costmap, footprint_points(pts, cm), RES).amax(-1)
  assert torch.equal(footprint_max_cost(costmap, pts, cm, RES), full)


def test_inflation_matches_nav2s_layer():
  cfg = CostmapCfg()
  obst = torch.zeros(1, 60, 60, dtype=torch.bool)
  obst[:, :, 30] = True  # A wall along y at x index 30.
  cost = inflate(obst, RES, cfg)[0, 30]
  assert cost[30] == LETHAL
  assert cost[27] == INSCRIBED  # 0.15 m: inside the 0.18 m inscribed radius.
  # 0.30 m: 252 exp(-3 (0.30 - 0.18)), floored.
  assert cost[24] == math.floor(252 * math.exp(-3 * (0.30 - 0.18)))
  assert cost[30 - 15] == 0  # 0.75 m: beyond the 0.70 m inflation radius.


def test_inflation_margins_push_the_layer_out_by_what_obstacles_are():
  obst = torch.zeros(1, 60, 60, dtype=torch.bool)
  obst[:, :, 30] = True
  sev = torch.zeros(1, 60, 60)
  sev[:, :30, 30] = 25.0  # Glass on one half of the wall, crates (5) on the other.
  sev[:, 30:, 30] = 5.0
  # Uniform: 252 out to the inscribed radius plus 0.3 m, either half.
  cost = inflate(obst, RES, CostmapCfg(margin=0.3), sev)[0]
  assert cost[10, 30 - 9] == INSCRIBED - 1 and cost[50, 30 - 9] == INSCRIBED - 1
  # Semantic: the full 0.3 m from the glass, 0.06 m from the crates.
  cost = inflate(obst, RES, CostmapCfg(margin=0.3, severity_ref=25.0), sev)[0]
  assert cost[10, 30 - 9] == INSCRIBED - 1
  assert cost[50, 30 - 9] < 150


def test_offset_candidates_step_the_mean_sideways_and_carry_on_parallel():
  cfg = Nav2MppiCfg(batch_size=8, offset_m=(-0.4, 0.2))
  planner = Nav2Mppi(cfg, 1, "cpu")
  planner.seq[..., 0] = 1.0  # Walking straight on.
  v = planner._predict(planner._offsets(), torch.tensor([[1.0, 0.0, 0.0]]))
  y = planner._integrate(v, torch.zeros(1, 3))[0, :, :, 1]  # [O, T]
  assert torch.allclose(y[:, -1], torch.tensor([-0.4, 0.2]), atol=0.03)
  assert torch.allclose(y[:, -1], y[:, -5], atol=1e-4)  # Parallel by the end.


def test_steady_candidates_take_the_place_of_the_last_noise_samples():
  grid = ((1.0, 0.2, 0.0), (0.6, -0.3, 0.1))
  planner = Nav2Mppi(Nav2MppiCfg(batch_size=8, steady_candidates=grid), 1, "cpu")
  out = planner.step(
    torch.zeros(1, 80, 80), RES, torch.tensor([[3.0, 0.0]]), torch.zeros(1, 3)
  )
  steady = torch.tensor(grid)[:, None].expand(-1, planner.cfg.time_steps, -1)
  assert torch.equal(out["cv"][0, -2:], steady)


class _WantsSidestep:
  """Charges each sample for how far its mean sideways command is from 0.3."""

  __name__ = "wants_sidestep"

  def __call__(self, data):
    return 200.0 * (data["cv"][..., 1].mean(-1) - 0.3).square()


def test_more_iterations_settle_the_plan_nearer_what_the_critics_want():
  def plan_vy(iterations):
    torch.manual_seed(0)
    cfg = Nav2MppiCfg(iteration_count=iterations)
    planner = Nav2Mppi(cfg, 4, "cpu", extra_critics=[_WantsSidestep()])
    out = planner.step(
      torch.zeros(4, 80, 80),
      RES,
      torch.tensor([[3.0, 0.0]]).expand(4, 2),
      torch.zeros(4, 3),
    )
    return out["plan"][:, : cfg.time_steps - 1, 1].mean()

  one, four = plan_vy(1), plan_vy(4)
  assert abs(four - 0.3) < 0.5 * abs(one - 0.3)


def test_log_mppi_noise_keeps_each_axis_spread_with_heavier_tails():
  torch.manual_seed(0)
  cfg = Nav2MppiCfg(batch_size=4000, log_normal_std=0.8)
  noise = Nav2Mppi(cfg, 2, "cpu")._noise()  # [n, K, T, 3]
  std = torch.tensor([cfg.vx_std, cfg.vy_std, cfg.wz_std])
  flat = noise.reshape(-1, 3)
  assert torch.allclose(flat.std(0), std, rtol=0.1)
  kurtosis = (flat / flat.std(0)).pow(4).mean(0)
  assert (kurtosis > 5.0).all()  # A Gaussian's is 3: more far-out samples.


def test_correlated_noise_keeps_each_axis_spread_and_carries_on_along_the_horizon():
  torch.manual_seed(0)
  cfg = Nav2MppiCfg(batch_size=4000, noise_corr=(0.0, 0.97, 0.0))
  noise = Nav2Mppi(cfg, 2, "cpu")._noise()  # [n, K, T, 3]
  std = torch.tensor([cfg.vx_std, cfg.vy_std, cfg.wz_std])
  assert torch.allclose(noise.reshape(-1, 3).std(0), std, rtol=0.1)
  a, b = noise[:, :, :-1], noise[:, :, 1:]
  lag1 = (a * b).mean((0, 1, 2)) / (a.std((0, 1, 2)) * b.std((0, 1, 2)))
  assert lag1[1] > 0.9 and lag1[0].abs() < 0.05 and lag1[2].abs() < 0.05
  # A sample keeps its sideways drift: its 3 s mean spreads 5x the iid one's.
  assert noise[..., 1].mean(-1).std() > 4 * cfg.vy_std / cfg.time_steps**0.5


def test_the_plain_planner_walks_to_a_goal_in_the_open_within_its_limits():
  poses, cmds, hit = _drive([], (4.0, 0.0), cycles=70)
  cfg = Nav2MppiCfg()
  assert not hit
  dist = (poses[-1, :, :2] - torch.tensor([4.0, 0.0])).norm(dim=-1)
  assert (dist < 0.5).all()
  assert (cmds[..., 0] >= -1e-6).all() and (cmds[..., 0] <= cfg.vx_max + 1e-6).all()
  assert (cmds[..., 2].abs() <= cfg.wz_max + 1e-6).all()
  # It walks, at a good pace: 4 m in under 7 s.
  arrive = ((poses[:, :, :2] - torch.tensor([4.0, 0.0])).norm(dim=-1) < 0.5).float()
  assert (arrive.argmax(0) < 70).all()


def test_the_plain_planner_threads_an_opening_without_touching_it():
  # A wall across at x 2.0-2.3 with a 0.9 m opening on the line.
  wall = [(2.0, 2.3, 0.45, 3.0), (2.0, 2.3, -3.0, -0.45)]
  poses, _, hit = _drive(wall, (5.0, 0.0), cycles=80)
  assert not hit
  assert ((poses[-1, :, :2] - torch.tensor([5.0, 0.0])).norm(dim=-1) < 0.5).all()


def test_a_trajectory_into_a_wall_costs_the_collision_cost():
  cfg = Nav2MppiCfg(batch_size=2)
  planner = Nav2Mppi(cfg, 1, "cpu")
  h = _local_heights([(1.0, 1.3, -3.0, 3.0)], torch.zeros(1, 3))
  cmap = nav2_costmap(h, RES, cfg.costmap)
  v = torch.zeros(1, 2, cfg.time_steps, 3)
  v[0, 0, :, 0] = 1.0  # Straight into the wall, or...
  v[0, 1, :, 2] = 0.5  # ...turning on the spot.
  traj = planner._integrate(v, torch.zeros(1, 3))
  path, yaw, real, length = planner._path(torch.zeros(1, 2), torch.tensor([[3.0, 0.0]]))
  data = {"traj": traj, "costmap": cmap, "resolution": RES, "path_length": length}
  cost, fail = planner._cost(data)
  scale = cfg.critics.cost.collision_cost / (254 * traj[:, :, ::2].shape[2])
  assert torch.isclose(cost[0, 0], torch.tensor(scale))
  assert cost[0, 1] < 1.0 and not bool(fail[0])


def test_the_fault_risk_critic_keeps_away_from_where_a_fault_would_throw():
  # Glass 0.6-1.1 m right of the line, 1.5-4.0 m ahead; faults throw right.
  glass = [(1.5, 4.0, -1.1, -0.6)]
  plain, _, _ = _drive(glass, (6.0, 0.0), cycles=40, n=2)
  careful, _, hit = _drive(glass, (6.0, 0.0), cycles=40, n=2, risk_weight=3.0)
  beside = lambda p: p[..., 1][(p[..., 0] > 1.5) & (p[..., 0] < 4.0)].mean()  # noqa: E731
  assert not hit
  assert beside(careful) > beside(plain) + 0.15


def test_a_critic_scaled_for_one_robot_changes_only_its_plan():
  goal = torch.tensor([[3.0, 1.5]]).expand(2, 2)
  cmap = torch.zeros(2, 80, 80)

  def run(scale=None, temperature=None):
    torch.manual_seed(0)
    planner = Nav2Mppi(Nav2MppiCfg(), 2, "cpu")
    if scale is not None:
      planner.critic_scale = scale
    if temperature is not None:
      planner.temperature_scale = temperature
    out = planner.step(cmap, RES, goal, torch.zeros(2, 3))
    for _ in range(2):
      out = planner.step(cmap, RES, goal, torch.zeros(2, 3))
    return out["cmd"]

  base = run()
  both = torch.tensor([1.0, 4.0])
  for kw in (
    {"scale": {k: both for k in Nav2Mppi.CRITICS}},
    {"temperature": torch.tensor([1.0, 0.1])},
  ):
    cmd = run(**kw)
    assert torch.equal(cmd[0], base[0]) and not torch.equal(cmd[1], base[1])
