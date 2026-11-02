"""Tests for the planner-driven data mode: phantom obstacles (only the
planner sees them) and the driver that runs a stock planner around them."""

import math

import torch
from failbench.driver import PlannerDriveCfg, PlannerDriver
from failbench.mppi.nav2 import Nav2Mppi, accel_limited, integrate
from failbench.phantoms import PhantomCfg, phantom_height, sample_phantoms
from failbench.sensing import cell_centres


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


def _walk(driver, cycles, goal_w, still=None):
  """Drive ``driver`` from the origin facing +x, the robot moved one cycle
  at a time by the shared motion model on the command sent; returns poses
  ``[cycles + 1, n, 3]`` and commands ``[cycles, n, 3]``."""
  n = goal_w.shape[0]
  cfg = driver.planner.cfg
  pose, speed = torch.zeros(n, 3), torch.zeros(n, 3)
  still = torch.zeros(n, dtype=torch.bool) if still is None else still
  poses, cmds = [pose.clone()], []
  for _ in range(cycles):
    cmd = driver.cycle(pose, speed, goal_w, still)
    cmds.append(cmd)
    v = accel_limited(cmd[:, None, None].expand(-1, 1, 2, -1), speed, cfg)[:, 0, 1]
    pose = integrate(v[:, None, None], pose, cfg)[:, 0, 0]
    speed = v
    poses.append(pose.clone())
  return torch.stack(poses), torch.stack(cmds)


def _driver(n, **kw):
  torch.manual_seed(0)
  driver = PlannerDriver(n, PlannerDriveCfg(**kw), "cpu")
  goal = torch.tensor([[6.0, 0.0]]).expand(n, 2)
  driver.reset(torch.arange(n), torch.zeros(n, 2), goal)
  return driver, goal


def test_the_driver_sidesteps_a_phantom_on_the_line():
  driver, goal = _driver(4)
  driver.boxes[:] = 0.0
  driver.boxes[:, 0] = torch.tensor([2.5, 0.0, 0.3, 0.4, 0.0])  # On the line.
  poses, cmds = _walk(driver, 60, goal)
  lateral = poses[..., 1][(poses[..., 0] > 2.0) & (poses[..., 0] < 3.0)]
  assert lateral.abs().min() > 0.3  # Went round it, never through.
  assert cmds[..., 1].abs().max() > 0.05 or cmds[..., 2].abs().max() > 0.2


def test_a_commit_is_one_of_the_cycles_candidates_and_holds_three_seconds():
  driver, goal = _driver(2, sample_prob=1.0)
  _walk(driver, 5, goal)
  seq = driver.commit(torch.tensor([0, 1]))
  assert seq.shape == (2, 30, 3) and (driver.source == 0).all()
  # A row of the last cycle's batch, sent as the plan would be (clamped to
  # the limits from the cycle's start speed).
  cv = driver.last["cv"]  # [n, K, T, 3]
  planner = driver.planner
  assert isinstance(planner, Nav2Mppi)
  for b in range(2):
    sent = planner._constrain(cv[b], driver.v0[b].expand(cv.shape[1], 3))
    rows = torch.cat([sent[:, 1:], sent[:, -1:]], 1)
    assert torch.isclose(rows, seq[b]).all(-1).all(-1).any()


def test_a_dwb_commit_is_a_grid_twist_held():
  driver, goal = _driver(2, planner="dwb", sample_prob=1.0)
  _walk(driver, 5, goal)
  seq = driver.commit(torch.tensor([0, 1]))
  assert seq.shape == (2, 30, 3) and (driver.source == 2).all()
  assert torch.equal(seq[:, 0], seq[:, -1])


def test_still_trials_stand_and_their_planner_starts_from_rest():
  driver, goal = _driver(2)
  still = torch.tensor([True, False])
  _, cmds = _walk(driver, 10, goal, still=still)
  assert (cmds[:, 0] == 0).all() and (cmds[-1, 1, 0] > 0.3)
  # Its planner starts every cycle from rest: the first command it would
  # send is one cycle's acceleration from 0.
  assert (driver.planner.last_cmd[0] == 0).all()
  assert (driver.v0[0] == 0).all()


def test_phantoms_stand_where_the_robot_walks_before_its_fault():
  torch.manual_seed(1)
  n = 500
  start, goal = torch.zeros(n, 2), torch.tensor([[10.0, 0.0]]).expand(n, 2)
  cfg = PhantomCfg()
  b = sample_phantoms(start, goal, cfg)
  real = b[..., 2] > 0
  lo, hi = cfg.along_m
  assert (b[..., 0][real] >= lo - 1e-6).all() and (b[..., 0][real] <= hi + 1e-6).all()


def test_a_committed_mppi_sample_keeps_the_planners_limits():
  driver, goal = _driver(8, sample_prob=1.0)
  _walk(driver, 5, goal)
  seq = driver.commit(torch.arange(8))
  cfg = driver.planner.cfg
  assert (seq[..., 0] >= -1e-6).all() and (seq[..., 0] <= 1.2 + 1e-6).all()
  assert (seq[..., 1].abs() <= 0.5 + 1e-6).all()
  assert (seq[..., 2].abs() <= 1.0 + 1e-6).all()
  step = seq.diff(dim=1).abs().amax((0, 1))
  assert (step <= torch.tensor([3.0, 3.0, 3.5]) * cfg.model_dt + 1e-5).all()


def test_a_still_trials_own_plan_sets_off_toward_the_goal():
  # A trial standing at its fault that commits to the planner's own plan
  # sets off as the planner would from rest, not stands for 3 s.
  driver, goal = _driver(2, sample_prob=0.0)
  still = torch.tensor([True, False])
  _walk(driver, 5, goal, still=still)
  seq = driver.commit(torch.tensor([0]))
  assert (driver.source[0] == 1) and seq[0, :, 0].max() > 0.3


def test_an_off_cycle_plan_leaves_the_other_trials_planners_as_they_were():
  driver, goal = _driver(3)
  _walk(driver, 5, goal)
  before = {k: v.clone() for k, v in driver.snapshot().items()}
  keep = torch.tensor([False, True, False])
  saved = driver.snapshot()
  driver.cycle(
    torch.zeros(3, 3), torch.zeros(3, 3), goal, torch.zeros(3, dtype=torch.bool)
  )
  driver.restore(saved, keep)
  for k, v in before.items():
    now = getattr(driver.planner, k)
    assert torch.equal(now[0], v[0]) and torch.equal(now[2], v[2]), k
  planner = driver.planner
  assert isinstance(planner, Nav2Mppi)
  assert not torch.equal(planner.seq[1], before["seq"][1])


def test_slow_trials_are_driven_no_faster_than_their_cruise_speed():
  driver, goal = _driver(2)
  cap = torch.tensor([0.3, float("inf")])
  cmds = [
    driver.cycle(
      torch.zeros(2, 3), torch.zeros(2, 3), goal, torch.zeros(2, dtype=torch.bool), cap
    )
    for _ in range(15)
  ]
  cmds = torch.stack(cmds)
  assert cmds[:, 0, 0].max() <= 0.3 + 1e-6 and cmds[:, 1, 0].max() > 0.3
