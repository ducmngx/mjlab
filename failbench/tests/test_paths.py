"""Tests for resampling a global path and the part of it ahead of a robot
that a local planner tracks."""

import math

import torch
from failbench.mppi.dwb import Dwb, DwbCfg
from failbench.mppi.nav2 import Nav2Mppi, Nav2MppiCfg, straight_path
from failbench.paths import local_path, resample


def test_a_resampled_path_steps_evenly_from_its_start_to_its_end():
  corner = torch.tensor([[[0.0, 0.0], [3.0, 0.0], [3.0, 2.0]]])
  pts, real = resample(corner, 0.05, 200)
  assert int(real[0]) == 101  # 5 m in 0.05 m steps.
  assert torch.allclose(pts[0, 0], torch.zeros(2))
  assert torch.allclose(pts[0, -1], torch.tensor([3.0, 2.0]))
  step = pts[0, : int(real[0])].diff(dim=0).norm(dim=-1)
  # Along the path; across the corner the chord is a little shorter.
  assert (step <= 0.05 + 1e-4).all() and (step > 0.035).all()


def test_the_path_ahead_of_a_robot_on_a_straight_path_is_the_stock_one():
  c = Nav2MppiCfg()
  n = 3
  goal = torch.tensor([[6.0, 2.0]]).expand(n, 2)
  line = torch.stack([torch.zeros(n, 2), goal], 1)
  path, real = resample(line, 0.05, 600)
  pose = torch.tensor([[1.5, 0.5, 0.3]]).expand(n, 3)  # On the line.
  local, progress = local_path(path, real, torch.zeros(n, dtype=torch.long), pose, c)
  pts, yaw, count, length = local
  # Same frame and pruning as straight_path from the robot to the goal.
  to_goal = goal - pose[:, :2]
  cy, sy = math.cos(0.3), math.sin(0.3)
  goal_b = torch.stack(
    [cy * to_goal[:, 0] + sy * to_goal[:, 1], -sy * to_goal[:, 0] + cy * to_goal[:, 1]],
    -1,
  )
  ref = straight_path(torch.zeros(n, 2), goal_b, c)
  assert pts.shape == ref[0].shape
  assert torch.allclose(pts, ref[0], atol=0.03)
  assert torch.allclose(yaw, ref[1], atol=1e-3)
  assert (count - ref[2]).abs().max() <= 1
  assert torch.allclose(length, ref[3], atol=0.06)
  assert (progress > 0).all()


def test_progress_along_a_u_turn_never_jumps_to_the_far_leg():
  # Out 2 m behind the spawn, then back past it to a goal 3 m ahead: at the
  # spawn the return leg is as near as the start.
  u_turn = torch.tensor([[[0.0, 0.0], [-2.0, 0.3], [3.0, 0.0]]]).expand(4, 3, 2)
  path, real = resample(u_turn.contiguous(), 0.05, 600)
  progress = torch.zeros(4, dtype=torch.long)
  for _ in range(5):
    _, progress = local_path(path, real, progress, torch.zeros(4, 3), Nav2MppiCfg())
  assert (progress < 10).all()


def test_a_planner_given_the_straight_path_plans_as_without_one():
  goal = torch.tensor([[3.0, 1.5]]).expand(2, 2)
  cmap = torch.zeros(2, 80, 80)
  for make in (
    lambda: Nav2Mppi(Nav2MppiCfg(), 2, "cpu"),
    lambda: Dwb(DwbCfg(), 2, "cpu"),
  ):
    outs = []
    for given in (False, True):
      torch.manual_seed(0)
      planner = make()
      path = straight_path(torch.zeros(2, 2), goal, planner.cfg) if given else None
      outs.append(planner.step(cmap, 0.05, goal, torch.zeros(2, 3), path=path)["cmd"])
    assert torch.equal(outs[0], outs[1])
