"""Tests for the FailBench fault sampler, LiDAR visibility and dataset."""

import json
import math
import os

import numpy as np
import pytest
import torch
from failbench.commands import GoalVelocityCommand
from failbench.data.build import SPLITS
from failbench.data.dataset import FailBenchDataset
from failbench.faults import (
  FAULT_TERM,
  LOCK,
  NOMINAL,
  TORQUE,
  FaultFamily,
  FaultMixture,
  encode_fault,
  faulted_joint,
  sample_fault_params,
)
from failbench.model.losses import LossWeights, hazard_nll, impact_loss
from failbench.model.model import ImpactModel, ImpactModelCfg
from failbench.plans import PlanCfg, SegmentPlan
from failbench.sensing import LidarCfg, sensor_poses, visible_cells


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def test_sampled_families_fault_one_joint_each():
  torch.manual_seed(0)
  params, family = sample_fault_params(4000, 12, FaultMixture(), "cpu")
  touched = ~torch.isclose(params, torch.tensor(NOMINAL)).all(-1)  # [N, J]
  per_env = touched.sum(-1)
  assert (per_env[family == FaultFamily.NOMINAL] == 0).all()
  assert (per_env[family != FaultFamily.NOMINAL] == 1).all()
  joint = faulted_joint(params)
  assert ((joint >= 0) == (family != FaultFamily.NOMINAL)).all()
  rows = torch.arange(len(family))
  hit = params[rows, joint.clamp(min=0)]
  lock, weak = family == FaultFamily.LOCK, family == FaultFamily.WEAK
  assert (hit[lock, LOCK] == 1).all() and (hit[lock, TORQUE] == 1).all()
  assert (hit[weak, LOCK] == 0).all() and (hit[weak, TORQUE] <= 0.5).all()
  limp = (hit[weak, TORQUE] == 0).float().mean()
  assert abs(limp - FaultMixture().limp_prob) < 0.05
  freq = torch.bincount(family, minlength=3).float() / len(family)
  torch.testing.assert_close(
    freq, torch.tensor(FaultMixture().probs), atol=0.03, rtol=0
  )


def test_healthy_joint_encodes_to_reference():
  enc = encode_fault(torch.tensor(NOMINAL).repeat(2, 12, 1))
  assert enc.shape == (2, 24)
  torch.testing.assert_close(enc[0, :2], torch.tensor([1.0, 0.0]))


def _run_plan(hold: bool, steps: int = 350, n: int = 512) -> tuple:
  torch.manual_seed(0)
  plan = SegmentPlan(n, PlanCfg(), "cpu")
  ids = torch.arange(n)
  c0 = torch.tensor([0.8, 0.0, 0.1]).repeat(n, 1)
  goal = torch.tensor([0.0, 5.0]).repeat(n, 1)  # 90 deg to the left
  plan.start(ids, c0, torch.zeros(n, 3), goal, hold=torch.full((n,), hold))
  cmds = []
  for _ in range(steps):
    plan.step(ids, 0.02)
    cmds.append(plan.cmd.clone())
  return plan, torch.stack(cmds, dim=1)


def test_hold_plan_keeps_the_fault_time_command():
  _, cmds = _run_plan(hold=True)
  assert (cmds == torch.tensor([0.8, 0.0, 0.1])).all()


def test_segment_plans_stay_in_limits_and_pull_toward_the_goal():
  plan, cmds = _run_plan(hold=False)
  lim = torch.tensor(PlanCfg().limits)
  assert (cmds[..., 0] >= 0).all() and (cmds.abs() <= lim + 1e-6).all()
  # vx ramps at most max_accel per step.
  assert (cmds[:, 1:, 0] - cmds[:, :-1, 0]).abs().max() <= 0.02 + 1e-6
  assert (plan.seg_count >= 2).all()
  # The goal is to the left: plans mostly end up turned toward it.
  assert plan.pose[:, 2].median() > math.pi / 4


def test_lane_plans_return_to_the_centre_line():
  torch.manual_seed(0)
  n = 512
  plan = SegmentPlan(n, PlanCfg(), "cpu")
  ids = torch.arange(n)
  # Start 0.4 m left of a lane along +x through (10, 0), heading +x.
  pose = torch.tensor([0.0, 0.4, 0.0]).repeat(n, 1)
  plan.start(
    ids,
    torch.tensor([0.8, 0.0, 0.0]).repeat(n, 1),
    pose,
    torch.tensor([10.0, 0.0]).repeat(n, 1),
    hold=torch.zeros(n, dtype=torch.bool),
    lane=torch.ones(n, dtype=torch.bool),
    bearing=torch.zeros(n),
  )
  for _ in range(250):
    plan.step(ids, 0.02)
  assert plan.pose[:, 1].abs().median() < 0.15


def _fake_batch(b: int = 3) -> dict[str, torch.Tensor]:
  t, cells = 50, 64
  upright = torch.ones(b, t, dtype=torch.bool)
  upright[0, 20:] = False  # Sample 0 falls in step 20.
  return {
    "map": torch.stack(
      [torch.full((b, cells, cells), -0.3), torch.ones(b, cells, cells)], dim=1
    ),
    "state": torch.randn(b, 33),
    "him_latent": torch.randn(b, 16),
    "history": torch.randn(b, 20, 36),
    "fault": torch.tensor(NOMINAL).repeat(b, 12),
    "cmd_seq": torch.rand(b, t, 3),
    "cmd_path": torch.cumsum(torch.rand(b, t, 3) * 0.1, dim=1),
    "pose": torch.cumsum(torch.rand(b, t, 3) * 0.1, dim=1),
    "dv": torch.randn(b, t, 3) * 0.1,
    "upright": upright,
    "valid": torch.ones(b, t, dtype=torch.bool),
    "collision": torch.zeros(b, t, dtype=torch.bool),
    "contact_offmap": torch.zeros(b, t, dtype=torch.bool),
    "contact": torch.rand(b, t, cells, cells) > 0.999,
  }


def _unit_stats() -> dict:
  dims = {
    "state": 33,
    "him_latent": 16,
    "history": 36,
    "cmd_seq": 3,
    "cmd_path": 3,
    "map": 1,
  }
  return {k: {"mean": [0.0] * n, "std": [1.0] * n} for k, n in dims.items()}


def test_impact_model_shapes_and_plan_batching():
  torch.manual_seed(0)
  cfg = ImpactModelCfg(d_model=32, context_layers=1, decoder_layers=1)
  model = ImpactModel(cfg, _unit_stats()).eval()
  batch = _fake_batch()
  pred = model(batch)
  assert pred["drift"].shape == (3, 50, 3)
  assert pred["contact_logit"].shape == (3, 5, 64, 64)
  # P(fallen by t) never decreases.
  assert (pred["p_fallen"].diff(dim=1) >= -1e-6).all()
  total, terms = impact_loss(pred, batch, LossWeights(), cfg.steps_per_map)
  assert torch.isfinite(total) and set(terms) >= {"drift", "fall", "contact"}
  # One context scores many plans.
  ctx = model.encode({k: v[:1] for k, v in batch.items()})
  many = model.decode(ctx, torch.rand(8, 50, 3), torch.rand(8, 50, 3), False)
  assert many["drift"].shape == (8, 50, 3) and "contact_logit" not in many


def test_hazard_nll_prefers_the_true_fall_step():
  upright = torch.ones(1, 50, dtype=torch.bool)
  upright[0, 20:] = False
  valid = torch.ones(1, 50, dtype=torch.bool)
  right = torch.full((1, 50), -6.0)
  right[0, 20] = 6.0
  wrong = torch.full((1, 50), -6.0)
  wrong[0, 40] = 6.0
  assert hazard_nll(right, upright, valid) < hazard_nll(wrong, upright, valid)


def _visible(height: torch.Tensor, lidar: LidarCfg) -> torch.Tensor:
  pos, axis = sensor_poses(
    torch.zeros(1, 1, 3), torch.tensor([[[1.0, 0.0, 0.0, 0.0]]]), lidar
  )
  return visible_cells(
    height[None], pos, axis, torch.zeros(1, 1, 3), torch.zeros(1, 1), 0.05, lidar
  )[0]


def test_lidar_sees_flat_ground_ahead_but_not_under_the_trunk():
  lidar = LidarCfg()
  seen = _visible(torch.full((64, 64), -0.3), lidar)
  centre = (64 - 1) / 2
  ahead = seen[int(centre), int(centre) + 20]  # 1 m ahead
  under = seen[int(centre), int(centre) - 2]  # under the trunk
  assert ahead and not under


def test_lidar_cannot_see_the_tread_below_a_step_down():
  # Ground at -0.3 m up to 0.6 m ahead, then a 0.3 m drop.
  height = torch.full((64, 64), -0.3)
  x = (torch.arange(64) - 31.5) * 0.05
  height[:, x > 0.6] = -0.6
  seen = _visible(height, LidarCfg())
  row = 32
  just_below = seen[row, (x > 0.62) & (x < 0.8)]
  far_below = seen[row, x > 1.4]
  assert not just_below.any()
  assert far_below.all()


def test_dataset_rasterises_csr_contacts(tmp_path):
  shard = tmp_path / "s0"
  shard.mkdir()
  n, nb, cells = 2, 50, 64
  per_sample = {
    "state": np.zeros((n, 33), np.float32),
    "state_true_vel": np.zeros((n, 3), np.float32),
    "him_latent": np.zeros((n, 16), np.float32),
    "history": np.zeros((n, 20, 36), np.float32),
    "cmd_seq": np.zeros((n, nb, 3), np.float32),
    "fault": np.zeros((n, 24), np.float32),
    "dv": np.zeros((n, nb, 3), np.float32),
    "pose": np.zeros((n, nb, 3), np.float32),
    "cmd_path": np.zeros((n, nb, 3), np.float32),
    "upright": np.ones((n, nb), bool),
    "collision": np.zeros((n, nb), bool),
    "valid": np.ones((n, nb), bool),
    "contact_offmap": np.zeros((n, nb), bool),
    "fault_family": np.zeros(n, np.uint8),
    "plan_hold": np.zeros(n, bool),
    "offset_s": np.zeros(n, np.float32),
    "split": np.zeros(n, np.uint8),
    "fallen_at_start": np.zeros(n, bool),
    "map_clean": np.full((n, cells, cells), -0.3, np.float16),
    "known_fused": np.ones((n, cells, cells), bool),
    "contact_ptr": np.array([0, 0, 2]),
    "contact_bin": np.array([3, 7], np.uint8),
    "contact_ix": np.array([10, 20], np.uint8),
    "contact_iy": np.array([30, 40], np.uint8),
    "contact_part": np.array([0, 0], np.uint8),
  }
  for k, v in per_sample.items():
    np.save(shard / f"{k}.npy", v)
  meta = {"num_bins": nb, "map_cells": cells, "terrain_id": 0}
  (shard / "meta.json").write_text(json.dumps(meta))
  ds = FailBenchDataset(tmp_path, SPLITS[0], seed=0)
  assert len(ds) == 2
  assert not ds[0]["contact"].any()
  c = ds[1]["contact"]
  assert c.sum() == 2 and c[3, 30, 10] and c[7, 40, 20]
  assert ds[1]["map"].shape == (2, cells, cells)


@pytest.mark.slow
def test_fault_writes_and_plan_starts_in_real_env():
  from failbench.env_cfg import failbench_env_cfg

  from mjlab.envs import ManagerBasedRlEnv

  device = get_test_device()
  cfg = failbench_env_cfg("Go2-HIM-Rough", "rough", 0, fault_time_s=(0.2, 0.2))
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=device)
  env.reset()
  fault = env.event_manager.get_term_cfg(FAULT_TERM).func
  command = env.command_manager.get_term("twist")
  assert isinstance(command, GoalVelocityCommand)
  # Env 0: joint 2 (FL calf; asset order is FL, FR, RL, RR) locked; env 1:
  # every joint limp.
  fault.params[:] = torch.tensor(NOMINAL, device=device)
  fault.params[0, 2, LOCK] = 1.0
  fault.params[1, :, TORQUE] = 0.0
  robot = env.scene["robot"]
  act = torch.zeros(2, 12, device=device)
  q_at_fault = None
  for _ in range(80):
    env.step(act)
    if q_at_fault is None and fault.faulted.all():
      # On the fault step the plan starts at the follower's command.
      q_at_fault = fault.fault_qpos[0, 2].item()
      assert command.started.all()
      torch.testing.assert_close(command.plan.cmd, command.c0)
  assert q_at_fault is not None
  assert abs(robot.data.joint_pos[0, 2].item() - q_at_fault) < 0.02
  assert robot.data.actuator_force[1].abs().max().item() < 1e-4
  # Open loop: the predicted pose integrates the plan, not the robot.
  assert (command.plan.pose[:, :2] != robot.data.root_link_pos_w[:, :2]).any()
  assert math.isfinite(robot.data.joint_pos.sum().item())
  env.close()
