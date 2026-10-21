"""Tests for the FailBench fault sampler, LiDAR visibility and dataset."""

import json
import math
import os

import numpy as np
import pytest
import torch
from failbench.commands import KnotVelocityCommand
from failbench.data.build import SPLITS
from failbench.data.dataset import FailBenchDataset
from failbench.faults import (
  FAULT_TERM,
  KD,
  KP,
  NOMINAL,
  RANGE,
  TORQUE,
  FaultFamily,
  FaultMixture,
  encode_fault,
  sample_fault_params,
)
from failbench.sensing import LidarCfg, sensor_poses, visible_cells


def get_test_device() -> str:
  if os.environ.get("FORCE_CPU") == "1":
    return "cpu"
  return "cuda" if torch.cuda.is_available() else "cpu"


def test_sampled_families_touch_the_right_joints():
  torch.manual_seed(0)
  params, family = sample_fault_params(4000, 12, FaultMixture(), "cpu")
  touched = ~torch.isclose(params, torch.tensor(NOMINAL)).all(-1)  # [N, J]
  per_env = touched.sum(-1)
  assert (per_env[family == FaultFamily.NOMINAL] == 0).all()
  assert (per_env[family == FaultFamily.SEVERE] == 1).all()
  assert (per_env[family == FaultFamily.CONTINUOUS] == 1).all()
  mild = per_env[family == FaultFamily.MULTI_MILD]
  assert mild.min() >= 2 and mild.max() <= 4
  severe = params[family == FaultFamily.SEVERE]
  hit = touched[family == FaultFamily.SEVERE]
  locked = severe[hit][:, RANGE] == 0
  weak = severe[hit][:, TORQUE] <= 0.3
  assert (locked | weak).all() and locked.any() and weak.any()
  freq = torch.bincount(family, minlength=4).float() / len(family)
  torch.testing.assert_close(
    freq, torch.tensor(FaultMixture().probs), atol=0.03, rtol=0
  )


def test_healthy_joint_encodes_to_reference():
  enc = encode_fault(torch.tensor(NOMINAL).repeat(2, 12, 1))
  assert enc.shape == (2, 72)
  torch.testing.assert_close(enc[0, :6], torch.tensor([1.0, 0.0, 0.0, 1.0, 1.0, 1.0]))


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
    "cmd_seq": np.zeros((n, nb, 3), np.float32),
    "fault": np.zeros((n, 72), np.float32),
    "dv": np.zeros((n, nb, 3), np.float32),
    "pose": np.zeros((n, nb, 3), np.float32),
    "upright": np.ones((n, nb), bool),
    "valid": np.ones((n, nb), bool),
    "contact_offmap": np.zeros((n, nb), bool),
    "fault_family": np.zeros(n, np.uint8),
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
def test_fault_writes_and_knot_switches_in_real_env():
  from failbench.env_cfg import failbench_env_cfg

  from mjlab.envs import ManagerBasedRlEnv

  device = get_test_device()
  cfg = failbench_env_cfg("Go2-HIM-Rough", "rough", 0, fault_time_s=(0.2, 0.2))
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg, device=device)
  env.reset()
  fault = env.event_manager.get_term_cfg(FAULT_TERM).func
  command = env.command_manager.get_term("twist")
  assert isinstance(command, KnotVelocityCommand)
  # Env 0: joint 2 (FL calf; asset order is FL, FR, RL, RR) locked; env 1:
  # every joint's PD gains zeroed.
  fault.params[:] = torch.tensor(NOMINAL, device=device)
  fault.params[0, 2, RANGE] = 0.0
  fault.params[1, :, KP] = 0.0
  fault.params[1, :, KD] = 0.0
  robot = env.scene["robot"]
  act = torch.zeros(2, 12, device=device)
  q_at_fault = None
  for step in range(1, 80):
    env.step(act)
    if step == 15:
      assert fault.faulted.all()
      q_at_fault = fault.fault_qpos[0, 2].item()
      assert (command.knot_index == 1).all()
  assert q_at_fault is not None
  assert abs(robot.data.joint_pos[0, 2].item() - q_at_fault) < 0.02
  assert robot.data.actuator_force[1].abs().max().item() < 1e-4
  assert (command.knot_index == 2).all()  # 1 s after the fault at 0.2 s
  assert math.isfinite(robot.data.joint_pos.sum().item())
  env.close()
