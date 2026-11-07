"""Forking a run: a restored snapshot replays what the robot did."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

POLICY = Path(__file__).parents[2] / "policies/go2_him_rough_original_7200/model.pt"


@pytest.mark.slow
@pytest.mark.skipif(not POLICY.exists(), reason="needs the Go2 HIM policy")
def test_a_restored_snapshot_retraces_the_walk_it_was_taken_from():
  from failbench.commands import GoalVelocityCommand
  from failbench.env_cfg import DEFAULT_TASK, failbench_env_cfg
  from failbench.sim_state import restore, snapshot
  from heat_bench.scripts.policy_loading import load_runner

  from mjlab.envs import ManagerBasedRlEnv

  device = "cuda:0" if torch.cuda.is_available() else "cpu"
  cfg = failbench_env_cfg(DEFAULT_TASK, "house_lobby", fault_time_s=(1e9, 1e9))
  cfg.scene.num_envs = 2
  env = ManagerBasedRlEnv(cfg=cfg, device=device)
  wrapped, runner = load_runner(
    env, device, checkpoint_file=str(POLICY), task_id=DEFAULT_TASK
  )
  policy = runner.get_inference_policy(device=device)
  command = env.command_manager.get_term("twist")
  assert isinstance(command, GoalVelocityCommand)
  command.use_external[:] = True
  # A walk that turns: forward, then forward while turning left.
  cmds = [(0.8, 0.0, 0.0)] * 60 + [(0.7, 0.1, 0.5)] * 60
  obs, _ = wrapped.reset()
  snap, then = None, []
  with torch.no_grad():
    for i, c in enumerate(cmds):
      command.external[:] = torch.tensor(c, device=device)
      obs, _, _, _ = wrapped.step(policy(obs))
      if i == 59:
        snap = snapshot(env, dict(obs))
      if i >= 60:
        then.append(env.scene["robot"].data.root_link_pos_w[0].clone())
    assert snap is not None
    # Both envs from env 0's snapshot, whatever they were doing.
    src = {k: torch.as_tensor(v[[0, 0]], device=device) for k, v in snap.items()}
    rows = torch.arange(2, device=device)
    got = restore(env, src, rows)
    for k, v in got.items():
      obs[k] = v
    again = []
    for c in cmds[60:]:
      command.external[:] = torch.tensor(c, device=device)
      obs, _, _, _ = wrapped.step(policy(obs))
      again.append(env.scene["robot"].data.root_link_pos_w.clone())
  ref = torch.stack(then)  # [T, 3]
  rep = torch.stack(again)  # [T, 2, 3]
  # 1.2 s of turning walk: both copies retrace it to within a millimetre
  # (restoring the pose alone, without the policy's history, is ~2 cm off).
  err = (rep - ref[:, None]).norm(dim=-1)
  assert err.max() < 1e-3, err.max()
  env.close()
