"""Nominal (fault-free) runs of the planner-driven data's driver on library
paths: does the robot walk the varied routes the paths ask for?

Each trial tracks a stretch of a library path (``failbench/scripts/
plan_paths.py``) on its phantom map with a stock planner
(``failbench/driver.py``) for ``duration_s``, no fault ever firing. Writes
one ``.npz``:

- per step (``step_dt``): ``base_pos_w [B, T, 3]``, ``base_quat_w``,
  ``joint_pos``, ``lin_vel_b``, ``ang_vel_b``, ``cmd``;
- per trial: ``path [B, P, 2]`` (world, every ``path_every``-th point),
  ``path_real`` (its own points, at full resolution), ``subgoal [B, 2]``,
  ``boxes [B, M, 5]`` its map (world), ``path_id``, ``map_id``,
  ``layout`` (``failbench.maps.LAYOUTS``), ``style [B, 10]``,
  ``reached_s`` (first time within ``arrive_m`` of the subgoal; NaN never).

  uv run python -m failbench.scripts.nominal --checkpoint-file ... \\
    --library paths.npz --output nominal.npz
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import torch
import tyro
from heat_bench.scripts.policy_loading import load_runner

from failbench.commands import GoalVelocityCommand
from failbench.driver import PlannerDriveCfg
from failbench.env_cfg import DEFAULT_TASK, failbench_env_cfg
from failbench.faults import FAULT_TERM
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.torch import configure_torch_backends

NEVER_S = 1e6
"""A fault time (and stair fire radius) no run reaches."""


@dataclass
class NominalConfig:
  checkpoint_file: str
  library: str
  output: str
  task: str = DEFAULT_TASK
  terrain: str = "rough"
  terrain_seed: int | None = 0
  num_envs: int = 256
  duration_s: float = 15.0
  arrive_m: float = 0.3
  path_every: int = 4
  device: str | None = None
  driver: PlannerDriveCfg = PlannerDriveCfg()
  """The driver; ``library`` is set from ``--library``."""


def main(cfg: NominalConfig) -> Path:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  if cfg.terrain_seed is not None:
    torch.manual_seed(cfg.terrain_seed)
  drive = replace(cfg.driver, library=cfg.library)
  env_cfg = failbench_env_cfg(
    cfg.task,
    cfg.terrain,
    cfg.terrain_seed,
    fault_time_s=(NEVER_S, NEVER_S),
    driver=drive,
  )
  params = env_cfg.events[FAULT_TERM].params
  if "fire_radius" in params:  # Stair trials fault by position: never.
    params["fire_radius"] = (NEVER_S, NEVER_S)
    params["fault_time_s"] = (NEVER_S, NEVER_S)
  env_cfg.scene.num_envs = cfg.num_envs
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped, runner = load_runner(
    env, device, checkpoint_file=cfg.checkpoint_file, task_id=cfg.task
  )
  policy = runner.get_inference_policy(device=device)
  robot = env.scene["robot"]
  command = env.command_manager.get_term("twist")
  assert isinstance(command, GoalVelocityCommand)
  driver = command.driver
  assert driver is not None
  n, dt = env.num_envs, env.step_dt
  steps = round(cfg.duration_s / dt)
  keys = ("base_pos_w", "base_quat_w", "joint_pos", "lin_vel_b", "ang_vel_b", "cmd")
  log: dict[str, list[torch.Tensor]] = {k: [] for k in keys}
  rows = torch.arange(n, device=device)
  reached = torch.full((n,), float("nan"), device=device)

  obs, _ = wrapped.reset()
  for t in range(steps):
    with torch.no_grad():
      obs, _, _, _ = wrapped.step(policy(obs))
    data = robot.data
    for k, v in (
      ("base_pos_w", data.root_link_pos_w),
      ("base_quat_w", data.root_link_quat_w),
      ("joint_pos", data.joint_pos),
      ("lin_vel_b", data.root_link_lin_vel_b),
      ("ang_vel_b", data.root_link_ang_vel_b),
      ("cmd", command.command),
    ):
      log[k].append(v.detach().clone())
    subgoal = driver.path[rows, driver.path_real - 1]
    near = (data.root_link_pos_w[:, :2] - subgoal).norm(dim=-1) < cfg.arrive_m
    reached = torch.where(near & reached.isnan(), (t + 1) * dt, reached)
  if (env.episode_length_buf < steps).any():
    print("[WARN] Some trials reset mid-run (fell?).")

  layout = np.load(cfg.library)["layout"]
  map_id = driver.map_id.cpu().numpy()
  out = {k: torch.stack(v, 1).cpu().numpy() for k, v in log.items()}
  out |= {
    "path": driver.path[:, :: cfg.path_every].cpu().numpy(),
    "path_real": driver.path_real.cpu().numpy(),
    "path_every": np.array(cfg.path_every),
    "subgoal": driver.path[rows, driver.path_real - 1].cpu().numpy(),
    "boxes": driver.boxes.cpu().numpy(),
    "path_id": driver.path_id.cpu().numpy(),
    "map_id": map_id,
    "layout": layout[map_id],
    "style": driver.style.cpu().numpy(),
    "reached_s": reached.cpu().numpy(),
    "episode_steps": env.episode_length_buf.cpu().numpy(),
    "step_dt": np.array(dt),
    "planner": np.array(drive.planner),
    "terrain": np.array(cfg.terrain),
    "terrain_seed": np.array(-1 if cfg.terrain_seed is None else cfg.terrain_seed),
    "joint_names": np.array(robot.joint_names),
  }
  path = Path(cfg.output)
  path.parent.mkdir(parents=True, exist_ok=True)
  np.savez_compressed(path, allow_pickle=False, **out)
  done = np.isfinite(out["reached_s"])
  print(
    f"[INFO] {done.sum()}/{n} reached their subgoal (median"
    f" {np.nanmedian(out['reached_s']):.1f} s) -> {path}"
  )
  env.close()
  return path


if __name__ == "__main__":
  cfg = tyro.cli(NominalConfig)
  if not Path(cfg.checkpoint_file).is_file():
    print(f"No checkpoint at {cfg.checkpoint_file}.")
    sys.exit(1)
  main(cfg)
