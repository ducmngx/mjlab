"""Load the reference velocity policy onto a heat_bench env (shared by the
headless fault scripts)."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from mjlab.utils.os import get_checkpoint_path, get_wandb_checkpoint_path

REFERENCE_TASK_ID = "Mjlab-Velocity-Rough-Unitree-Go1"


def load_policy(
  env: ManagerBasedRlEnv,
  device: str,
  checkpoint_file: str | None = None,
  wandb_run_path: str | None = None,
  wandb_checkpoint_name: str | None = None,
  log_root: str = "logs/rsl_rl",
):
  """Wrap ``env`` and load a checkpoint: a local file, a W&B run, or the
  latest under ``log_root``. Returns ``(wrapped_env, policy)``."""
  agent_cfg = load_rl_cfg(REFERENCE_TASK_ID)
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  if checkpoint_file is not None:
    resume_path = Path(checkpoint_file)
  else:
    log_root_path = (Path(log_root) / agent_cfg.experiment_name).resolve()
    if wandb_run_path is not None:
      resume_path, _ = get_wandb_checkpoint_path(
        log_root_path, Path(wandb_run_path), wandb_checkpoint_name
      )
    else:
      resume_path = get_checkpoint_path(log_root_path)
  print(f"[INFO] Loading checkpoint: {resume_path}")
  runner_cls = load_runner_cls(REFERENCE_TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
  runner.load(str(resume_path), map_location=device)
  return wrapped, runner.get_inference_policy(device=device)
