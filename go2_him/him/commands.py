"""HIMLoco's velocity command sampling and command curriculum.

legged_gym ``_resample_commands`` / ``update_command_curriculum`` in
HIMLoco's ``legged_robot.py``:

- The first ``high_vel_fraction`` (20%) of envs, by index, draw forward
  speed from the curriculum range (``ranges.lin_vel_x``) and get no lateral
  command once ``|vx| >= 1``. The rest draw ``vx`` from ``low_vel_x``
  (+-1 m/s).
- Every env draws ``vy`` from ``ranges.lin_vel_y`` and a heading target;
  yaw rate follows the heading (``heading_command``).
- Planar commands with norm <= ``zero_below`` (0.2 m/s) are zeroed.
- The curriculum widens ``ranges.lin_vel_x`` by ``step`` (0.2 m/s) each
  side, up to ``max_curriculum``, whenever both groups track linear
  velocity at > 80% of the maximum reward over their episodes; it is
  checked once per episode length.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class HIMVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg: HIMVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.him_cfg = cfg
    self.num_high_vel = int(cfg.high_vel_fraction * self.num_envs)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    super()._resample_command(env_ids)
    cmd = self.vel_command_b
    r = torch.empty(len(env_ids), device=self.device)
    high = env_ids < self.num_high_vel
    cmd[env_ids, 0] = torch.where(
      high,
      r.uniform_(*self.cfg.ranges.lin_vel_x),
      torch.empty_like(r).uniform_(*self.him_cfg.low_vel_x),
    )
    fast = high & (cmd[env_ids, 0].abs() >= 1.0)
    cmd[env_ids, 1] = torch.where(fast, 0.0, cmd[env_ids, 1])
    small = torch.norm(cmd[env_ids, :2], dim=1) <= self.him_cfg.zero_below
    cmd[env_ids, :2] *= (~small).float().unsqueeze(1)
    self.vel_command_w[env_ids] = cmd[env_ids]


@dataclass(kw_only=True)
class HIMVelocityCommandCfg(UniformVelocityCommandCfg):
  high_vel_fraction: float = 0.2
  low_vel_x: tuple[float, float] = (-1.0, 1.0)
  zero_below: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> HIMVelocityCommand:
    return HIMVelocityCommand(self, env)


def him_command_curriculum(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor,
  command_name: str,
  reward_name: str = "tracking_lin_vel",
  max_curriculum: float = 2.0,
  step: float = 0.2,
  threshold: float = 0.8,
) -> dict[str, torch.Tensor]:
  """Widen the high-velocity group's forward-speed range when tracking is
  good (see module docstring)."""
  term = cast(HIMVelocityCommand, env.command_manager.get_term(command_name))
  ranges = term.cfg.ranges
  if (
    isinstance(env_ids, torch.Tensor)
    and len(env_ids) > 0
    and env.common_step_counter % env.max_episode_length == 0
  ):
    sums = env.reward_manager._episode_sums[reward_name][env_ids]
    weight = env.reward_manager.get_term_cfg(reward_name).weight
    # Episode sums are weight * dt-scaled; normalize to the mean raw reward.
    mean_raw = sums / (env.max_episode_length * env.step_dt * weight)
    high = env_ids < term.num_high_vel
    good = [mean_raw[mask].mean() > threshold for mask in (high, ~high) if mask.any()]
    if good and all(bool(g) for g in good):
      lo, hi = ranges.lin_vel_x
      ranges.lin_vel_x = (
        max(lo - step, -max_curriculum),
        min(hi + step, max_curriculum),
      )
  return {
    "lin_vel_x_min": torch.tensor(ranges.lin_vel_x[0]),
    "lin_vel_x_max": torch.tensor(ranges.lin_vel_x[1]),
  }
