"""Piecewise-constant twist commands keyed to the fault time.

Each env holds a constant pre-fault command, then a sequence of post-fault
knots of ``knot_s`` each, starting on the step the fault fires (knot ``k``
covers ``[k, k + 1) * knot_s`` after it); the last knot holds after that.
Post-fault sequences come in three styles per env:

- ``stop``: every knot is (0, 0, 0), an emergency stop.
- ``hold``: every knot repeats the pre-fault command.
- ``random``: each knot is drawn independently from the post-fault ranges.

Planar commands with norm at or below ``zero_below`` are zeroed, as in
HIMLoco's sampler (``go2_him/him/commands.py``), so commands stay in the
distribution the policy was trained on. The term reads the fault term's
``faulted`` / ``fault_step``; step events run before the command update in
``ManagerBasedRlEnv.step``, so the first knot starts on the fault step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from failbench.faults import FAULT_TERM
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

STYLES = ("stop", "hold", "random")


class KnotVelocityCommand(UniformVelocityCommand):
  def __init__(self, cfg: KnotVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.knot_cfg = cfg
    n = self.num_envs
    self.knots = torch.zeros(n, cfg.num_knots + 1, 3, device=self.device)
    """Pre-fault command (index 0) and post-fault knots (1..num_knots)."""
    self.style = torch.zeros(n, dtype=torch.long, device=self.device)
    """Index into ``STYLES`` per env."""
    self.knot_index = torch.zeros(n, dtype=torch.long, device=self.device)

  def _deadband(self, cmd: torch.Tensor) -> torch.Tensor:
    small = cmd[..., :2].norm(dim=-1, keepdim=True) <= self.knot_cfg.zero_below
    return torch.cat([torch.where(small, 0.0, cmd[..., :2]), cmd[..., 2:]], dim=-1)

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    m, k, dev = len(env_ids), self.knot_cfg.num_knots, self.device

    def draw(ranges: UniformVelocityCommandCfg.Ranges, *shape: int) -> torch.Tensor:
      lo = torch.tensor(
        [ranges.lin_vel_x[0], ranges.lin_vel_y[0], ranges.ang_vel_z[0]], device=dev
      )
      hi = torch.tensor(
        [ranges.lin_vel_x[1], ranges.lin_vel_y[1], ranges.ang_vel_z[1]], device=dev
      )
      return lo + (hi - lo) * torch.rand(*shape, 3, device=dev)

    pre = self._deadband(draw(self.cfg.ranges, m))
    post = self._deadband(draw(self.knot_cfg.post_fault_ranges, m, k))
    probs = torch.tensor(self.knot_cfg.style_probs, device=dev)
    style = torch.multinomial(probs / probs.sum(), m, replacement=True)
    post = torch.where((style == 0)[:, None, None], 0.0, post)
    post = torch.where((style == 1)[:, None, None], pre[:, None], post)
    self.knots[env_ids, 0] = pre
    self.knots[env_ids, 1:] = post
    self.style[env_ids] = style
    self.knot_index[env_ids] = 0
    self.vel_command_b[env_ids] = pre
    self.vel_command_w[env_ids] = pre

  def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
    fault = self._env.event_manager.get_term_cfg(FAULT_TERM).func
    since = (self._env.episode_length_buf - fault.fault_step).float()
    k = (since * self._env.step_dt / self.knot_cfg.knot_s).floor().long() + 1
    k = torch.where(fault.faulted, k.clamp(1, self.knot_cfg.num_knots), 0)
    self.knot_index[:] = k
    rows = torch.arange(self.num_envs, device=self.device)
    self.vel_command_b[:] = self.knots[rows, k]


@dataclass(kw_only=True)
class KnotVelocityCommandCfg(UniformVelocityCommandCfg):
  """``ranges`` is the pre-fault command range."""

  post_fault_ranges: UniformVelocityCommandCfg.Ranges
  num_knots: int = 7
  knot_s: float = 1.0
  style_probs: tuple[float, float, float] = (0.3, 0.2, 0.5)
  """Probability of each of ``STYLES``."""
  zero_below: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> KnotVelocityCommand:
    return KnotVelocityCommand(self, env)
