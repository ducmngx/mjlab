"""The impact model's inputs, built online the way ``failbench/data/build.py``
builds them offline.

- ``HistoryBuffer``: the last ``history_s`` of state plus command at the
  plan rate, oldest first, ending at the current step (build's
  ``history``).
- ``integrate_plan``: a plan's commanded path. A plan step is held for
  ``substeps`` control steps, so this is build's ``_integrate`` over the
  held commands, read at the end of each plan step: exactly ``cmd_path``
  for ``cmd_seq``.
- ``deadband``: the command term's deadband (planar speeds at or under
  ``zero_below`` are zeroed), applied to plans before scoring, since that is
  what the robot will be told.
- ``crop_center``: the model's 64x64 map from a larger map on the same grid.
"""

from __future__ import annotations

import torch

from mjlab.utils.lab_api.math import wrap_to_pi


class HistoryBuffer:
  """``[n, steps, dim]`` ring of samples, oldest first."""

  def __init__(self, n: int, steps: int, dim: int, device):
    self.data = torch.zeros(n, steps, dim, device=device)
    self.filled = torch.zeros(n, dtype=torch.bool, device=device)

  def reset(self, ids: torch.Tensor) -> None:
    self.filled[ids] = False

  def push(self, x: torch.Tensor) -> None:
    """Append ``x [n, dim]``. An env's first sample fills its whole
    buffer, as if the robot had been in that state all along."""
    fresh = ~self.filled
    self.data = torch.cat([self.data[:, 1:], x[:, None]], dim=1)
    self.data[fresh] = x[fresh, None].expand(-1, self.data.shape[1], -1)
    self.filled[:] = True


def integrate_plan(cmd: torch.Tensor, step_dt: float, substeps: int) -> torch.Tensor:
  """(x, y, yaw) at the end of each plan step of body twists ``cmd [..., T,
  3]``, each held ``substeps`` control steps of ``step_dt``, from the origin
  facing +x."""
  fine = cmd.repeat_interleave(substeps, dim=-2)
  yaw = torch.cumsum(fine[..., 2] * step_dt, dim=-1)
  yaw0 = yaw - fine[..., 2] * step_dt
  c, s = torch.cos(yaw0), torch.sin(yaw0)
  dx = (c * fine[..., 0] - s * fine[..., 1]) * step_dt
  dy = (s * fine[..., 0] + c * fine[..., 1]) * step_dt
  path = torch.stack(
    [torch.cumsum(dx, -1), torch.cumsum(dy, -1), wrap_to_pi(yaw)], dim=-1
  )
  return path[..., substeps - 1 :: substeps, :]


def deadband(cmd: torch.Tensor, zero_below: float) -> torch.Tensor:
  small = cmd[..., :2].norm(dim=-1, keepdim=True) <= zero_below
  return torch.cat([torch.where(small, 0.0, cmd[..., :2]), cmd[..., 2:]], dim=-1)


def crop_center(x: torch.Tensor, cells: int) -> torch.Tensor:
  """The centre ``cells x cells`` of ``x [..., H, W]``. The grids share cell
  centres when ``H - cells`` is even."""
  h, w = x.shape[-2:]
  if (h - cells) % 2 or (w - cells) % 2:
    raise ValueError(f"Can't centre a {cells}-cell crop in a {h}x{w} map.")
  y0, x0 = (h - cells) // 2, (w - cells) // 2
  return x[..., y0 : y0 + cells, x0 : x0 + cells]
