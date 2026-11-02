"""Global paths for the planner-driven data: resampling a path, and the
part of it ahead of a robot.

A stock local planner (Nav2 MPPI or DWB) tracks a global plan; it never
picks the route itself. The planner-driven data (``failbench/driver.py``)
can give each trial a stretch of a path planned on a phantom map
(``failbench/scripts/plan_paths.py``). A path is points ``[n, P, 2]``
(world) every ``resolution`` metres along it, ``real [n]`` of them its own,
the rest repeating its end. ``local_path`` gives the part ahead of a robot
in its frame, in the form ``failbench.mppi.nav2.straight_path`` gives the
straight one.
"""

from __future__ import annotations

from typing import Any

import torch

from failbench.mppi.nav2 import _wrap


def resample(poly: torch.Tensor, resolution: float, points: int):
  """Points every ``resolution`` metres along polylines ``[n, V, 2]``:
  ``[n, points, 2]`` and how many are their own ``[n]`` (the rest repeat
  the end)."""
  seg = poly.diff(dim=1).norm(dim=-1)
  cum = torch.cat([torch.zeros_like(seg[:, :1]), seg.cumsum(-1)], 1)  # [n, V]
  total = cum[:, -1]
  s = torch.arange(points, device=poly.device) * resolution
  s = torch.minimum(s[None], total[:, None]).contiguous()
  idx = torch.searchsorted(cum.contiguous(), s, right=True).clamp(1, poly.shape[1] - 1)
  c0, c1 = cum.gather(1, idx - 1), cum.gather(1, idx)
  w = ((s - c0) / (c1 - c0).clamp(min=1e-9)).clamp(0, 1)
  p0 = poly.gather(1, (idx - 1)[..., None].expand(-1, -1, 2))
  p1 = poly.gather(1, idx[..., None].expand(-1, -1, 2))
  pts = p0 + w[..., None] * (p1 - p0)
  real = ((total / resolution).floor().long() + 1).clamp(max=points)
  return pts, real


def local_path(
  path_w: torch.Tensor,
  real: torch.Tensor,
  progress: torch.Tensor,
  pose_w: torch.Tensor,
  c: Any,
  window: int = 40,
):
  """The part of each global path ``[n, P, 2]`` ahead of a robot at
  ``pose_w [n, 3]``, in its frame, as ``straight_path`` gives the straight
  one (points from the nearest on, pruned to ``c.prune_distance``; the
  goal's heading, the path's last; how many are its own; its length), and
  the new ``progress [n]``: the index of the nearest point, looked for only
  in the ``window`` points from the last, so a path that comes back near
  itself (a U-turn) is followed in order. The path's spacing must be
  ``c.path_resolution``."""
  dev = path_w.device
  last = (real - 1)[:, None]

  def take(idx: torch.Tensor) -> torch.Tensor:
    return path_w.gather(1, idx[..., None].expand(-1, -1, 2))

  near = torch.minimum(progress[:, None] + torch.arange(window, device=dev), last)
  dist = (take(near) - pose_w[:, None, :2]).norm(dim=-1)
  progress = near.gather(1, dist.argmin(-1, keepdim=True))[:, 0]
  count = round(c.prune_distance / c.path_resolution) + 1
  ahead = take(torch.minimum(progress[:, None] + torch.arange(count, device=dev), last))
  cy, sy = torch.cos(pose_w[:, 2:3]), torch.sin(pose_w[:, 2:3])
  rel = ahead - pose_w[:, None, :2]
  pts = torch.stack(
    [cy * rel[..., 0] + sy * rel[..., 1], -sy * rel[..., 0] + cy * rel[..., 1]], -1
  )
  own = (real - progress).clamp(1, count)
  tail = take(torch.cat([(last - 1).clamp(min=0), last], 1))
  step = tail[:, 1] - tail[:, 0]
  heading = torch.atan2(step[:, 1], step[:, 0])
  yaw = _wrap(heading - pose_w[:, 2])
  length = (own - 1).float() * c.path_resolution
  return (pts, yaw, own, length), progress
