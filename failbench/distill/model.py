"""The student: the risk critic's damage per imagined fault for every
candidate, from the map layers, the robot's context and the candidate's
commands.

Once per robot and cycle a small U-Net encodes the map layers
(``failbench/distill/layers.py``) into a feature grid at half the map's
resolution and a global vector. Each candidate is then priced cheaply: at
each of its steps the feature grid is sampled where its commanded path is,
joined with the step's command and pose, the robot's context (the frozen
impact model's encoder, pooled) and the global vector, and a causal dilated
1D convolution along the horizon gives each step's damage per fault in
``slots`` learned groups. As the teacher (``RiskCritic._damage_touch``),
which takes each severity's worst step and sums over the severities, a
candidate's damage per fault is each slot's worst step summed over the
slots; the groups are learned, so any severity fits.

A step sees the commands up to itself only, as the impact model's decoder:
what happens at step t can't depend on later commands.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from failbench.distill.layers import NUM_LAYERS
from failbench.mppi.inputs import integrate_plan


@dataclass(frozen=True)
class StudentCfg:
  map_channels: int = NUM_LAYERS
  resolution: float = 0.05
  """The map layers' cell size (m)."""
  feat: int = 32
  """Channels of the sampled feature grid."""
  hidden: int = 64
  context_dim: int = 128
  """The pooled impact-model context (its ``d_model``)."""
  num_faults: int = 8
  slots: int = 4
  """Learned groups, each priced by its worst step (the teacher's
  severities)."""
  steps: int = 30
  dilations: tuple[int, ...] = (1, 2, 4, 8)
  kernel: int = 3
  control_dt: float = 0.02
  substeps: int = 5

  @classmethod
  def from_saved(cls, saved: dict) -> StudentCfg:
    kwargs: dict[str, Any] = {
      k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()
    }
    return cls(**kwargs)

  def to_saved(self) -> dict:
    return asdict(self)


def _block(c_in: int, c_out: int, stride: int) -> nn.Sequential:
  return nn.Sequential(
    nn.Conv2d(c_in, c_out, 3, stride, 1),
    nn.GroupNorm(8, c_out),
    nn.GELU(),
    nn.Conv2d(c_out, c_out, 3, 1, 1),
    nn.GroupNorm(8, c_out),
    nn.GELU(),
  )


class MapEncoder(nn.Module):
  """``[n, C, H, W]`` layers -> feature grid ``[n, feat, H/2, W/2]`` and a
  global vector ``[n, hidden]``."""

  def __init__(self, cfg: StudentCfg):
    super().__init__()
    w = (32, 64, 96, 128)
    self.down = nn.ModuleList(
      [
        _block(cfg.map_channels, w[0], 2),
        _block(w[0], w[1], 2),
        _block(w[1], w[2], 2),
        _block(w[2], w[3], 2),
      ]
    )
    self.up = nn.ModuleList(
      [_block(w[3] + w[2], w[2], 1), _block(w[2] + w[1], w[1], 1)]
    )
    self.last = _block(w[1] + w[0], cfg.feat, 1)
    self.glob = nn.Linear(w[3], cfg.hidden)

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    skips = []
    for stage in self.down:
      x = stage(x)
      skips.append(x)
    glob = self.glob(x.mean((-1, -2)))
    for stage, skip in zip([*self.up, self.last], skips[-2::-1], strict=True):
      x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
      x = stage(torch.cat([x, skip], 1))
    return x, glob


class CausalConv(nn.Module):
  def __init__(self, width: int, kernel: int, dilation: int):
    super().__init__()
    self.pad = (kernel - 1) * dilation
    self.conv = nn.Conv1d(width, width, kernel, dilation=dilation)
    self.norm = nn.LayerNorm(width)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """``x [B, T, width]`` -> same, residual."""
    y = self.conv(F.pad(x.transpose(1, 2), (self.pad, 0))).transpose(1, 2)
    return x + F.gelu(self.norm(y))


def sample_grid(
  feat: torch.Tensor, xy: torch.Tensor, extent: tuple[float, float]
) -> torch.Tensor:
  """``feat [n, C, h, w]`` bilinearly at robot-frame points ``xy [n, P, 2]``
  (m) on a grid spanning ``±extent`` (x, y half widths, m); zero off it.
  -> ``[n, P, C]``."""
  g = xy / xy.new_tensor(extent)
  out = F.grid_sample(feat, g[:, :, None], align_corners=False)  # [n, C, P, 1]
  return out[..., 0].transpose(1, 2)


class Student(nn.Module):
  def __init__(self, cfg: StudentCfg):
    super().__init__()
    self.cfg = cfg
    h = cfg.hidden
    self.map = MapEncoder(cfg)
    self.step_in = nn.Linear(cfg.feat + 3 + 4, h)
    self.step_pos = nn.Parameter(torch.zeros(cfg.steps, h))
    self.ctx_in = nn.Linear(cfg.context_dim, h)
    self.glob_in = nn.Linear(h, h)
    self.temporal = nn.Sequential(
      *[CausalConv(h, cfg.kernel, d) for d in cfg.dilations]
    )
    self.head = nn.Linear(h, cfg.num_faults * cfg.slots)

  def encode(self, layers: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, tuple]:
    """Once per robot and cycle: the feature grid, global vector and the
    grid's half extents (m)."""
    feat, glob = self.map(layers)
    h, w = layers.shape[-2:]
    r = self.cfg.resolution
    return feat, glob, (w * r / 2, h * r / 2)

  def price(
    self,
    feat: torch.Tensor,
    glob: torch.Tensor,
    extent: tuple,
    context: torch.Tensor,
    cmd: torch.Tensor,
  ) -> dict[str, torch.Tensor]:
    """Damage of commands ``cmd [n, K, T, 3]`` (as sent, deadbanded) of
    robots with encoded maps and ``context [n, context_dim]``."""
    c = self.cfg
    n, k, t, _ = cmd.shape
    path = integrate_plan(cmd, c.control_dt, c.substeps)  # [n, K, T, 3]
    at = sample_grid(feat, path[..., :2].reshape(n, k * t, 2), extent)
    yaw = path[..., 2:]
    step = torch.cat(
      [at.reshape(n, k, t, -1), cmd, path[..., :2], torch.cos(yaw), torch.sin(yaw)],
      -1,
    )
    robot = self.ctx_in(context) + self.glob_in(glob)  # [n, h]
    x = self.step_in(step) + self.step_pos[:t] + robot[:, None, None]
    x = self.temporal(x.flatten(0, 1))
    out = F.softplus(self.head(x)).unflatten(0, (n, k))
    per_step = out.unflatten(-1, (c.num_faults, c.slots))
    return {"per_step": per_step, "damage": per_step.amax(2).sum(-1)}

  def forward(
    self, layers: torch.Tensor, context: torch.Tensor, cmd: torch.Tensor
  ) -> dict[str, torch.Tensor]:
    feat, glob, extent = self.encode(layers)
    return self.price(feat, glob, extent, context, cmd)
