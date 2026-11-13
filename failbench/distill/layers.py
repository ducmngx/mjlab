"""The student's map: the planner's robot-centred, yaw-aligned grid as four
layers, the terrain and what there is to hit and how bad hitting it is.

The layers are what a prebuilt site map gives as well: a semantic grid
(severity per cell, ledges) cropped around the localised robot. Nothing
else about the objects goes in: how a hit's damage grows with its energy
follows from the severity (``energy_scale_for``), and the energy from the
candidate's own commands. Channels:

0. terrain height relative to the floor, objects flattened to the ground
   around them (as ``blank_obstacles`` does for the impact model), 0 where
   unknown;
1. known (1) or not;
2. collision: 1 where there is something to hit, every cell the critic
   checks throws against (``severity_levels``: obstacle edges, ledges) and
   the whole of each labelled object;
3. severity of what is there, over ``SEVERITY_NORM`` (0 where nothing is):
   any value, so a new entity is just a new severity.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from failbench.mppi.cbf import severity_levels
from failbench.mppi.costmap import FootprintCfg

NUM_LAYERS = 4
SEVERITY_NORM = 50.0
"""A person's severity: the map's severity layer is 1 at a person."""
FILL_M = 0.6
"""How far (m) the ground is carried in under an object."""


def fill_from_neighbours(x: torch.Tensor, passes: int) -> torch.Tensor:
  """``x [n, H, W]`` with NaN cells filled, ``passes`` times, by the mean of
  their known 3x3 neighbours; cells further in stay NaN."""
  for _ in range(passes):
    hole = x.isnan()
    if not hole.any():
      break
    val = F.avg_pool2d(x.nan_to_num(0.0)[:, None], 3, 1, 1, count_include_pad=False)
    cnt = F.avg_pool2d((~hole).float()[:, None], 3, 1, 1, count_include_pad=False)
    mean = (val / cnt.clamp(min=1e-9))[:, 0]
    x = torch.where(hole & (cnt[:, 0] > 0), mean, x)
  return x


def map_layers(
  height: torch.Tensor,
  severity: torch.Tensor | None,
  resolution: float,
  fp: FootprintCfg,
  level: torch.Tensor | None = None,
) -> torch.Tensor:
  """``[n, NUM_LAYERS, H, W]`` layers of the planner's ``height [n, H, W]``
  (NaN unknown) and per-cell ``severity`` (None: every obstacle a wall);
  ``level``: their ``severity_levels`` if already built."""
  if level is None:
    level = severity_levels(height, severity, resolution, fp)
  if severity is not None:
    # A labelled object counts whole, at its severity; the critic's
    # levels win where they set one (ledges).
    level = torch.where(level > 0, level, severity.clamp(min=0))
  hit = level > 0
  known = ~height.isnan()
  ground = torch.where(known & ~hit, height, float("nan"))
  floor = ground.flatten(1).nanmedian(-1).values.nan_to_num(0.0)[:, None, None]
  # Under an object, the ground around it (a person on a platform stands
  # on the platform); the floor where nothing is near.
  filled = fill_from_neighbours(ground, round(FILL_M / resolution))
  terrain = torch.where(known & ~filled.isnan(), filled - floor, 0.0)
  return torch.stack([terrain, known.float(), hit.float(), level / SEVERITY_NORM], 1)
