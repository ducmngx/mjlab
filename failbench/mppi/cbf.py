"""A control barrier function (CBF) safety filter on the velocity command,
the reactive baseline to the fault risk.

Each cycle the filter takes MPPI's command ``u = (vx, vy, wz)`` (body frame)
and returns the closest command, in ``W``'s metric, that never closes on an
obstacle's margin faster than the barrier allows. It plans nothing: what it
doesn't see this instant (a fault that hasn't happened, the next obstacle
round the corner) it can't keep away from.

Model: the HIM policy tracks the command, so each footprint circle (offset
``x_i`` along the trunk, ``FootprintCfg``) moves at ``(vx, vy + wz x_i)``,
linear in ``u``. The map is robot-centred and yaw-aligned, so body and map
frames agree.

Barriers: one per circle and severity level (``failbench/severity.py``,
drops at ``DROP_SEVERITY``), ``h = d_c(p_i) - radius - m_c``, ``d_c`` the
distance to the nearest obstacle of that level. ``m_c`` is
``InflationCfg``'s margin: uniform, or scaled by severity and by the
robot's measured speed (a braking-distance margin: strictly the barrier is
then no CBF of the velocity model, since speed is state, not input).

Constraint, discrete time over the cycle ``dt`` the command is held:
``h(t + dt) >= (1 - gamma) h(t)``, linearised as
``grad d_c(p_i) . pdot_i(u) dt >= -gamma h``: closing on a margin at most
``gamma h / dt``, none at its edge, pushed back out inside it; moving along
it is free. Far barriers (beyond ``max_dist_m``) have no gradient and drop
out.

QP: ``min (u - u0)^T W (u - u0)`` subject to the barriers and the command
limits, solved for every robot at once by Hildreth's dual coordinate
ascent (exact for a small strictly convex QP, no solver dependency). An
infeasible problem (margins on all sides) stops at ``iters`` sweeps; the
result is clipped to the limits and flagged.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from failbench.mppi.costmap import (
  FootprintCfg,
  clearance,
  distance_field,
  drop_cells,
  obstacle_cells,
)
from failbench.severity import DEFAULT_SEVERITY, DROP_SEVERITY


@dataclass(frozen=True)
class CbfCfg:
  margin: float = 0.0
  """As ``InflationCfg.margin``: footprint to obstacle (m) kept clear, at
  ``severity_ref`` and standing still; 0 turns the filter off."""
  severity_ref: float | None = None
  """Each level's margin is ``margin * severity / severity_ref``; None: one
  margin for all."""
  speed_ref: float | None = None
  """The margin grows by ``1 + speed / speed_ref`` with the robot's
  measured planar speed; None: fixed."""
  gamma: float = 0.2
  """Share of the gap to the margin the robot may close per cycle."""
  dt: float = 0.1
  """How long each filtered command is held (the planning cycle)."""
  weights: tuple[float, float, float] = (1.0, 1.0, 0.1)
  """``W``'s diagonal: 1 rad/s of turn costs as much as about 0.3 m/s."""
  limits: tuple[float, float, float] = (1.2, 0.5, 1.0)
  """vx in [0, limits[0]], |vy| and |wz| at most limits[1] and limits[2]
  (``SamplerCfg.max_vx``, ``PlanCfg.limits``)."""
  iters: int = 30
  """Hildreth sweeps over the constraints."""
  footprint: FootprintCfg = field(default_factory=FootprintCfg)


def severity_fields(
  height: torch.Tensor,
  severity: torch.Tensor | None,
  resolution: float,
  fp: FootprintCfg,
) -> list[tuple[float, torch.Tensor]]:
  """(severity, distance field) per severity level of the map ``height [n,
  H, W]`` (``severity`` per cell; ``DEFAULT_SEVERITY`` without), drops at
  ``DROP_SEVERITY``, as the planner's fault-risk fields."""
  obst = obstacle_cells(height, fp.wall_jump_m)
  level = torch.where(obst, DEFAULT_SEVERITY if severity is None else severity, 0.0)
  drops = drop_cells(height, fp, resolution)
  level = torch.where(drops, level.clamp(min=DROP_SEVERITY), level)
  return [
    (v, distance_field(level == v, resolution, fp.max_dist_m))
    for v in level.unique().tolist()
    if v > 0
  ]


def barriers(
  cfg: CbfCfg,
  fields: list[tuple[float, torch.Tensor]],
  speed: torch.Tensor,
  resolution: float,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Constraint rows ``A u >= b`` from every circle and level: ``A [n, M,
  3]``, ``b [n, M]`` (M = circles x levels), for robots at the maps'
  centres facing +x, at planar ``speed [n]``."""
  fp = cfg.footprint
  n = speed.shape[0]
  dev = speed.device
  ox = torch.tensor(fp.circles_x, device=dev)
  margin = torch.full((n,), cfg.margin, device=dev)
  if cfg.speed_ref is not None:
    margin = margin * (1 + speed / cfg.speed_ref)
  pose = torch.zeros(n, 1, 1, 3, device=dev)
  delta = resolution
  shifts = torch.tensor([[delta, 0.0], [-delta, 0.0], [0.0, delta], [0.0, -delta]])
  rows_a, rows_b = [], []
  for severity, dist in fields:
    scale = 1.0 if cfg.severity_ref is None else severity / cfg.severity_ref
    clear = clearance(dist, pose, resolution, fp)[:, 0, 0]  # [n, C]
    near = [
      clearance(dist, pose + torch.tensor([*s, 0.0], device=dev), resolution, fp)[
        :, 0, 0
      ]
      for s in shifts.tolist()
    ]
    grad = torch.stack([near[0] - near[1], near[2] - near[3]], -1) / (2 * delta)
    # Beyond the field's cap nothing is known: no gradient, no constraint.
    grad = torch.where(clear[..., None] < fp.max_dist_m - delta, grad, 0.0)
    h = clear - fp.radius - scale * margin[:, None]
    # d(clear_i)/du = grad . (vx, vy + wz x_i) = (gx, gy, gy x_i).
    a = torch.stack([grad[..., 0], grad[..., 1], grad[..., 1] * ox], -1)
    rows_a.append(a * cfg.dt)
    rows_b.append(-cfg.gamma * h)
  if not rows_a:
    return torch.zeros(n, 0, 3, device=dev), torch.zeros(n, 0, device=dev)
  return torch.cat(rows_a, 1), torch.cat(rows_b, 1)


def solve(
  u0: torch.Tensor,
  a: torch.Tensor,
  b: torch.Tensor,
  weights: torch.Tensor,
  iters: int,
) -> torch.Tensor:
  """``argmin (u - u0)^T diag(weights) (u - u0)`` s.t. ``a u >= b`` for every
  robot (``u0 [n, 3]``, ``a [n, M, 3]``, ``b [n, M]``) by Hildreth's
  method: ``u = u0 + W^-1 a^T lam``, one constraint's ``lam >= 0`` at a
  time."""
  winv = 1.0 / weights
  aw = a * winv  # W^-1 a_j per row
  norm = (aw * a).sum(-1)  # a_j W^-1 a_j
  live = norm > 1e-12
  lam = torch.zeros_like(b)
  u = u0.clone()
  for _ in range(iters):
    for j in range(a.shape[1]):
      slack = b[:, j] - (a[:, j] * u).sum(-1)
      step = torch.where(live[:, j], slack / norm[:, j].clamp(min=1e-12), 0.0)
      new = (lam[:, j] + step).clamp(min=0)
      u = u + (new - lam[:, j])[:, None] * aw[:, j]
      lam[:, j] = new
  return u


class CbfFilter:
  def __init__(self, cfg: CbfCfg):
    self.cfg = cfg

  @torch.no_grad()
  def filter(
    self,
    cmd: torch.Tensor,
    fields: list[tuple[float, torch.Tensor]],
    speed: torch.Tensor,
    resolution: float,
  ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """The safe command ``[n, 3]`` for ``cmd`` given the (severity, distance
    field) ``fields`` (``severity_fields``) and the robots' planar ``speed
    [n]``, and logs: ``active`` barriers (binding after the solve),
    ``change`` (``W``-norm of the edit), ``h_min`` (the smallest barrier,
    m), ``clipped`` (the solve left the limits: an infeasible problem)."""
    c = self.cfg
    dev = cmd.device
    a, b = barriers(c, fields, speed, resolution)
    lim = torch.tensor(c.limits, device=dev)
    eye = torch.eye(3, device=dev)
    # Limits as rows: vx >= 0, -vx >= -max, +-vy, +-wz within theirs.
    box_a = torch.cat([eye, -eye])
    box_b = torch.cat([torch.tensor([0.0], device=dev), -lim[1:], -lim])
    n = cmd.shape[0]
    a_all = torch.cat([a, box_a.expand(n, -1, -1)], 1)
    b_all = torch.cat([b, box_b.expand(n, -1)], 1)
    w = torch.tensor(c.weights, device=dev)
    u = solve(cmd, a_all, b_all, w, c.iters)
    lo = torch.cat([torch.zeros(1, device=dev), -lim[1:]])
    safe = torch.maximum(torch.minimum(u, lim), lo)
    h = -b / c.gamma  # Rows' own h (m); padded rows never bind.
    binding = (a.abs().sum(-1) > 0) & ((a * safe[:, None]).sum(-1) <= b + 1e-4)
    return safe, {
      "active": binding.sum(-1).float(),
      "change": ((safe - cmd).square() * w).sum(-1).sqrt(),
      "h_min": h.amin(-1) if h.shape[1] else torch.full((n,), float("inf"), device=dev),
      "clipped": (safe - u).abs().amax(-1) > 1e-3,
    }
