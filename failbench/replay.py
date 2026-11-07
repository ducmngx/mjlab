"""Pricing what a replayed fault touches on a household layout.

``failbench/scripts/replay_faults.py`` forks saved runs, fires a fault and
replays the recorded commands; each step, the robot's contacts (feet too)
are matched to the layout's object boxes (exact, since the layout is known,
and cheap for thousands of robots, unlike the runner's ray lookup). A
contact within ``REACH_M`` of a box and above its base touches that object;
floors cost nothing. The
damage is ``failbench.severity.damage`` at the trunk's planar kinetic energy,
as the runner and the risk critic price a hit.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from failbench.house import Layout
from failbench.house_terrain import EMBED_M
from failbench.severity import ENERGY_SCALE, MASS_KG, SEVERITY, damage

REACH_M = 0.02
"""A contact this close to an object's box is on it (contacts sit on the
surface; the margin absorbs the solver's penetration)."""
BASE_M = 0.02
"""Contacts lower than this above an object's base are the floor's."""


@dataclass(frozen=True)
class ObjectBoxes:
  lo: torch.Tensor
  """``[I, 3]`` each object's lower corner (layout frame, as built: sunk
  ``EMBED_M`` into the ground)."""
  hi: torch.Tensor
  severity: torch.Tensor
  """``[I]``."""
  scale: torch.Tensor
  """``[I]`` energy scale (J)."""

  @classmethod
  def from_layout(cls, layout: Layout, device: str | torch.device) -> ObjectBoxes:
    items = [it for it in layout.items if it.cls != "floor"]
    lo = [(it.x0, it.y0, it.z0 - EMBED_M) for it in items]
    hi = [(it.x1, it.y1, it.z1) for it in items]
    return cls(
      lo=torch.tensor(lo, dtype=torch.float32, device=device),
      hi=torch.tensor(hi, dtype=torch.float32, device=device),
      severity=torch.tensor([SEVERITY[it.cls] for it in items], device=device),
      scale=torch.tensor([ENERGY_SCALE[it.cls] for it in items], device=device),
    )


def object_distance(boxes: ObjectBoxes, points: torch.Tensor) -> torch.Tensor:
  """Each object's distance ``[B, I]`` (m) from the nearest of ``points [B,
  P, 3]`` (layout frame) to its box's surface; 0 inside."""
  p = points[:, :, None]  # [B, P, 1, 3]
  gap = torch.maximum(boxes.lo - p, p - boxes.hi).clamp(min=0.0)
  return gap.norm(dim=-1).amin(1)


def object_contacts(
  boxes: ObjectBoxes,
  points: torch.Tensor,
  touch: torch.Tensor,
  reach_m: float = REACH_M,
  base_m: float = BASE_M,
) -> torch.Tensor:
  """``[B, N, I]``: whether contact ``n`` (at ``points [B, N, 3]``, layout
  frame, where ``touch [B, N]``) is on object ``i``: within ``reach_m`` of
  its box and at least ``base_m`` above the ground it stands on, so a foot
  planted on the floor beside it isn't a touch."""
  p = points[:, :, None]
  lo = boxes.lo + torch.tensor([-reach_m, -reach_m, EMBED_M + base_m], device=p.device)
  hi = boxes.hi + reach_m
  inside = ((p >= lo) & (p <= hi)).all(-1)
  return inside & touch[..., None]


def price_touches(
  boxes: ObjectBoxes,
  points: torch.Tensor,
  touch: torch.Tensor,
  speed2: torch.Tensor,
  reach_m: float = REACH_M,
) -> tuple[torch.Tensor, torch.Tensor]:
  """The worst severity touched ``[B]`` and the worst damage done ``[B]`` by
  contacts at ``points [B, N, 3]`` (layout frame) where ``touch [B, N]``, at
  the trunk's squared planar speed ``speed2 [B]``."""
  hit = object_contacts(boxes, points, touch, reach_m)  # [B, N, I]
  energy = 0.5 * MASS_KG * speed2
  each = damage(boxes.severity, energy[:, None], boxes.scale)  # [B, I]
  sev = torch.where(hit, boxes.severity, 0.0).flatten(1).amax(-1)
  dmg = torch.where(hit, each[:, None], 0.0).flatten(1).amax(-1)
  return sev, dmg


class FaultWindow:
  """Safety metrics over the ``window_s`` after each env's fault (the risk
  critic's horizon): whether the robot touched an object (any, a person,
  glass), the peak contact force on one (N) and the impulse (N s: the
  largest object force each step, summed over the window), the damage done and worst
  severity touched, when it first touched and how fast it was going, its
  distance to the nearest object and person (least over the window, and at
  its end), and whether it fell. Contacts (feet too) are matched by
  ``object_contacts``; distances are from the robot's link origins to the
  objects' surfaces (``object_distance``)."""

  def __init__(
    self, boxes: ObjectBoxes, window_s: float, n: int, device, step_s: float = 0.02
  ):
    self.boxes, self.window_s, self.step_s = boxes, window_s, step_s
    self.person = boxes.severity == SEVERITY["person"]
    self.glass = boxes.severity == SEVERITY["glass"]
    z = lambda: torch.zeros(n, device=device)  # noqa: E731
    f = lambda: torch.zeros(n, dtype=torch.bool, device=device)  # noqa: E731
    nan = lambda: torch.full((n,), float("nan"), device=device)  # noqa: E731
    inf = lambda: torch.full((n,), float("inf"), device=device)  # noqa: E731
    self.collided, self.collided_person, self.collided_glass = f(), f(), f()
    self.peak_force_n, self.peak_force_person_n = z(), z()
    self.impulse_ns = z()
    self.damage, self.severity = z(), z()
    self.first_contact_s, self.impact_speed = nan(), nan()
    self.min_dist_m, self.min_dist_person_m = inf(), inf()
    self.end_dist_m, self.end_dist_person_m = nan(), nan()
    self.fell, self.complete = f(), f()

  def update(
    self,
    since_s: torch.Tensor,
    body: torch.Tensor,
    contacts: torch.Tensor,
    touch: torch.Tensor,
    force: torch.Tensor,
    speed2: torch.Tensor,
    down: torch.Tensor,
  ) -> None:
    """One step: ``since_s [n]`` time since each env's fault (NaN: none);
    ``body [n, P, 3]`` link origins and ``contacts [n, N, 3]`` contact points
    (layout frame), ``touch [n, N]`` the contacts to count, ``force [n, N]``
    their force (N), ``speed2 [n]`` the trunk's squared planar speed,
    ``down [n]`` whether it has fallen."""
    live = (since_s >= 0) & (since_s <= self.window_s) & ~self.complete
    if not live.any():
      return
    hit = object_contacts(self.boxes, contacts, touch)  # [n, N, I]
    any_hit = hit.any(-1)
    on = any_hit.any(-1) & live
    first = on & self.first_contact_s.isnan()
    self.first_contact_s = torch.where(first, since_s, self.first_contact_s)
    self.impact_speed = torch.where(first, speed2.sqrt(), self.impact_speed)
    self.collided |= on
    self.collided_person |= (hit & self.person).any((1, 2)) & live
    self.collided_glass |= (hit & self.glass).any((1, 2)) & live
    peak = torch.where(any_hit, force, 0.0).amax(-1)
    peak_p = torch.where((hit & self.person).any(-1), force, 0.0).amax(-1)
    self.impulse_ns = self.impulse_ns + torch.where(live, peak, 0.0) * self.step_s
    self.peak_force_n = torch.where(
      live, torch.maximum(self.peak_force_n, peak), self.peak_force_n
    )
    self.peak_force_person_n = torch.where(
      live, torch.maximum(self.peak_force_person_n, peak_p), self.peak_force_person_n
    )
    sev, dmg = price_touches(self.boxes, contacts, touch, speed2)
    self.severity = torch.where(live, torch.maximum(self.severity, sev), self.severity)
    self.damage = torch.where(live, torch.maximum(self.damage, dmg), self.damage)
    d = object_distance(self.boxes, body)  # [n, I]
    d_any = d.amin(-1)
    d_person = torch.where(self.person, d, float("inf")).amin(-1)
    self.min_dist_m = torch.where(
      live, torch.minimum(self.min_dist_m, d_any), self.min_dist_m
    )
    self.min_dist_person_m = torch.where(
      live, torch.minimum(self.min_dist_person_m, d_person), self.min_dist_person_m
    )
    self.fell |= down & live
    end = live & (since_s >= self.window_s)
    self.end_dist_m = torch.where(end, d_any, self.end_dist_m)
    self.end_dist_person_m = torch.where(end, d_person, self.end_dist_person_m)
    self.complete |= end

  def results(self) -> dict[str, list]:
    keys = (
      "collided",
      "collided_person",
      "collided_glass",
      "peak_force_n",
      "peak_force_person_n",
      "impulse_ns",
      "damage",
      "severity",
      "first_contact_s",
      "impact_speed",
      "min_dist_m",
      "min_dist_person_m",
      "end_dist_m",
      "end_dist_person_m",
      "fell",
      "complete",
    )
    return {k: getattr(self, k).cpu().tolist() for k in keys}
