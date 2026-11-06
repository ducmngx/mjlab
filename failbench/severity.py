"""How bad it is to hit each kind of object: a severity per obstacle class.

Scene geoms are named ``course_<class>_<n>`` (``failbench/door_course.py``
and the other courses); the class is the middle part. ``geom_severity``
turns the model's geom names into a per-geom severity table, which the
planning map's ray hits index (``RayCastData.hit_geom_id``) to give each
obstacle cell the severity of what it is. Unnamed or unlisted geoms (the
heat_bench course's crates) get ``default``; floors and the terrain itself
(``terrain_*``: rough tiles, stairs, slopes) get 0: touching the ground is
no damage. Only obstacle cells (``failbench/mppi/costmap.py``) are ever
charged.

Severity is the cost of one incident, in the planner's cost units, the
scale of a certain fall (``CostWeights.fall`` = 20) and of a metre of route
(1): MPPI's fault risk charges P(hit) times severity.

How much of it a hit does depends on its energy against the object's
``ENERGY_SCALE`` (``damage``): a person is hurt by any knock, so the damage
is nearly the whole severity at walking speed and only keeping away helps;
a box is only broken by a hard hit, so slowing near it helps as much. A
fall is priced the same way, by the trunk's kinetic energy as it is thrown
into the object: the Go2 is low, so what it falls onto is what it was
thrown sideways into, not what is under it.
"""

from __future__ import annotations

import mujoco
import torch

SEVERITY: dict[str, float] = {
  "floor": 0.0,
  "box": 8.0,
  "wall": 4.0,
  "glass": 25.0,
  "person": 50.0,
}
DEFAULT_SEVERITY = SEVERITY["wall"]
"""An obstacle of no listed class: as a wall."""
DROP_SEVERITY = 50.0
"""Going over a ledge (``drop_cells``), as hitting a person."""

ENERGY_SCALE: dict[str, float] = {
  "floor": 1.0,
  "box": 20.0,
  "wall": 5.0,
  "glass": 5.0,
  "person": 1.0,
}
"""The trunk's kinetic energy (J) at which a hit does 63% of the severity
(``damage``). Placeholders: the person's sits in the range ISO/TS
15066's transient-contact limits give in the robot's kinetic energy,
``(m_robot / mu) F^2 / 2k`` with ``mu`` the robot and body part's reduced
mass: 0.5 J (hand, pinned) to 3.5 J (back, free) from the Annex A values
quoted by Kirschner et al., "ISO/TS 15066: How Different Interpretations
Affect Risk Assessment", arXiv:2203.02706. The Go2 hits shins and knees,
whose values are still to be taken from Annex A; the rest are set by hand."""
DEFAULT_ENERGY_SCALE = ENERGY_SCALE["wall"]
DROP_ENERGY_SCALE = ENERGY_SCALE["person"]
"""Going over a ledge is bad at any speed."""
MASS_KG = 15.0
"""The Go2's mass, for the trunk's kinetic energy."""

LEVEL_ENERGY_SCALE: dict[float, float] = {
  DEFAULT_SEVERITY: DEFAULT_ENERGY_SCALE,
  DROP_SEVERITY: DROP_ENERGY_SCALE,
  **{SEVERITY[k]: ENERGY_SCALE[k] for k in SEVERITY if SEVERITY[k] > 0},
}
"""Energy scale per severity level, for the maps (``severity_fields``),
which carry only the severity: classes sharing a severity share a scale."""


def damage(severity, energy, scale):
  """The damage of a hit at kinetic ``energy`` (J) on an object of
  ``severity`` and energy ``scale``: ``severity * (1 - exp(-energy /
  scale))``. Works on floats and tensors alike."""
  return severity * -torch.expm1(-torch.as_tensor(energy) / scale)


def geom_class(name: str) -> str | None:
  """``course_<class>_<n>`` -> ``<class>``; None for any other name."""
  parts = name.split("_")
  if len(parts) >= 3 and parts[0] == "course":
    return "_".join(parts[1:-1])
  return None


def geom_severity(
  model: mujoco.MjModel,
  table: dict[str, float] = SEVERITY,
  default: float = DEFAULT_SEVERITY,
  device: torch.device | str = "cpu",
  ground: float = 0.0,
) -> torch.Tensor:
  """Severity per geom ``[ngeom + 1]``; the last entry, for ray misses
  (geom id -1), is 0. With ``ENERGY_SCALE`` as the ``table`` (and
  ``ground`` 1), the energy scale per geom."""
  sev = []
  for g in range(model.ngeom):
    name = model.geom(g).name
    cls = geom_class(name)
    if name.startswith("terrain"):
      sev.append(ground)
    else:
      sev.append(table.get(cls, default) if cls is not None else default)
  return torch.tensor([*sev, ground], device=device)
