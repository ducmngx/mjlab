"""How bad it is to hit each kind of object: a severity per obstacle class.

Scene geoms are named ``course_<class>_<n>`` (``failbench/door_course.py``
and the other courses); the class is the middle part. ``geom_severity``
turns the model's geom names into a per-geom severity table, which the
planning map's ray hits index (``RayCastData.hit_geom_id``) to give each
obstacle cell the severity of what it is. Unnamed or unlisted geoms (the
heat_bench course's crates, height-field terrain) get ``default``; floors
get 0, though only obstacle cells (``failbench/mppi/costmap.py``) are ever
charged.

Severity is the cost of one incident, in the planner's cost units, the
scale of a certain fall (``CostWeights.fall`` = 20) and of a metre of route
(1): MPPI's fault risk charges P(hit) times severity.
"""

from __future__ import annotations

import mujoco
import torch

SEVERITY: dict[str, float] = {
  "floor": 0.0,
  "box": 5.0,
  "wall": 10.0,
  "glass": 25.0,
  "person": 50.0,
}
DEFAULT_SEVERITY = 10.0
"""An obstacle of no listed class: as a wall."""
DROP_SEVERITY = 50.0
"""Going over a ledge (``drop_cells``), as hitting a person."""


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
) -> torch.Tensor:
  """Severity per geom ``[ngeom + 1]``; the last entry, for ray misses
  (geom id -1), is 0."""
  sev = []
  for g in range(model.ngeom):
    cls = geom_class(model.geom(g).name)
    sev.append(table.get(cls, default) if cls is not None else default)
  return torch.tensor([*sev, 0.0], device=device)
