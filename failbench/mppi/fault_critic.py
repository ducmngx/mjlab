"""The fault-risk critic: our method, as one more critic of Nav2's MPPI.

It charges each sampled trajectory for the damage a fault would do if it
fired anywhere on the route still to walk: the trajectory, then straight on
to the goal (the cost-to-go). The pricing is ``Mppi``'s
(``failbench/mppi/planner.py``, ``FaultRiskCfg``): each fault's throw,
predicted once per robot and cycle by the impact model at a few held
speeds, placed at firing points along the route and charged severity times
P(touch or fall onto) from the severity map, combined over faults and
places by the risk measure. Nothing else in the planner uses the model.
"""

from __future__ import annotations

from typing import cast

import torch

from failbench.mppi.planner import Mppi


class FaultRiskCritic:
  """``(data) -> cost [n, K]`` for ``Nav2Mppi``'s extra critics. ``data``
  carries the planner's ``traj``, ``v`` and ``goal`` and, from the runner,
  the model ``inputs``, the severity ``risk_fields`` and ``exposed [n]``
  (robots not yet faulted)."""

  __name__ = "fault_risk"

  def __init__(self, engine: Mppi, weight: float, chunk: int = 128):
    self.engine = engine
    self.weight = weight
    self.chunk = chunk

  def __call__(self, data: dict) -> torch.Tensor:
    traj, v = data["traj"], data["v"]
    fields = data.get("risk_fields")
    if not fields:
      return traj.new_zeros(traj.shape[:2])
    throws = self.engine._throws(data["inputs"])
    parts = []
    for k0 in range(0, traj.shape[1], self.chunk):
      ks = slice(k0, k0 + self.chunk)
      risk = self.engine._fault_risk(
        throws, v[:, ks], traj[:, ks], fields, data["resolution"], data["goal"]
      )
      parts.append(cast(torch.Tensor, risk))
    cost = self.weight * torch.cat(parts, 1)
    exposed = data.get("exposed")
    return cost if exposed is None else cost * exposed[:, None]
