"""Record the risk critic's labels for the distilled critic, from
``run_mppi`` cycles (``--distill-dump``).

Labels come from the teacher (``RiskCritic``), not from simulated faults:
every cycle it prices every MPPI sample anyway. Each recorded robot-cycle
keeps what the student needs to reproduce that price:

- the planner's map (``height``, float16, NaN unknown) and per-cell
  ``severity`` (float16, any value), from which ``map_layers`` builds the
  student's layers at training time;
- the impact model's inputs (``state``, ``him_latent``, ``history``,
  ``map``, ``fault``) for the frozen context encoder;
- a subset of the cycle's candidates (``commands``, as sent, deadbanded):
  the ``top`` the planner likes best (stock cost plus the critic's) and
  ``rand`` more at random, with their ``stock_cost`` and the teacher's
  damage per fault (``per_fault [S, K, F]``).

Shards of ``shard_size`` robot-cycles go to ``<prefix>_<i>.npz``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

INPUT_KEYS = ("state", "him_latent", "history", "map", "fault")


class LabelDump:
  def __init__(
    self,
    prefix: str | Path,
    top: int = 64,
    rand: int = 192,
    shard_size: int = 2048,
    seed: int = 0,
  ):
    self.prefix = Path(prefix)
    self.prefix.parent.mkdir(parents=True, exist_ok=True)
    self.top, self.rand, self.shard_size = top, rand, shard_size
    self.gen = torch.Generator().manual_seed(seed)
    self.rows: dict[str, list[np.ndarray]] = {}
    self.count = 0
    self.shards = 0
    self.written: list[Path] = []

  def pick(self, total_cost: torch.Tensor) -> torch.Tensor:
    """Candidate indices ``[n, top + rand]`` of costs ``[n, K]``: the
    cheapest ``top``, then ``rand`` of the rest at random."""
    n, k = total_cost.shape
    order = total_cost.argsort(-1).cpu()
    rest = order[:, self.top :]
    take = torch.rand(n, rest.shape[1], generator=self.gen).argsort(-1)[:, : self.rand]
    return torch.cat([order[:, : self.top], rest.gather(1, take)], 1)

  def record(
    self,
    last: dict[str, Any],
    total_cost: torch.Tensor,
    inputs: dict[str, torch.Tensor],
    height: torch.Tensor,
    severity: torch.Tensor,
    keep: torch.Tensor,
    extra: dict[str, np.ndarray] | None = None,
  ) -> None:
    """One cycle: the critic's ``last`` (``per_fault``, ``commands``,
    ``stock_cost``; all kept as they are if ``picked``), the planner's ``total_cost [n, K]``, the impact
    model's ``inputs``, the planner's ``height`` / ``severity [n, H, W]``,
    for robots ``keep [n]``; ``extra`` per-robot arrays (e.g. the step)."""
    ids = keep.nonzero()[:, 0].cpu()
    if len(ids) == 0:
      return
    if last.get("picked"):  # The critic priced only the samples it picked.
      idx = torch.arange(last["per_fault"].shape[1]).expand(len(ids), -1)
    else:
      idx = self.pick(total_cost[ids])

    def rows(x: torch.Tensor) -> torch.Tensor:
      return x.cpu()[ids]

    def pick(x: torch.Tensor) -> torch.Tensor:
      at = idx.view(*idx.shape, *([1] * (x.dim() - 2)))
      return rows(x).gather(1, at.expand(-1, -1, *x.shape[2:]))

    out = {
      "height": rows(height).half().numpy(),
      "severity": rows(severity).half().numpy(),
      "commands": pick(last["commands"]).half().numpy(),
      "stock_cost": pick(last["stock_cost"]).float().numpy(),
      "per_fault": pick(last["per_fault"]).float().numpy(),
    }
    for key in INPUT_KEYS:
      x = rows(inputs[key])
      out[key] = (x.half() if key == "map" else x.float()).numpy()
    for key, v in (extra or {}).items():
      out[key] = np.asarray(v)[ids.numpy()]
    for key, v in out.items():
      self.rows.setdefault(key, []).append(v)
    self.count += len(ids)
    if self.count >= self.shard_size:
      self.flush()

  def flush(self) -> None:
    if not self.count:
      return
    path = self.prefix.parent / f"{self.prefix.name}_{self.shards:03d}.npz"
    arrays: dict[str, Any] = {k: np.concatenate(v) for k, v in self.rows.items()}
    np.savez(path, **arrays)
    self.written.append(path)
    self.rows, self.count = {}, 0
    self.shards += 1
