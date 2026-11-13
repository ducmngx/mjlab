"""What the student learns from the teacher's labels, and how well it ranks.

MPPI only uses how a cycle's candidates compare (a softmin over their total
cost at ``temperature``), so the main loss is listwise: the KL from the
teacher's softmin weights to the student's over each recorded cycle's
candidates, with the total cost ``stock + weight * risk`` at a weight drawn
per cycle from ``weights``. The damages are heavy-tailed and mostly small,
so a regression on ``log1p`` of each fault's damage keeps the scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class DistillLossCfg:
  temperature: float = 0.3
  """Nav2 MPPI's (``Nav2MppiCfg.temperature``)."""
  weights: tuple[float, ...] = (0.5, 1.0, 3.0)
  """Critic weights the listwise loss is taken at (one per cycle)."""
  listwise: float = 1.0
  regression: float = 1.0


def risk(per_fault: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
  """Expected damage given a fault ``[B, K]`` of damages ``[B, K, F]``."""
  return (per_fault * prior).sum(-1)


def softmin_kl(
  teacher: torch.Tensor, student: torch.Tensor, temperature: float
) -> torch.Tensor:
  """Per cycle ``[B]``: KL(teacher's softmin weights || student's) over
  total costs ``[B, K]``."""
  log_t = F.log_softmax(-teacher / temperature, -1)
  log_s = F.log_softmax(-student / temperature, -1)
  return (log_t.exp() * (log_t - log_s)).sum(-1)


def distill_loss(
  pred: torch.Tensor,
  label: torch.Tensor,
  stock: torch.Tensor,
  prior: torch.Tensor,
  cfg: DistillLossCfg,
  generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
  """``pred`` / ``label [B, K, F]`` damages, ``stock [B, K]`` the
  planner's own critics."""
  b = pred.shape[0]
  w = torch.tensor(cfg.weights, device=pred.device)
  pick = torch.randint(len(w), (b,), generator=generator).to(pred.device)
  wb = w[pick][:, None]
  kl = softmin_kl(
    stock + wb * risk(label, prior), stock + wb * risk(pred, prior), cfg.temperature
  ).mean()
  reg = F.smooth_l1_loss(torch.log1p(pred), torch.log1p(label))
  total = cfg.listwise * kl + cfg.regression * reg
  return total, {"kl": float(kl), "reg": float(reg)}


def _ranks(x: torch.Tensor) -> torch.Tensor:
  return x.argsort(-1).argsort(-1).float()


@torch.no_grad()
def ranking_metrics(
  pred: torch.Tensor,
  label: torch.Tensor,
  stock: torch.Tensor,
  prior: torch.Tensor,
  weight: float,
  temperature: float,
  top: int = 10,
) -> dict[str, torch.Tensor]:
  """Per cycle ``[B]``: Spearman of the risk, the overlap of the ``top``
  cheapest total costs, softmin KL at ``weight``, and the risk's absolute
  error on the cycle's riskiest tenth of candidates."""
  rt, rs = risk(label, prior), risk(pred, prior)
  a, b = _ranks(rt), _ranks(rs)
  a, b = a - a.mean(-1, keepdim=True), b - b.mean(-1, keepdim=True)
  spearman = (a * b).sum(-1) / (a.norm(dim=-1) * b.norm(dim=-1)).clamp(min=1e-9)
  tt, ts = stock + weight * rt, stock + weight * rs
  k = min(top, tt.shape[-1])
  it, is_ = tt.topk(k, largest=False).indices, ts.topk(k, largest=False).indices
  overlap = (it[..., :, None] == is_[..., None, :]).any(-1).float().mean(-1)
  n = max(1, rt.shape[-1] // 10)
  worst = rt.topk(n).indices
  err = (rs.gather(-1, worst) - rt.gather(-1, worst)).abs().mean(-1)
  return {
    "spearman": spearman,
    "top_overlap": overlap,
    "kl": softmin_kl(tt, ts, temperature),
    "risky_abs_err": err,
  }
