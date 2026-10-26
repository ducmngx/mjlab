"""Losses and metrics for ``ImpactModel`` on ``FailBenchDataset`` batches.

- drift: Gaussian NLL of ``pose - cmd_path`` (yaw wrapped) over upright,
  valid steps; the variance is what a planner's ``+- k sigma`` footprint
  sweep uses, so it is trained, not optional.
- dv: Huber over upright, valid steps (auxiliary).
- fall: discrete-time hazard likelihood: ``log(1 - h)`` for every step
  before the fall, ``log h`` on the step it happens (up-weighted by
  ``fall_pos``: falls are rare), nothing after (the label, ``upright`` or
  the point of no return ``safe``, is absorbing).
- with a contact-head model: collision, offmap: BCE per valid step,
  positives up-weighted; contact maps: per map (``steps_per_map`` steps),
  "any non-foot contact in the cell", focal BCE plus soft Dice.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from failbench.faults import FaultFamily
from mjlab.utils.lab_api.math import wrap_to_pi


@dataclass(frozen=True)
class LossWeights:
  drift: float = 1.0
  dv: float = 0.5
  fall: float = 1.0
  fall_pos: float = 1.0
  """Weight on the log-likelihood of the step a fall happens."""
  collision: float = 0.5
  offmap: float = 0.1
  contact: float = 1.0
  pos_weight: float = 5.0
  """BCE weight on positive collision / offmap steps (both are rare)."""
  focal_gamma: float = 2.0


def drift_target(batch: dict[str, torch.Tensor]) -> torch.Tensor:
  d = batch["pose"] - batch["cmd_path"]
  return torch.cat([d[..., :2], wrap_to_pi(d[..., 2:])], dim=-1)


def contact_target(batch: dict[str, torch.Tensor], steps_per_map: int) -> torch.Tensor:
  c = batch["contact"]
  b, t, h, w = c.shape
  return c.reshape(b, t // steps_per_map, steps_per_map, h, w).any(dim=2).float()


def _masked_mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
  mask = mask.to(x.dtype)
  while mask.dim() < x.dim():
    mask = mask[..., None]
  mask = mask.expand_as(x)
  return (x * mask).sum() / mask.sum().clamp(min=1.0)


def hazard_nll(
  hazard_logit: torch.Tensor,
  upright: torch.Tensor,
  valid: torch.Tensor,
  fall_pos: float = 1.0,
) -> torch.Tensor:
  """Negative log-likelihood of when (if) each sample falls; the fall step's
  term weighted ``fall_pos``."""
  up = upright.bool()
  falls_here = ~up & torch.cat([torch.ones_like(up[:, :1]), up[:, :-1]], dim=1)
  ll = torch.where(
    falls_here, fall_pos * F.logsigmoid(hazard_logit), F.logsigmoid(-hazard_logit)
  )
  at_risk = (up | falls_here) & valid.bool()
  return -_masked_mean(ll, at_risk)


def _focal_bce(logit: torch.Tensor, target: torch.Tensor, gamma: float) -> torch.Tensor:
  bce = F.binary_cross_entropy_with_logits(logit, target, reduction="none")
  p = torch.sigmoid(logit)
  p_t = p * target + (1 - p) * (1 - target)
  return ((1 - p_t) ** gamma * bce).mean()


def _soft_dice(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
  p = torch.sigmoid(logit).flatten(2)
  t = target.flatten(2)
  inter = (p * t).sum(-1)
  denom = p.sum(-1) + t.sum(-1)
  has = t.sum(-1) > 0  # Dice only where the map has contacts.
  dice = 1 - (2 * inter + 1) / (denom + 1)
  return (dice * has).sum() / has.sum().clamp(min=1)


def impact_loss(
  pred: dict[str, torch.Tensor],
  batch: dict[str, torch.Tensor],
  w: LossWeights,
  steps_per_map: int,
  fall_label: str = "upright",
) -> tuple[torch.Tensor, dict[str, float]]:
  """``fall_label``: the batch key the hazard learns (``upright``, or
  ``safe`` for the point of no return)."""
  valid = batch["valid"].bool()
  up = batch["upright"].bool() & valid
  target = drift_target(batch)
  logvar = pred["drift_logvar"]
  nll = 0.5 * ((target - pred["drift"]) ** 2 * torch.exp(-logvar) + logvar)
  terms = {
    "drift": _masked_mean(nll, up),
    "dv": _masked_mean(F.huber_loss(pred["dv"], batch["dv"], reduction="none"), up),
    "fall": hazard_nll(pred["hazard_logit"], batch[fall_label], valid, w.fall_pos),
  }
  pw = torch.tensor(w.pos_weight, device=valid.device)
  for name, key in (("collision", "collision"), ("offmap", "contact_offmap")):
    if f"{name}_logit" not in pred:
      continue
    bce = F.binary_cross_entropy_with_logits(
      pred[f"{name}_logit"], batch[key].float(), pos_weight=pw, reduction="none"
    )
    terms[name] = _masked_mean(bce, valid)
  if "contact_logit" in pred:
    ct = contact_target(batch, steps_per_map)
    terms["contact"] = _focal_bce(
      pred["contact_logit"], ct, w.focal_gamma
    ) + _soft_dice(pred["contact_logit"], ct)
  total = sum(getattr(w, k) * v for k, v in terms.items())
  assert isinstance(total, torch.Tensor)
  return total, {k: float(v.detach()) for k, v in terms.items()}


def auroc(score: torch.Tensor, label: torch.Tensor) -> float:
  """Rank-based area under the ROC curve (NaN without both classes)."""
  label = label.bool()
  n_pos, n_neg = int(label.sum()), int((~label).sum())
  if n_pos == 0 or n_neg == 0:
    return float("nan")
  rank = torch.empty_like(score)
  rank[score.argsort()] = torch.arange(1, len(score) + 1, dtype=score.dtype)
  return float((rank[label].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


@torch.no_grad()
def impact_metrics(
  pred: dict[str, torch.Tensor],
  batch: dict[str, torch.Tensor],
  step_s: float = 0.1,
  fall_label: str = "upright",
  soon_s: float = 1.0,
) -> dict[str, torch.Tensor]:
  """Per-sample quantities to pool over an eval set (concatenate, then
  ``summarize_metrics``). Falls are by ``fall_label``; ``fell_soon`` /
  ``p_fell_soon``: within ``soon_s``."""
  up = batch["upright"].bool() & batch["valid"].bool()
  err = (pred["drift"][..., :2] - drift_target(batch)[..., :2]).norm(dim=-1)
  fall_up = batch[fall_label].bool()
  out = {"fell": ~fall_up.all(dim=1)}
  if "fault_family" in batch:
    out["family"] = batch["fault_family"].long()
  out["p_fell"] = pred["p_fallen"][:, -1]
  k_soon = round(soon_s / step_s) - 1
  out["fell_soon"] = ~fall_up[:, k_soon]
  out["p_fell_soon"] = pred["p_fallen"][:, k_soon]
  if "collision_logit" in pred:
    out["collided"] = (batch["collision"].bool() & batch["valid"].bool()).any(dim=1)
    out["p_collided"] = torch.sigmoid(pred["collision_logit"]).amax(dim=1)
  for s in (1, 3, 5):
    k = round(s / step_s) - 1
    out[f"drift_err_{s}s"] = torch.where(up[:, k], err[:, k], float("nan"))
    zero = drift_target(batch)[:, k, :2].norm(dim=-1)
    out[f"zero_err_{s}s"] = torch.where(up[:, k], zero, float("nan"))
    # Fraction of true drifts inside the predicted 2-sigma ellipse (axis).
    sigma = torch.exp(0.5 * pred["drift_logvar"][:, k, :2])
    inside = (
      (drift_target(batch)[:, k, :2] - pred["drift"][:, k, :2]).abs() <= 2 * sigma
    ).all(dim=-1)
    out[f"cover2s_{s}s"] = torch.where(up[:, k], inside.float(), float("nan"))
  return out


def summarize_metrics(m: dict[str, torch.Tensor]) -> dict[str, float]:
  """Pooled metrics, and the same per fault family (``nominal/...``: the
  healthy-robot dynamics MPPI plans with before any fault)."""

  def pool(sel: torch.Tensor, prefix: str) -> dict[str, float]:
    out = {
      f"{prefix}fall_auroc": auroc(m["p_fell"][sel], m["fell"][sel]),
      f"{prefix}fall_soon_auroc": auroc(m["p_fell_soon"][sel], m["fell_soon"][sel]),
    }
    if "p_collided" in m:
      out[f"{prefix}collision_auroc"] = auroc(m["p_collided"][sel], m["collided"][sel])
    for k, v in m.items():
      if k.startswith(("drift_err", "zero_err", "cover2s")):
        out[f"{prefix}{k}"] = float(torch.nanmean(v[sel].float()))
    return out

  out = pool(torch.ones_like(m["fell"]), "")
  if "family" in m:
    for f in FaultFamily:
      sel = m["family"] == f
      if sel.any():
        out |= pool(sel, f"{f.name.lower()}/")
  return out
