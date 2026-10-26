"""Evaluate a trained impact model on the held-out splits.

For each split (``val``, ``test_terrain``: unseen terrain seeds,
``test_fault``: the held-out joint and fault pair), pooled and per fault
family and terrain:

- drift error at 1, 3 and 5 s against the zero-drift baseline, and the share
  of true drifts inside the predicted 2 sigma (``impact_metrics``);
- fall AUROC, by 5 s and within ``soon_s`` (1 s), on the model's own fall
  label (``train_cfg.fall_label``: the fall, or the point of no return);
- calibration of P(fall within 5 s) and within ``soon_s`` (and, for a
  contact-head model, of the highest per-step P(trunk contact)): Brier score,
  expected calibration error over ``bins`` equal-width bins, and the bins
  themselves (a reliability diagram);
- warning lead time, as a monitor would use the model: an alarm when
  P(fall within ``soon_s``) passes the threshold that flags
  ``false_alarm`` of the samples that don't fall in 5 s; then, by how long
  until the robot is really on the ground (``upright``, whatever the
  label), the share of falls warned. Comparable across models trained on
  either label;
- all of it per fault group (healthy, locked, weak, limp: torque 0).

Writes ``<output_dir>/eval.json`` and ``reliability.png``. The map the model
sees follows its training (``train_cfg.map_mode``) unless ``map_mode`` is
given.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import matplotlib
import numpy as np
import torch
import tyro
from torch.utils.data import DataLoader

from failbench.data.build import TERRAINS
from failbench.data.dataset import FailBenchDataset, MapMode
from failbench.faults import FaultFamily
from failbench.model.losses import auroc, impact_metrics, summarize_metrics
from failbench.mppi.models import ImpactDynamics

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@dataclass(frozen=True)
class EvalConfig:
  checkpoint: str
  root: str
  """Built sample shards."""
  output_dir: str
  splits: tuple[str, ...] = ("val", "test_terrain", "test_fault")
  map_mode: MapMode | None = None
  batch_size: int = 512
  num_workers: int = 8
  bins: int = 10
  soon_s: float = 1.0
  false_alarm: float = 0.05
  fall_label: str | None = None
  """Overrides the checkpoint's (``upright`` / ``safe``)."""
  device: str | None = None


def calibration(p: np.ndarray, y: np.ndarray, bins: int) -> dict:
  """Brier score, expected calibration error and per-bin (mean predicted,
  observed rate, count)."""
  edges = np.linspace(0.0, 1.0, bins + 1)
  idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
  rows, ece = [], 0.0
  for b in range(bins):
    sel = idx == b
    if not sel.any():
      continue
    pred, obs = float(p[sel].mean()), float(y[sel].mean())
    ece += sel.mean() * abs(pred - obs)
    rows.append({"pred": pred, "obs": obs, "n": int(sel.sum())})
  return {"brier": float(((p - y) ** 2).mean()), "ece": float(ece), "bins": rows}


LEAD_BINS = ((0.0, 0.5), (0.5, 1.0), (1.0, 2.0), (2.0, 5.0))
"""Time to the ground fall (s) the warned share is reported over."""
GROUPS = ("healthy", "locked", "weak", "limp")


def fault_group(fault: torch.Tensor) -> torch.Tensor:
  """Encoded faults ``[B, 24]`` -> group index into ``GROUPS``."""
  p = fault.view(-1, 12, 2)
  torque, lock = p[..., 0].amin(-1), p[..., 1].amax(-1)
  return torch.where(
    lock > 0.5, 1, torch.where(torque <= 1e-6, 3, torch.where(torque < 1, 2, 0))
  )


def warning_lead(
  p_soon: np.ndarray, fall_t: np.ndarray, false_alarm: float, horizon_s: float = 5.0
) -> dict:
  """Alarm at the P(fall soon) that flags ``false_alarm`` of the samples
  that stay up for ``horizon_s``; the share of falls warned per time to the
  fall (``LEAD_BINS``)."""
  stays = fall_t > horizon_s
  if not stays.any() or stays.all():
    return {}
  thr = float(np.quantile(p_soon[stays], 1 - false_alarm))
  alarm = p_soon >= thr
  out: dict[str, Any] = {"threshold": thr, "false_alarm": float(alarm[stays].mean())}
  for lo, hi in LEAD_BINS:
    sel = (fall_t > lo) & (fall_t <= hi)
    out[f"warned_{lo:g}_{hi:g}s"] = float(alarm[sel].mean()) if sel.any() else None
    out[f"n_{lo:g}_{hi:g}s"] = int(sel.sum())
  return out


@torch.no_grad()
def evaluate_split(
  dyn: ImpactDynamics, ds: FailBenchDataset, cfg: EvalConfig, device, fall_label: str
):
  model = dyn.model
  dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
  pooled: dict[str, list[torch.Tensor]] = {}
  extra: dict[str, list[np.ndarray]] = {k: [] for k in ("terrain", "fall_t", "group")}
  step_s = 0.1
  for batch in dl:
    batch = {k: v.to(device) for k, v in batch.items()}
    batch = {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}
    with torch.autocast("cuda", torch.bfloat16, enabled=device != "cpu"):
      # Contact maps aren't scored here; skipping them saves most of the
      # memory and time.
      ctx = model.encode(batch)
      pred = model.decode(ctx, batch["cmd_seq"], batch["cmd_path"], False)
    pred = {k: v.float() for k, v in pred.items()}
    for k, v in impact_metrics(pred, batch, step_s, fall_label, cfg.soon_s).items():
      pooled.setdefault(k, []).append(v.cpu())
    up = batch["upright"].bool()
    t = torch.arange(up.shape[1], device=device)
    # Time to the ground fall; past the horizon for samples that stay up.
    fall_t = (torch.where(~up, t, up.shape[1] + 9).amin(1) + 1) * step_s
    extra["terrain"].append(batch["terrain_id"].cpu().numpy())
    extra["fall_t"].append(fall_t.cpu().numpy())
    extra["group"].append(fault_group(batch["fault"]).cpu().numpy())
  m = {k: torch.cat(v) for k, v in pooled.items()}
  e = {k: np.concatenate(v) for k, v in extra.items()}
  out: dict[str, Any] = dict(summarize_metrics(m))
  out["samples"] = len(m["fell"])
  out["fall_label"] = fall_label

  def falls(sel: np.ndarray) -> dict:
    r: dict[str, Any] = {
      "samples": int(sel.sum()),
      "fall_calibration": calibration(
        m["p_fell"].numpy()[sel], m["fell"].numpy()[sel].astype(float), cfg.bins
      ),
      "fall_soon_calibration": calibration(
        m["p_fell_soon"].numpy()[sel],
        m["fell_soon"].numpy()[sel].astype(float),
        cfg.bins,
      ),
      "warning": warning_lead(
        m["p_fell_soon"].numpy()[sel], e["fall_t"][sel], cfg.false_alarm
      ),
    }
    return r

  out |= falls(np.ones(len(e["group"]), bool))
  if "p_collided" in m:
    out["collision_calibration"] = calibration(
      m["p_collided"].numpy(), m["collided"].numpy().astype(float), cfg.bins
    )
  out["per_group"] = {
    g: falls(e["group"] == i) for i, g in enumerate(GROUPS) if (e["group"] == i).any()
  }
  per_terrain = {}
  for tid, name in enumerate(TERRAINS):
    sel = torch.as_tensor(e["terrain"] == tid)
    if sel.sum() < 20:
      continue
    s = {k: v[sel] for k, v in m.items()}
    per_terrain[name] = {
      "samples": int(sel.sum()),
      "drift_err_5s": float(torch.nanmean(s["drift_err_5s"].float())),
      "zero_err_5s": float(torch.nanmean(s["zero_err_5s"].float())),
      "fall_auroc": auroc(s["p_fell"], s["fell"]),
    }
  out["per_terrain"] = per_terrain
  return out


def main():
  cfg = tyro.cli(EvalConfig)
  device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
  dyn = ImpactDynamics.from_checkpoint(cfg.checkpoint, device, amp=False)
  train_cfg = torch.load(cfg.checkpoint, map_location="cpu", weights_only=False)[
    "train_cfg"
  ]
  fall_label = cfg.fall_label or train_cfg.get("fall_label", "upright")
  contacts = dyn.model.cfg.contact_heads
  map_mode = cast(MapMode, cfg.map_mode or dyn.map_mode)
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  splits: dict[str, dict] = {}
  results = {"checkpoint": cfg.checkpoint, "map_mode": map_mode, "splits": splits}
  for split in cfg.splits:
    ds = FailBenchDataset(cfg.root, split, map_mode, seed=0, contacts=contacts)
    if not len(ds):
      print(f"[WARN] No samples in {split}.")
      continue
    print(f"[INFO] {split}: {len(ds)} samples")
    r = evaluate_split(dyn, ds, cfg, device, fall_label)
    splits[split] = r
    fam = " ".join(
      f"{f.name.lower()} {r.get(f'{f.name.lower()}/drift_err_5s', float('nan')):.2f}"
      for f in FaultFamily
    )
    w = r["warning"]
    warned = " ".join(
      f"{lo:g}-{hi:g}s {w.get(f'warned_{lo:g}_{hi:g}s') or float('nan'):.2f}"
      for lo, hi in LEAD_BINS
    )
    print(
      f"[INFO] {split}: drift 5 s {r['drift_err_5s']:.3f} m (zero"
      f" {r['zero_err_5s']:.3f}; {fam}), cover2s {r['cover2s_5s']:.3f}, fall"
      f" AUROC {r['fall_auroc']:.3f} (within {cfg.soon_s:g} s"
      f" {r['fall_soon_auroc']:.3f}, ECE {r['fall_soon_calibration']['ece']:.3f});"
      f" warned at {cfg.false_alarm:.0%} false alarms, by time to fall: {warned}"
    )
  (out_dir / "eval.json").write_text(json.dumps(results, indent=2))

  fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
  for a, key, title in (
    (ax[0], "fall_calibration", "P(fall within 5 s)"),
    (ax[1], "fall_soon_calibration", f"P(fall within {cfg.soon_s:g} s)"),
  ):
    a.plot([0, 1], [0, 1], "--", color="#888888", lw=1)
    for split, r in splits.items():
      b = r[key]["bins"]
      a.plot(
        [x["pred"] for x in b],
        [x["obs"] for x in b],
        "o-",
        label=f"{split} (ECE {r[key]['ece']:.3f})",
      )
    a.set_xlabel("predicted")
    a.set_ylabel("observed rate")
    a.set_title(title)
    a.set_xlim(0, 1)
    a.set_ylim(0, 1)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
  plt.tight_layout()
  plt.savefig(out_dir / "reliability.png", dpi=90)
  print(f"[INFO] -> {out_dir}")


if __name__ == "__main__":
  main()
