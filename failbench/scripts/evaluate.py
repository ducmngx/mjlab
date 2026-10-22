"""Evaluate a trained impact model on the held-out splits.

For each split (``val``, ``test_terrain``: unseen terrain seeds,
``test_fault``: the held-out joint and fault pair), pooled and per fault
family and terrain:

- drift error at 1, 3 and 5 s against the zero-drift baseline, and the share
  of true drifts inside the predicted 2 sigma (``impact_metrics``);
- fall and trunk-contact AUROC;
- calibration of P(fallen by 5 s) and of the highest per-step P(trunk
  contact): Brier score, expected calibration error over ``bins`` equal-width
  bins, and the bins themselves (a reliability diagram);
- fall timing, on samples that fall: when P(fallen) first crosses 0.5 against
  the true fall time.

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


@torch.no_grad()
def evaluate_split(dyn: ImpactDynamics, ds: FailBenchDataset, cfg: EvalConfig, device):
  model = dyn.model
  dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=cfg.num_workers)
  pooled: dict[str, list[torch.Tensor]] = {}
  extra: dict[str, list[np.ndarray]] = {
    k: [] for k in ("terrain", "fall_t", "pred_fall_t")
  }
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
    for k, v in impact_metrics(pred, batch).items():
      pooled.setdefault(k, []).append(v.cpu())
    up = batch["upright"].bool()
    t = torch.arange(up.shape[1], device=device)
    big = up.shape[1]
    fall_t = torch.where(~up, t, big).amin(1)
    cross = torch.where(pred["p_fallen"] >= 0.5, t, big).amin(1)
    extra["terrain"].append(batch["terrain_id"].cpu().numpy())
    extra["fall_t"].append(((fall_t + 1) * step_s).cpu().numpy())
    extra["pred_fall_t"].append(((cross + 1) * step_s).cpu().numpy())
  m = {k: torch.cat(v) for k, v in pooled.items()}
  e = {k: np.concatenate(v) for k, v in extra.items()}
  out: dict[str, Any] = dict(summarize_metrics(m))
  out["samples"] = len(m["fell"])

  fell = m["fell"].numpy().astype(float)
  coll = m["collided"].numpy().astype(float)
  out["fall_calibration"] = calibration(m["p_fell"].numpy(), fell, cfg.bins)
  out["collision_calibration"] = calibration(m["p_collided"].numpy(), coll, cfg.bins)
  falls = fell > 0.5
  if falls.any():
    seen = falls & (e["pred_fall_t"] <= 5.0)
    out["fall_timing"] = {
      "falls": int(falls.sum()),
      "predicted_within_5s": float(seen.sum() / falls.sum()),
      "mean_abs_error_s": float(
        np.abs(e["pred_fall_t"][seen] - e["fall_t"][seen]).mean()
      )
      if seen.any()
      else None,
      "mean_lead_s": float((e["fall_t"][seen] - e["pred_fall_t"][seen]).mean())
      if seen.any()
      else None,
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
  map_mode = cast(MapMode, cfg.map_mode or dyn.map_mode)
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  splits: dict[str, dict] = {}
  results = {"checkpoint": cfg.checkpoint, "map_mode": map_mode, "splits": splits}
  for split in cfg.splits:
    ds = FailBenchDataset(cfg.root, split, map_mode, seed=0)
    if not len(ds):
      print(f"[WARN] No samples in {split}.")
      continue
    print(f"[INFO] {split}: {len(ds)} samples")
    r = evaluate_split(dyn, ds, cfg, device)
    splits[split] = r
    fam = " ".join(
      f"{f.name.lower()} {r.get(f'{f.name.lower()}/drift_err_5s', float('nan')):.2f}"
      for f in FaultFamily
    )
    print(
      f"[INFO] {split}: drift 5 s {r['drift_err_5s']:.3f} m (zero"
      f" {r['zero_err_5s']:.3f}; {fam}), cover2s {r['cover2s_5s']:.3f}, fall"
      f" AUROC {r['fall_auroc']:.3f} ECE {r['fall_calibration']['ece']:.3f},"
      f" contact AUROC {r['collision_auroc']:.3f} ECE"
      f" {r['collision_calibration']['ece']:.3f}"
    )
  (out_dir / "eval.json").write_text(json.dumps(results, indent=2))

  fig, ax = plt.subplots(1, 2, figsize=(11, 4.6))
  for a, key, title in (
    (ax[0], "fall_calibration", "P(fallen by 5 s)"),
    (ax[1], "collision_calibration", "highest P(trunk contact)"),
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
