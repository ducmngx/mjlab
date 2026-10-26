"""Train the FailBench impact model on built sample shards.

Two phases, as planned: ``--map-mode clean`` learns the dynamics from the
full map, then ``--map-mode fused --init-from <phase 1 checkpoint>``
fine-tunes on what the head LiDAR could see (occlusion, noise, dropout).
Run ``failbench.data.stats`` on the root first: the model normalises its
inputs with ``stats.json``, which is saved in every checkpoint.

Logs to tensorboard under ``output_dir``; ``--wandb`` also logs to W&B (on
the cluster only, never from a local run without asking). Saves
``last.pt`` every epoch and ``best.pt`` on the lowest validation loss.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import torch
import tyro
import wandb
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from failbench.data.dataset import FailBenchDataset, MapMode
from failbench.model.losses import (
  LossWeights,
  impact_loss,
  impact_metrics,
  summarize_metrics,
)
from failbench.model.model import ImpactModel, ImpactModelCfg


@dataclass(frozen=True)
class TrainConfig:
  root: str
  """Built sample shards (with ``stats.json``)."""
  output_dir: str
  map_mode: MapMode = "clean"
  init_from: str | None = None
  """Checkpoint to start from (phase 2 starts from phase 1's best)."""
  epochs: int = 20
  max_steps: int | None = None
  """Stop after this many optimiser steps (overfit checks)."""
  batch_size: int = 256
  lr: float = 3e-4
  weight_decay: float = 0.05
  warmup_steps: int = 500
  grad_clip: float = 1.0
  num_workers: int = 8
  amp: bool = True
  """bf16 autocast on CUDA."""
  device: str | None = None
  seed: int = 0
  wandb: bool = False
  wandb_project: str = "failbench"
  run_name: str | None = None
  log_every: int = 50
  fall_label: Literal["upright", "safe"] = "upright"
  """What the fall hazard learns: the fall itself, or the point of no
  return before it (``safe``, ``failbench/data/build.py``)."""
  model: ImpactModelCfg = field(default_factory=ImpactModelCfg)
  loss: LossWeights = field(default_factory=LossWeights)


def _to(batch: dict[str, torch.Tensor], device: str) -> dict[str, torch.Tensor]:
  return {k: v.to(device, non_blocking=True) for k, v in batch.items()}


def _float(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
  return {k: v.float() if v.is_floating_point() else v for k, v in batch.items()}


@torch.no_grad()
def evaluate(
  model: ImpactModel,
  loader: DataLoader,
  cfg: TrainConfig,
  device: str,
  latent_std: torch.Tensor,
) -> dict[str, float]:
  model.eval()
  losses: dict[str, float] = {}
  pooled: dict[str, list[torch.Tensor]] = {}
  n = 0
  for batch in loader:
    batch = _float(_to(batch, device))
    with torch.autocast("cuda", torch.bfloat16, enabled=cfg.amp and device != "cpu"):
      pred = model(batch)
    pred = {k: v.float() for k, v in pred.items()}
    total, terms = impact_loss(
      pred,
      batch,
      cfg.loss,
      cfg.model.steps_per_map,
      cfg.fall_label,
      latent_std,
      cfg.model.drift_nu,
    )
    b = batch["state"].shape[0]
    n += b
    for k, v in {"total": float(total), **terms}.items():
      losses[k] = losses.get(k, 0.0) + v * b
    for k, v in impact_metrics(pred, batch, fall_label=cfg.fall_label).items():
      pooled.setdefault(k, []).append(v.cpu())
  model.train()
  out = {f"loss/{k}": v / max(n, 1) for k, v in losses.items()}
  m = summarize_metrics({k: torch.cat(v) for k, v in pooled.items()})
  return out | {f"metric/{k}": v for k, v in m.items()}


def main():
  cfg = tyro.cli(TrainConfig)
  torch.manual_seed(cfg.seed)
  device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
  root, out_dir = Path(cfg.root), Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  stats_path = root / "stats.json"
  if not stats_path.is_file():
    raise FileNotFoundError(
      f"No {stats_path}: run `uv run python -m failbench.data.stats --root {root}`."
    )
  stats = json.loads(stats_path.read_text())
  latent_std = torch.tensor(stats["him_latent"]["std"], device=device)

  contacts = cfg.model.contact_heads
  train_ds = FailBenchDataset(root, "train", cfg.map_mode, contacts=contacts)

  def loader(ds: FailBenchDataset, train: bool) -> DataLoader:
    return DataLoader(
      ds,
      batch_size=cfg.batch_size,
      shuffle=train,
      drop_last=train,
      num_workers=cfg.num_workers,
      pin_memory=device != "cpu",
      persistent_workers=cfg.num_workers > 0,
    )

  train_dl = loader(train_ds, True)
  val_ds = FailBenchDataset(root, "val", cfg.map_mode, seed=0, contacts=contacts)
  val_dl = loader(val_ds, False) if len(val_ds) else None
  print(f"[INFO] train {len(train_ds)} samples, val {len(val_ds)}")

  model = ImpactModel(cfg.model, stats).to(device)
  if cfg.init_from:
    ckpt = torch.load(cfg.init_from, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    print(f"[INFO] Initialised from {cfg.init_from}")
  print(f"[INFO] {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters")
  opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
  total_steps = cfg.max_steps or cfg.epochs * len(train_dl)

  def lr_at(step: int) -> float:
    if step < cfg.warmup_steps:
      return (step + 1) / cfg.warmup_steps
    p = (step - cfg.warmup_steps) / max(total_steps - cfg.warmup_steps, 1)
    return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

  sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)
  writer = SummaryWriter(out_dir / "tb")
  run = None
  if cfg.wandb:
    run = wandb.init(project=cfg.wandb_project, name=cfg.run_name, config=asdict(cfg))

  def log(values: dict[str, float], step: int) -> None:
    for k, v in values.items():
      writer.add_scalar(k, v, step)
    if run is not None:
      run.log(values, step=step)

  def save(name: str, epoch: int, val: dict[str, float] | None) -> None:
    torch.save(
      {
        "model": model.state_dict(),
        "model_cfg": asdict(cfg.model),
        "stats": stats,
        "train_cfg": asdict(cfg),
        "epoch": epoch,
        "val": val,
      },
      out_dir / name,
    )

  step, best = 0, math.inf
  t0 = time.time()
  for epoch in range(cfg.epochs):
    for batch in train_dl:
      batch = _float(_to(batch, device))
      with torch.autocast("cuda", torch.bfloat16, enabled=cfg.amp and device != "cpu"):
        pred = model(batch)
      pred = {k: v.float() for k, v in pred.items()}
      total, terms = impact_loss(
        pred,
        batch,
        cfg.loss,
        cfg.model.steps_per_map,
        cfg.fall_label,
        latent_std,
        cfg.model.drift_nu,
      )
      opt.zero_grad(set_to_none=True)
      total.backward()
      gn = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
      opt.step()
      sched.step()
      step += 1
      if step % cfg.log_every == 0:
        log(
          {
            "train/total": float(total),
            "train/grad_norm": float(gn),
            "train/lr": sched.get_last_lr()[0],
          }
          | {f"train/{k}": v for k, v in terms.items()},
          step,
        )
        print(
          f"[step {step}] loss {float(total):.3f} "
          + " ".join(f"{k} {v:.3f}" for k, v in terms.items())
          + f" ({time.time() - t0:.0f} s)"
        )
      if cfg.max_steps and step >= cfg.max_steps:
        break
    val = (
      evaluate(model, val_dl, cfg, device, latent_std) if val_dl is not None else None
    )
    if val is not None:
      log(val, step)
      print(f"[epoch {epoch}] " + " ".join(f"{k} {v:.3f}" for k, v in val.items()))
      if val["loss/total"] < best:
        best = val["loss/total"]
        save("best.pt", epoch, val)
    save("last.pt", epoch, val)
    if cfg.max_steps and step >= cfg.max_steps:
      break
  writer.close()
  if run is not None:
    run.finish()


if __name__ == "__main__":
  main()
