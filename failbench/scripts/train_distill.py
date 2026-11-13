"""Train the distilled risk critic (``failbench/distill``) on the risk
critic's recorded labels (``run_mppi --distill-dump``).

Each batch is ``batch_size`` recorded robot-cycles with all their stored
candidates. The map layers are built on the GPU from the stored map
(``map_layers``) and the robot's context from the frozen impact model's
encoder (``ImpactContext``), exactly as the critic builds them in the loop.
Validation runs on separate shards (held-out layouts). Logs to tensorboard
under ``output_dir``; ``--wandb`` also to W&B (on the cluster only). Saves
``last.pt`` every epoch and ``best.pt`` on the best validation KL.
"""

from __future__ import annotations

import glob
import json
import math
import random
import time
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import tyro
import wandb
from torch.utils.tensorboard import SummaryWriter

from failbench.distill.dump import INPUT_KEYS
from failbench.distill.layers import map_layers
from failbench.distill.losses import DistillLossCfg, distill_loss, ranking_metrics
from failbench.distill.model import Student, StudentCfg
from failbench.mppi.distilled_critic import ImpactContext
from failbench.mppi.models import ImpactDynamics
from failbench.mppi.risk_critic import (
  REPRESENTATIVE_FAULTS,
  RiskCriticCfg,
  imagined_faults,
)

MAP_RESOLUTION = 0.05


@dataclass(frozen=True)
class DistillTrainConfig:
  train: tuple[str, ...]
  """Globs of label shards to train on."""
  val: tuple[str, ...]
  """Globs of held-out shards (other layouts)."""
  impact_model: str
  """The impact model whose encoder gives the context (the teacher's)."""
  output_dir: str
  epochs: int = 20
  max_steps: int | None = None
  batch_size: int = 32
  """Robot-cycles per step (each with all its stored candidates)."""
  lr: float = 3e-4
  weight_decay: float = 0.01
  warmup_steps: int = 300
  grad_clip: float = 1.0
  amp: bool = True
  device: str | None = None
  seed: int = 0
  wandb: bool = False
  wandb_project: str = "failbench"
  run_name: str | None = None
  log_every: int = 50
  faults: tuple[str, ...] = REPRESENTATIVE_FAULTS
  """The teacher's imagined faults, in its order (the labels' F axis)."""
  student: StudentCfg = field(default_factory=StudentCfg)
  loss: DistillLossCfg = field(default_factory=DistillLossCfg)


def shards(patterns: tuple[str, ...]) -> list[str]:
  files = sorted({f for p in patterns for f in glob.glob(p)})
  if not files:
    raise FileNotFoundError(f"No shards match {patterns}.")
  return files


def batches(
  files: list[str], size: int, shuffle: bool, rng: random.Random
) -> Iterator[dict[str, np.ndarray]]:
  """Batches of robot-cycles, a shard at a time (shards and rows shuffled
  for training)."""
  order = list(files)
  if shuffle:
    rng.shuffle(order)
  for f in order:
    with np.load(f) as d:
      data = {k: d[k] for k in d.files}
    rows = list(range(len(data["commands"])))
    if shuffle:
      rng.shuffle(rows)
    for i in range(0, len(rows), size):
      idx = rows[i : i + size]
      yield {k: v[idx] for k, v in data.items()}


class Prepare:
  """Turns a stored batch into the student's inputs and labels, on the
  device."""

  def __init__(self, encoder: ImpactContext, cfg: DistillTrainConfig, device: str):
    self.encoder = encoder
    self.footprint = RiskCriticCfg().footprint
    self.device = device

  def __call__(self, batch: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    t = lambda k: torch.as_tensor(batch[k], device=self.device).float()  # noqa: E731
    inputs = {k: t(k) for k in INPUT_KEYS}
    return {
      "layers": map_layers(t("height"), t("severity"), MAP_RESOLUTION, self.footprint),
      "context": self.encoder.context(inputs),
      "commands": t("commands"),
      "stock": t("stock_cost"),
      "label": t("per_fault"),
    }


@torch.no_grad()
def evaluate(
  student: Student,
  files: list[str],
  prep: Prepare,
  prior: torch.Tensor,
  cfg: DistillTrainConfig,
  device: str,
) -> dict[str, float]:
  student.eval()
  pooled: dict[str, list[torch.Tensor]] = {}
  loss_sum, n = 0.0, 0
  gen = torch.Generator().manual_seed(0)
  for raw in batches(files, cfg.batch_size, False, random.Random(0)):
    b = prep(raw)
    with torch.autocast("cuda", torch.bfloat16, enabled=cfg.amp and device != "cpu"):
      pred = student(b["layers"], b["context"], b["commands"])["damage"]
    pred = pred.float()
    total, _ = distill_loss(pred, b["label"], b["stock"], prior, cfg.loss, gen)
    loss_sum += float(total) * len(pred)
    n += len(pred)
    for w in cfg.loss.weights:
      m = ranking_metrics(pred, b["label"], b["stock"], prior, w, cfg.loss.temperature)
      for k, v in m.items():
        pooled.setdefault(f"{k}_w{w}", []).append(v.cpu())
  student.train()
  out = {"val/loss": loss_sum / max(n, 1)}
  for k, v in pooled.items():
    out[f"val/{k}"] = float(torch.cat(v).nanmean())
  return out


def main():
  cfg = tyro.cli(DistillTrainConfig)
  torch.manual_seed(cfg.seed)
  rng = random.Random(cfg.seed)
  device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
  out_dir = Path(cfg.output_dir)
  out_dir.mkdir(parents=True, exist_ok=True)
  (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2))
  train_files, val_files = shards(cfg.train), shards(cfg.val)
  print(f"[distill] {len(train_files)} train shards, {len(val_files)} val shards")

  impact = ImpactDynamics.from_checkpoint(cfg.impact_model, device)
  student_cfg = cfg.student
  ctx_dim = impact.model.cfg.d_model
  if student_cfg.context_dim != ctx_dim:
    raise ValueError(f"student.context_dim must be the encoder's d_model ({ctx_dim}).")
  if student_cfg.num_faults != len(cfg.faults):
    raise ValueError("student.num_faults must match the teacher's faults.")
  student = Student(student_cfg).to(device)
  prep = Prepare(ImpactContext(impact), cfg, device)
  _, prior = imagined_faults(RiskCriticCfg(faults=cfg.faults), 12)
  prior = prior.to(device)

  opt = torch.optim.AdamW(
    student.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )
  sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps)
  )
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
        "student_cfg": student_cfg.to_saved(),
        "student": student.state_dict(),
        "faults": list(cfg.faults),
        "impact_model": cfg.impact_model,
        "train_cfg": asdict(cfg),
        "epoch": epoch,
        "val": val,
      },
      out_dir / name,
    )

  step, best = 0, math.inf
  gen = torch.Generator().manual_seed(cfg.seed)
  for epoch in range(cfg.epochs):
    t0 = time.perf_counter()
    for raw in batches(train_files, cfg.batch_size, True, rng):
      b = prep(raw)
      with torch.autocast("cuda", torch.bfloat16, enabled=cfg.amp and device != "cpu"):
        pred = student(b["layers"], b["context"], b["commands"])["damage"]
      total, terms = distill_loss(
        pred.float(), b["label"], b["stock"], prior, cfg.loss, gen
      )
      opt.zero_grad(set_to_none=True)
      total.backward()
      torch.nn.utils.clip_grad_norm_(student.parameters(), cfg.grad_clip)
      opt.step()
      sched.step()
      step += 1
      if step % cfg.log_every == 0:
        log(
          {"train/loss": float(total), **{f"train/{k}": v for k, v in terms.items()}},
          step,
        )
        print(f"[distill] epoch {epoch} step {step} loss {float(total):.4f} {terms}")
      if cfg.max_steps is not None and step >= cfg.max_steps:
        break
    val = evaluate(student, val_files, prep, prior, cfg, device)
    log(val, step)
    print(f"[distill] epoch {epoch} ({time.perf_counter() - t0:.0f} s) {val}")
    if val["val/kl_w1.0"] < best:
      best = val["val/kl_w1.0"]
      save("best.pt", epoch, val)
    save("last.pt", epoch, val)
    if cfg.max_steps is not None and step >= cfg.max_steps:
      break
  writer.close()
  if run is not None:
    run.finish()


if __name__ == "__main__":
  main()
