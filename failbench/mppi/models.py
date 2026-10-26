"""Dynamics models MPPI scores plans with, behind one interface.

``predict(inputs, cmd_seq, cmd_path)`` takes the model inputs of ``n``
robots (``state``, ``him_latent``, ``history``, ``fault``, ``map``; see
``failbench/data/dataset.py``) and ``k`` plans per robot (``[n, k, T, 3]``)
and returns, per plan and step:

- ``drift``, ``drift_std`` ``[n, k, T, 3]``: predicted pose minus
  ``cmd_path`` (x, y, yaw), and its per-axis spread;
- ``p_fallen`` ``[n, k, T]``: P(fallen by the end of the step);
- models with contact heads only (phase 1, the stub): ``p_collision``
  ``[n, k, T]``, P(trunk or head contact in the step), and with
  ``contact_maps=True``, ``contact`` ``[n, k, S, H, W]``: P(a non-foot
  body part touches each map cell during each second of the plan) on the
  model's map grid. Costly over many plans; MPPI asks for it on the chosen
  plan only, to show it.

``ZeroDriftModel`` assumes the robot does exactly what it is told (a
kinematic planner); it is the stub the loop is tested with before a trained
model exists, and later the fault-unaware baseline. ``ImpactDynamics`` wraps
a trained ``ImpactModel`` checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch

from failbench.model.model import Context, ImpactModel, ImpactModelCfg

MODEL_INPUTS = ("state", "him_latent", "history", "fault", "map")


class DynamicsModel(Protocol):
  map_mode: str
  """What map the model was trained on: ``clean`` (the full map) or a LiDAR
  visibility (``fused``, ``single``)."""

  def predict(
    self,
    inputs: dict[str, torch.Tensor],
    cmd_seq: torch.Tensor,
    cmd_path: torch.Tensor,
    contact_maps: bool = False,
  ) -> dict[str, torch.Tensor]: ...


class ZeroDriftModel:
  """No drift, no falls, no contact; spread growing linearly with time."""

  map_mode = "fused"

  def __init__(
    self,
    step_s: float = 0.1,
    std0: tuple[float, float, float] = (0.03, 0.03, 0.02),
    std_rate: tuple[float, float, float] = (0.03, 0.03, 0.02),
  ):
    self.step_s = step_s
    self.std0 = std0
    self.std_rate = std_rate

  def predict(self, inputs, cmd_seq, cmd_path, contact_maps=False):
    n, k, t, _ = cmd_seq.shape
    dev = cmd_seq.device
    time = (torch.arange(t, device=dev) + 1) * self.step_s
    std = torch.tensor(self.std0, device=dev) + time[:, None] * torch.tensor(
      self.std_rate, device=dev
    )
    zeros = torch.zeros(n, k, t, device=dev)
    out = {
      "drift": torch.zeros_like(cmd_path),
      "drift_std": std.expand(n, k, t, 3),
      "p_fallen": zeros,
      "p_collision": zeros,
    }
    if contact_maps:
      h, w = inputs["map"].shape[-2:] if "map" in inputs else (64, 64)
      secs = round(t * self.step_s)
      out["contact"] = torch.zeros(n, k, secs, h, w, device=dev)
    return out


class ImpactDynamics:
  """A trained ``ImpactModel``: one encode per robot, one batched decode
  over all its plans."""

  def __init__(self, model: ImpactModel, map_mode: str, amp: bool = True):
    self.model = model.eval()
    self.map_mode = map_mode
    self.amp = amp

  @classmethod
  def from_checkpoint(cls, path: str | Path, device, amp: bool = True):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    cfg = ImpactModelCfg.from_saved(ckpt["model_cfg"])
    model = ImpactModel(cfg, ckpt["stats"]).to(device)
    model.load_state_dict(ckpt["model"])
    return cls(model, ckpt["train_cfg"]["map_mode"], amp)

  @torch.no_grad()
  def predict(self, inputs, cmd_seq, cmd_path, contact_maps=False):
    n, k, t, _ = cmd_seq.shape
    dev = cmd_seq.device
    with torch.autocast(
      "cuda", torch.bfloat16, enabled=self.amp and dev.type == "cuda"
    ):
      ctx = self.model.encode({key: inputs[key] for key in MODEL_INPUTS})
      feats = (
        [f.repeat_interleave(k, dim=0) for f in ctx.map_feats] if contact_maps else []
      )
      plans = Context(ctx.memory.repeat_interleave(k, dim=0), feats)
      pred = self.model.decode(
        plans, cmd_seq.flatten(0, 1), cmd_path.flatten(0, 1), contact_maps
      )
    pred = {key: v.float().unflatten(0, (n, k)) for key, v in pred.items()}
    out = {
      "drift": pred["drift"],
      "drift_std": torch.exp(0.5 * pred["drift_logvar"]),
      "p_fallen": pred["p_fallen"],
    }
    if "collision_logit" in pred:
      out["p_collision"] = torch.sigmoid(pred["collision_logit"])
    if "contact_logit" in pred:
      out["contact"] = torch.sigmoid(pred["contact_logit"])
    return out
