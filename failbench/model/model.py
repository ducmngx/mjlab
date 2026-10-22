"""The FailBench impact model: how a faulted robot moves under a plan.

Given what the robot knows at the start of a plan (the LiDAR map, its
state, HIM's latent, the last 2 s, and the known fault) and a 5 s plan
(``cmd_seq``, 50 steps of 0.1 s), it predicts per step:

- drift from the plan's commanded path, ``pose - cmd_path`` (x, y, yaw), as
  a Gaussian (mean and log-variance): the planner sweeps the robot's
  footprint along ``cmd_path + drift +- k sigma`` over the map, so obstacle
  avoidance is geometric and works on layouts never trained on;
- the velocity deviation ``dv``, an auxiliary target;
- the fall hazard (``P(fallen by t)`` is monotone by construction);
- trunk or head contact (``collision``) and a contact landing off the map;
- body-contact maps, one 64x64 map per second of the plan.

The model splits into a context encoder (map, state, history, fault; run
once per planning cycle, once per fault for a safety filter) and a plan
decoder (one pass per candidate plan), so a planner scoring many plans
pays for the encoder once. The decoder is non-autoregressive: one query
token per step built from that step's command and commanded pose, as in
ZLIK's future-action queries (Pokhrel et al., arXiv:2602.12385). As in
ZLIK, decoder self-attention is causal: a step sees the plan up to itself,
never commands that come later.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

NUM_JOINTS = 12
JOINTS_PER_LEG = 3


@dataclass(frozen=True)
class ImpactModelCfg:
  d_model: int = 128
  heads: int = 4
  context_layers: int = 4
  decoder_layers: int = 3
  dropout: float = 0.1
  map_cells: int = 64
  num_steps: int = 50
  steps_per_map: int = 10
  """Plan steps per contact map (10 = one map per second)."""
  state_dim: int = 33
  latent_dim: int = 16
  history_steps: int = 20
  history_dim: int = 36
  fault_params: int = 2
  logvar_range: tuple[float, float] = (-10.0, 4.0)


class Normalizer(nn.Module):
  """Fixed per-channel ``(x - mean) / std`` from ``stats.json``."""

  def __init__(self, stats: dict[str, dict[str, list[float]]], keys: list[str]):
    super().__init__()
    for k in keys:
      self.register_buffer(f"{k}_mean", torch.tensor(stats[k]["mean"]))
      self.register_buffer(f"{k}_std", torch.tensor(stats[k]["std"]).clamp(min=1e-3))

  def forward(self, key: str, x: torch.Tensor) -> torch.Tensor:
    return (x - getattr(self, f"{key}_mean")) / getattr(self, f"{key}_std")


def _conv(c_in: int, c_out: int, stride: int) -> nn.Sequential:
  return nn.Sequential(
    nn.Conv2d(c_in, c_out, 3, stride=stride, padding=1),
    nn.GroupNorm(8, c_out),
    nn.GELU(),
    nn.Conv2d(c_out, c_out, 3, padding=1),
    nn.GroupNorm(8, c_out),
    nn.GELU(),
  )


class MapEncoder(nn.Module):
  """2x64x64 map -> 8x8 tokens, plus skip features for the contact decoder."""

  def __init__(self, d: int):
    super().__init__()
    self.stages = nn.ModuleList(
      [_conv(2, 32, 1), _conv(32, 64, 2), _conv(64, 128, 2), _conv(128, d, 2)]
    )

  def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
    feats = []
    for stage in self.stages:
      x = stage(x)
      feats.append(x)
    return feats  # 64, 32, 16, 8 cells


class ContactDecoder(nn.Module):
  """Per-map tokens FiLM the 8x8 map features, then upsample with skips."""

  def __init__(self, d: int):
    super().__init__()
    self.film = nn.Linear(d, 2 * d)
    self.up = nn.ModuleList(
      [_conv(d + 128, 128, 1), _conv(128 + 64, 64, 1), _conv(64 + 32, 32, 1)]
    )
    self.out = nn.Conv2d(32, 1, 1)

  def forward(self, feats: list[torch.Tensor], tokens: torch.Tensor) -> torch.Tensor:
    """``tokens [B, M, d]`` -> logits ``[B, M, H, W]``."""
    b, m, d = tokens.shape
    rep = lambda f: f.repeat_interleave(m, dim=0)  # noqa: E731
    scale, shift = self.film(tokens).reshape(b * m, 2 * d, 1, 1).chunk(2, dim=1)
    x = rep(feats[3]) * (1 + scale) + shift
    for conv, skip in zip(self.up, feats[2::-1], strict=True):
      x = nn.functional.interpolate(x, scale_factor=2, mode="nearest")
      x = conv(torch.cat([x, rep(skip)], dim=1))
    return self.out(x).reshape(b, m, *x.shape[-2:])


@dataclass
class Context:
  """Encoder output, reusable across candidate plans."""

  memory: torch.Tensor
  """``[B, T, d]`` context tokens."""
  map_feats: list[torch.Tensor]


class ImpactModel(nn.Module):
  def __init__(self, cfg: ImpactModelCfg, stats: dict):
    super().__init__()
    self.cfg = cfg
    d = cfg.d_model
    self.norm = Normalizer(
      stats, ["state", "him_latent", "history", "cmd_seq", "cmd_path", "map"]
    )
    self.map_encoder = MapEncoder(d)
    grid = cfg.map_cells // 8
    self.map_pos = nn.Parameter(torch.randn(grid * grid, d) * 0.02)
    self.state_in = nn.Linear(cfg.state_dim, d)
    self.latent_in = nn.Linear(cfg.latent_dim, d)
    self.history_in = nn.Linear(cfg.history_dim, d)
    self.history_pos = nn.Parameter(torch.randn(cfg.history_steps, d) * 0.02)
    self.fault_in = nn.Linear(cfg.fault_params, d)
    # Joint identity factorised into leg and joint type, so a held-out
    # (joint, mode) pair shares structure with seen ones.
    self.leg_emb = nn.Embedding(NUM_JOINTS // JOINTS_PER_LEG, d)
    self.joint_type_emb = nn.Embedding(JOINTS_PER_LEG, d)
    self.type_emb = nn.Embedding(5, d)  # map, state, latent, history, fault
    layer = lambda: nn.TransformerEncoderLayer(  # noqa: E731
      d,
      cfg.heads,
      4 * d,
      cfg.dropout,
      activation="gelu",
      batch_first=True,
      norm_first=True,
    )
    self.context = nn.TransformerEncoder(layer(), cfg.context_layers)

    self.plan_in = nn.Linear(6, d)
    self.step_pos = nn.Parameter(torch.randn(cfg.num_steps, d) * 0.02)
    self.decoder = nn.TransformerDecoder(
      nn.TransformerDecoderLayer(
        d,
        cfg.heads,
        4 * d,
        cfg.dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
      ),
      cfg.decoder_layers,
    )
    self.register_buffer(
      "causal_mask",
      nn.Transformer.generate_square_subsequent_mask(cfg.num_steps),
      persistent=False,
    )
    self.head_norm = nn.LayerNorm(d)
    # drift mean (3), drift log-variance (3), dv (3), hazard, collision, offmap
    self.step_head = nn.Linear(d, 12)
    self.contacts = ContactDecoder(d)

  def encode(self, batch: dict[str, torch.Tensor]) -> Context:
    n = self.norm
    m = batch["map"]
    known = m[:, 1:2]
    height = torch.where(known > 0, n("map", m[:, :1]), 0.0)
    feats = self.map_encoder(torch.cat([height, known], dim=1))
    map_tok = feats[3].flatten(2).transpose(1, 2) + self.map_pos
    b = m.shape[0]
    fault = batch["fault"].reshape(b, NUM_JOINTS, self.cfg.fault_params)
    j = torch.arange(NUM_JOINTS, device=m.device)
    fault_tok = (
      self.fault_in(fault)
      + self.leg_emb(j // JOINTS_PER_LEG)
      + self.joint_type_emb(j % JOINTS_PER_LEG)
    )
    groups = [
      map_tok,
      self.state_in(n("state", batch["state"]))[:, None],
      self.latent_in(n("him_latent", batch["him_latent"]))[:, None],
      self.history_in(n("history", batch["history"])) + self.history_pos,
      fault_tok,
    ]
    tokens = torch.cat(
      [g + self.type_emb.weight[i] for i, g in enumerate(groups)], dim=1
    )
    return Context(self.context(tokens), feats)

  def decode(
    self,
    ctx: Context,
    cmd_seq: torch.Tensor,
    cmd_path: torch.Tensor,
    contact_maps: bool = True,
  ) -> dict[str, torch.Tensor]:
    """Predictions for plans ``cmd_seq`` / ``cmd_path [B, T, 3]``. ``ctx``
    may have batch 1 to score many plans against one context."""
    b = cmd_seq.shape[0]
    memory = ctx.memory.expand(b, -1, -1)
    plan = torch.cat(
      [self.norm("cmd_seq", cmd_seq), self.norm("cmd_path", cmd_path)], dim=-1
    )
    h = self.decoder(
      self.plan_in(plan) + self.step_pos,
      memory,
      tgt_mask=self.causal_mask,
      tgt_is_causal=True,
    )
    h = self.head_norm(h)
    out = self.step_head(h)
    lo, hi = self.cfg.logvar_range
    hazard = out[..., 9]
    # P(upright at t) = prod (1 - sigmoid(hazard)) up to t.
    log_up = torch.cumsum(nn.functional.logsigmoid(-hazard), dim=1)
    pred = {
      "drift": out[..., 0:3],
      "drift_logvar": out[..., 3:6].clamp(lo, hi),
      "dv": out[..., 6:9],
      "hazard_logit": hazard,
      "p_fallen": 1 - log_up.exp(),
      "collision_logit": out[..., 10],
      "offmap_logit": out[..., 11],
    }
    if contact_maps:
      k = self.cfg.steps_per_map
      tokens = h.reshape(b, self.cfg.num_steps // k, k, -1).mean(dim=2)
      feats = [f.expand(b, -1, -1, -1) for f in ctx.map_feats]
      pred["contact_logit"] = self.contacts(feats, tokens)
    return pred

  def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return self.decode(self.encode(batch), batch["cmd_seq"], batch["cmd_path"])
