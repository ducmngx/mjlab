"""The FailBench impact model: how a faulted robot moves under a plan.

Given what the robot knows at the start of a plan (the LiDAR map, its
state, HIM's latent, the last 2 s, and the known fault) and a 5 s plan
(``cmd_seq``, 50 steps of 0.1 s), it predicts per step:

- drift from the plan's commanded path, ``pose - cmd_path`` (x, y, yaw): the
  planner sweeps the robot's footprint along ``cmd_path + drift +- k sigma``
  over the map, so obstacle avoidance is geometric and works on layouts
  never trained on. Its spread is per axis of the step's own path frame
  (``drift_frame="path"``: along and across the commanded heading, where a
  turn no longer correlates the axes), and the likelihood is a Student-t
  with ``drift_nu`` degrees of freedom so rare large throws widen the tails
  instead of squeezing the middle (0: Gaussian, phase 1). ``drift`` comes
  back in the start frame; ``drift_std`` is the per-axis standard deviation
  in the path frame (``drift_yaw`` rotates it); a sweep only uses its xy
  norm, the same in either frame;
- the velocity deviation ``dv``, an auxiliary target;
- the fall hazard (``P(fallen by t)`` is monotone by construction), from
  which how soon a fall comes (P(fall within 1 s), the time to fall) is read;
- with ``latent_head``, HIM's latent (auxiliary: how the policy itself sees
  the robot responding, a dense signal tied to falls);
- with ``contact_heads`` (phase 1's model), also trunk or head contact
  (``collision``), a contact landing off the map, and body-contact maps, one
  64x64 map per second of the plan. The planner sweeps the drift over the
  map instead, so v2 drops them.

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
from typing import Any

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
  history_dim: int = 52
  """Per history step: state (33), command (3) and HIM's latent (16); 36
  without the latent (phase 1, and reads the first 36 of newer shards)."""
  fault_params: int = 2
  logvar_range: tuple[float, float] = (-10.0, 4.0)
  drift_frame: str = "path"
  """``path``: drift spread per axis of the commanded heading at each step;
  ``start``: of the start frame (phase 1)."""
  drift_nu: float = 4.0
  """Student-t degrees of freedom of the drift likelihood (> 2); 0 for a
  Gaussian (phase 1)."""
  norm_min_std: float = 1e-6
  """Input normalisation floor; HIM's latent moves by about 1e-3 per
  dimension, which phase 1's 1e-3 floor squashed."""
  contact_heads: bool = False
  """Also predict collision, off-map contact and contact maps (phase 1)."""
  latent_head: bool = True
  """Also predict HIM's latent per step (``latent_future``, auxiliary)."""

  @classmethod
  def from_saved(cls, saved: dict) -> ImpactModelCfg:
    """A checkpoint's config; ones saved before ``contact_heads`` had them."""
    kw: dict[str, Any] = {
      "contact_heads": True,
      "latent_head": False,
      "drift_frame": "start",
      "drift_nu": 0.0,
      "norm_min_std": 1e-3,
    }
    kw |= {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()}
    return cls(**kw)


def rotate_xy(v: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  """``v [..., 3]`` (x, y, yaw) with its xy rotated by ``yaw [...]``."""
  c, s = torch.cos(yaw), torch.sin(yaw)
  x, y = v[..., 0], v[..., 1]
  return torch.stack([c * x - s * y, s * x + c * y, v[..., 2]], dim=-1)


class Normalizer(nn.Module):
  """Fixed per-channel ``(x - mean) / std`` from ``stats.json``."""

  def __init__(
    self,
    stats: dict[str, dict[str, list[float]]],
    keys: list[str],
    min_std: float = 1e-3,
  ):
    super().__init__()
    for k in keys:
      self.register_buffer(f"{k}_mean", torch.tensor(stats[k]["mean"]))
      std = torch.tensor(stats[k]["std"]).clamp(min=min_std)
      self.register_buffer(f"{k}_std", std)

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
    if 0 < cfg.drift_nu <= 2:
      raise ValueError("drift_nu must be over 2 (a finite variance) or 0.")
    self.cfg = cfg
    d = cfg.d_model
    # A model reading fewer history numbers than the shards hold (36 of 52:
    # no latent) normalises just those.
    hist = stats["history"]
    stats = stats | {
      "history": {k: hist[k][: cfg.history_dim] for k in ("mean", "std")}
    }
    self.norm = Normalizer(
      stats,
      ["state", "him_latent", "history", "cmd_seq", "cmd_path", "map"],
      cfg.norm_min_std,
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
    # drift mean (3), drift log-variance (3), dv (3), hazard; then with
    # contact_heads, collision and offmap.
    self.step_head = nn.Linear(
      d, 10 + 2 * cfg.contact_heads + cfg.latent_dim * cfg.latent_head
    )
    self.contacts = ContactDecoder(d) if cfg.contact_heads else None

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
      self.history_in(n("history", batch["history"][..., : self.cfg.history_dim]))
      + self.history_pos,
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
    may have batch 1 to score many plans against one context. ``contact_maps``
    only applies with ``contact_heads``."""
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
    logvar = out[..., 3:6].clamp(lo, hi)
    yaw = cmd_path[..., 2] if self.cfg.drift_frame == "path" else 0 * cmd_path[..., 2]
    nu = self.cfg.drift_nu
    t_var = nu / (nu - 2) if nu > 0 else 1.0  # A Student-t's variance per scale^2.
    pred = {
      "drift": rotate_xy(out[..., 0:3], yaw),
      "drift_local": out[..., 0:3],
      "drift_logvar": logvar,
      "drift_std": torch.exp(0.5 * logvar) * t_var**0.5,
      "drift_yaw": yaw,
      "dv": out[..., 6:9],
      "hazard_logit": hazard,
      "p_fallen": 1 - log_up.exp(),
    }
    if self.cfg.latent_head:
      at = 10 + 2 * self.cfg.contact_heads
      # Predicted in normalised units (an output of order 1 at the start),
      # returned in the latent's own.
      z = out[..., at : at + self.cfg.latent_dim]
      pred["latent"] = z * self.norm.him_latent_std + self.norm.him_latent_mean
    if self.contacts is None:
      return pred
    pred["collision_logit"] = out[..., 10]
    pred["offmap_logit"] = out[..., 11]
    if contact_maps:
      k = self.cfg.steps_per_map
      tokens = h.reshape(b, self.cfg.num_steps // k, k, -1).mean(dim=2)
      feats = [f.expand(b, -1, -1, -1) for f in ctx.map_feats]
      pred["contact_logit"] = self.contacts(feats, tokens)
    return pred

  def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return self.decode(self.encode(batch), batch["cmd_seq"], batch["cmd_path"])
