"""The FailBench impact model: how a robot moves under a plan if a fault hits.

The question it answers: if fault *f* hit right now and the robot, not
noticing, kept running *this* candidate's commands, where would its body go
and how fast? An encoder takes the context once per planning cycle: the
terrain map, the robot's state, HIM's latent and the last 2 s. A decoder
then takes each candidate (``cmd_seq``, a step of 0.1 s each, and its
commanded poses ``cmd_path``) together with the imagined fault, and
predicts per step:

- drift from the plan's commanded path, ``pose - cmd_path`` (x, y, yaw): the
  planner sweeps the robot's footprint along ``cmd_path + drift +- k sigma``
  over its map, so obstacles are geometric and work on layouts never trained
  on. Its spread is per axis of the step's own path frame (``drift_frame=
  "path"``: along and across the commanded heading), and the likelihood is a
  Student-t with ``drift_nu`` degrees of freedom so rare large throws widen
  the tails instead of squeezing the middle. ``drift`` comes back in the
  start frame; ``drift_std`` is the per-axis standard deviation in the path
  frame (``drift_yaw`` rotates it);
- the trunk's velocity ``vel`` (x, y, z in the start yaw frame), with its
  spread in the same path frame: how fast the body moves where it is, so a
  contact the sweep finds can be priced by its energy, a fall onto
  something included (its vertical speed).

Models before 2026-11-02 took the fault as encoder tokens and also
predicted a fall hazard (``fall_head``), the velocity deviation ``dv`` and,
some, HIM's latent or body contacts (``contact_heads``: trunk or head
contact, a contact off the map, one 64x64 contact map per second); their
configs and checkpoints still load.

The decoder is non-autoregressive: one query token per step built from that
step's command and commanded pose (plus the fault), as in ZLIK's
future-action queries (Pokhrel et al., arXiv:2602.12385). As in ZLIK,
decoder self-attention is causal: a step sees the plan up to itself, never
commands that come later. The fault's per-joint tokens join the context the
decoder attends to, and their mean conditions every query.
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
  latent_head: bool = False
  """Also predict HIM's latent per step (``latent_future``, auxiliary)."""
  fault_in_decoder: bool = True
  """The fault enters the decoder with the candidate, so the encoder sees
  only the context (map, state, latent, history) and runs once per planning
  cycle whatever faults are imagined; False: an encoder token (models
  before 2026-11-02)."""
  fall_head: bool = False
  """Also predict the fall hazard (``p_fallen``; models before 2026-11-02)."""
  dv_head: bool = False
  """Also predict the velocity deviation ``dv`` (auxiliary; models before
  2026-11-02)."""
  velocity_head: bool = True
  """Predict the trunk's velocity per step (``vel``, x, y, z in the start
  yaw frame) with its spread, as the drift's: with the drift, how fast the
  body moves where it is, for an impact's energy."""

  @classmethod
  def from_saved(cls, saved: dict) -> ImpactModelCfg:
    """A checkpoint's config; ones saved before ``contact_heads`` had them."""
    kw: dict[str, Any] = {
      "contact_heads": True,
      "latent_head": False,
      "fault_in_decoder": False,
      "fall_head": True,
      "dv_head": True,
      "velocity_head": False,
      "drift_frame": "start",
      "drift_nu": 0.0,
      "norm_min_std": 1e-3,
    }
    kw |= {k: tuple(v) if isinstance(v, list) else v for k, v in saved.items()}
    return cls(**kw)


def head_layout(cfg: ImpactModelCfg) -> dict[str, slice]:
  """Where each output sits in the step head, in the order older models
  have them (so their checkpoints load)."""
  sizes = [("drift", 3), ("drift_logvar", 3)]
  sizes += [("dv", 3)] * cfg.dv_head + [("hazard", 1)] * cfg.fall_head
  sizes += [("collision", 1), ("offmap", 1)] * cfg.contact_heads
  sizes += [("latent", cfg.latent_dim)] * cfg.latent_head
  sizes += [("vel", 3), ("vel_logvar", 3)] * cfg.velocity_head
  out, at = {}, 0
  for name, size in sizes:
    out[name] = slice(at, at + size)
    at += size
  return out


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
    if cfg.fault_in_decoder:
      self.fault_pool = nn.Linear(d, d)
    self.head_norm = nn.LayerNorm(d)
    self.layout = head_layout(cfg)
    self.step_head = nn.Linear(d, max(s.stop for s in self.layout.values()))
    self.contacts = ContactDecoder(d) if cfg.contact_heads else None

  def encode(self, batch: dict[str, torch.Tensor]) -> Context:
    n = self.norm
    m = batch["map"]
    known = m[:, 1:2]
    height = torch.where(known > 0, n("map", m[:, :1]), 0.0)
    feats = self.map_encoder(torch.cat([height, known], dim=1))
    map_tok = feats[3].flatten(2).transpose(1, 2) + self.map_pos
    groups = [
      map_tok,
      self.state_in(n("state", batch["state"]))[:, None],
      self.latent_in(n("him_latent", batch["him_latent"]))[:, None],
      self.history_in(n("history", batch["history"][..., : self.cfg.history_dim]))
      + self.history_pos,
    ]
    if not self.cfg.fault_in_decoder:
      groups.append(self._fault_tokens(batch["fault"]))
    tokens = torch.cat(
      [g + self.type_emb.weight[i] for i, g in enumerate(groups)], dim=1
    )
    return Context(self.context(tokens), feats)

  def _fault_tokens(self, fault: torch.Tensor) -> torch.Tensor:
    """A token per joint ``[B, 12, d]`` from the encoded fault ``[B, 24]``."""
    fault = fault.reshape(len(fault), NUM_JOINTS, self.cfg.fault_params)
    j = torch.arange(NUM_JOINTS, device=fault.device)
    return (
      self.fault_in(fault)
      + self.leg_emb(j // JOINTS_PER_LEG)
      + self.joint_type_emb(j % JOINTS_PER_LEG)
    )

  def decode(
    self,
    ctx: Context,
    cmd_seq: torch.Tensor,
    cmd_path: torch.Tensor,
    contact_maps: bool = True,
    fault: torch.Tensor | None = None,
  ) -> dict[str, torch.Tensor]:
    """Predictions for plans ``cmd_seq`` / ``cmd_path [B, T, 3]`` under the
    encoded ``fault [B, 24]`` (``fault_in_decoder`` models; the others took
    it in ``encode``). ``ctx`` and ``fault`` may have batch 1 to score many
    plans against one context. ``contact_maps`` only applies with
    ``contact_heads``."""
    cfg = self.cfg
    b = cmd_seq.shape[0]
    memory = ctx.memory.expand(b, -1, -1)
    plan = torch.cat(
      [self.norm("cmd_seq", cmd_seq), self.norm("cmd_path", cmd_path)], dim=-1
    )
    query = self.plan_in(plan) + self.step_pos
    if cfg.fault_in_decoder:
      if fault is None:
        raise ValueError("This model takes the fault in decode().")
      tok = (self._fault_tokens(fault) + self.type_emb.weight[4]).expand(b, -1, -1)
      memory = torch.cat([memory, tok], dim=1)
      query = query + self.fault_pool(tok.mean(dim=1))[:, None]
    h = self.decoder(query, memory, tgt_mask=self.causal_mask, tgt_is_causal=True)
    h = self.head_norm(h)
    out = self.step_head(h)
    at = self.layout
    lo, hi = cfg.logvar_range
    yaw = cmd_path[..., 2] if cfg.drift_frame == "path" else 0 * cmd_path[..., 2]
    nu = cfg.drift_nu
    t_var = nu / (nu - 2) if nu > 0 else 1.0  # A Student-t's variance per scale^2.
    logvar = out[..., at["drift_logvar"]].clamp(lo, hi)
    pred = {
      "drift": rotate_xy(out[..., at["drift"]], yaw),
      "drift_local": out[..., at["drift"]],
      "drift_logvar": logvar,
      "drift_std": torch.exp(0.5 * logvar) * t_var**0.5,
      "drift_yaw": yaw,
    }
    if cfg.velocity_head:
      # In the path frame as the drift (x along, y across the commanded
      # heading, z up), returned in the start frame.
      vel_logvar = out[..., at["vel_logvar"]].clamp(lo, hi)
      pred |= {
        "vel": rotate_xy(out[..., at["vel"]], yaw),
        "vel_local": out[..., at["vel"]],
        "vel_logvar": vel_logvar,
        "vel_std": torch.exp(0.5 * vel_logvar) * t_var**0.5,
      }
    if cfg.dv_head:
      pred["dv"] = out[..., at["dv"]]
    if cfg.fall_head:
      hazard = out[..., at["hazard"]][..., 0]
      # P(upright at t) = prod (1 - sigmoid(hazard)) up to t.
      log_up = torch.cumsum(nn.functional.logsigmoid(-hazard), dim=1)
      pred |= {"hazard_logit": hazard, "p_fallen": 1 - log_up.exp()}
    if cfg.latent_head:
      # Predicted in normalised units (an output of order 1 at the start),
      # returned in the latent's own.
      z = out[..., at["latent"]]
      pred["latent"] = z * self.norm.him_latent_std + self.norm.him_latent_mean
    if self.contacts is None:
      return pred
    pred["collision_logit"] = out[..., at["collision"]][..., 0]
    pred["offmap_logit"] = out[..., at["offmap"]][..., 0]
    if contact_maps:
      k = self.cfg.steps_per_map
      tokens = h.reshape(b, self.cfg.num_steps // k, k, -1).mean(dim=2)
      feats = [f.expand(b, -1, -1, -1) for f in ctx.map_feats]
      pred["contact_logit"] = self.contacts(feats, tokens)
    return pred

  def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return self.decode(
      self.encode(batch), batch["cmd_seq"], batch["cmd_path"], fault=batch["fault"]
    )
