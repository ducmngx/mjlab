"""The fault-risk critic: FailBench's method, as one more critic of a stock
Nav2 planner (``Nav2Mppi``, ``Dwb``), its weight the one knob.

Each cycle it asks the impact model, for every candidate and every imagined
fault: if this fault hit now and the robot, not noticing, kept sending this
candidate's commands, where would the trunk go? The candidate pays, per
fault, the worst over its steps of P(the footprint touches an object) times
the damage a hit at the trunk's kinetic energy there does to that object
(``failbench.severity.damage``: the severity, reached by an energy on the
object's scale), summed over the map's severity levels; the faults are weighted by how often the
evaluation draws them (``imagined_faults``), so the cost is the expected
damage given a fault. Every candidate is priced from its own commands and
its own predicted path: nothing is shared between candidates or slid along
them.

- The commands are the ones the robot would be sent (the planner's
  ``data["commands"]``: an MPPI sample within the limits, from the command
  it sends next; a DWB twist held), deadbanded as the command term does,
  and their commanded path is ``failbench/data/build.py``'s ``_integrate``
  (``integrate_plan``), as in the model's data.
- The model was trained on maps of terrain alone (the data's obstacles live
  only in the planner's costmap), so the runner blanks the costmap's
  obstacles out of the model's map (``blank_obstacles``); the throw is a
  free-space throw, and the obstacles are priced here.
- The trunk speed is the model's velocity head where it has one, else the
  predicted pose's rate of change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import torch

from failbench.faults import (
  JOINT_NAMES,
  LOCK,
  NOMINAL,
  TORQUE,
  FaultMixture,
  encode_fault,
)
from failbench.mppi.costmap import FootprintCfg, hit_prob
from failbench.mppi.inputs import deadband, integrate_plan
from failbench.mppi.models import DynamicsModel
from failbench.severity import (
  DEFAULT_ENERGY_SCALE,
  LEVEL_ENERGY_SCALE,
  MASS_KG,
  damage,
)


@dataclass(frozen=True)
class RiskCriticCfg:
  weight: float = 0.1
  """The knob: cost per unit of expected damage given a fault (in
  severity units; a sure hit on a person at speed is 50). Nav2's critics
  differ between samples by a few units."""
  measure: Literal["mean", "cvar", "max"] = "mean"
  """How the imagined faults' damages combine: their prior-weighted mean
  (the expected damage given a fault), the CVaR, the mean over the worst
  ``cvar_alpha`` of the prior, or the max, the worst fault's damage
  whatever its prior (the planner's fault-risk ``measure``)."""
  cvar_alpha: float = 0.025
  mixture: FaultMixture = field(default_factory=FaultMixture)
  """The evaluation's fault distribution, which the imagined faults stand
  for."""
  joints: tuple[int, ...] | None = None
  """The joints imagined faulting (None: all); each stands for its share."""
  faults: tuple[str, ...] | None = None
  """Imagine only these ``<joint>:<lock|limp|weak>`` faults (e.g.
  ``REPRESENTATIVE_FAULTS``), each at its mixture odds, renormalised over
  the set; None: every kind on ``joints``."""
  weak_torque: float = 0.25
  """The torque scale standing for a weak (not limp) joint: the mean of
  ``mixture.weak_torque``."""
  mass_kg: float = MASS_KG
  """The Go2's mass, for the kinetic energy."""
  energy_scale: dict[float, float] = field(
    default_factory=lambda: dict(LEVEL_ENERGY_SCALE)
  )
  """Energy scale (J) per severity level (``failbench.severity``); levels
  not listed take ``DEFAULT_ENERGY_SCALE``."""
  footprint: FootprintCfg = field(default_factory=FootprintCfg)
  zero_below: float = 0.2
  """The command term's deadband (``GoalVelocityCommand``)."""
  control_dt: float = 0.02
  substeps: int = 5
  """Control steps per plan step (50 Hz policy, 10 Hz plans)."""
  chunk: int = 8192
  """Candidates per model call."""
  top_m: int | None = None
  """Price only the ``top_m`` candidates the stock critics like best
  (``data["stock_cost"]``); the rest pay the worst of theirs. The model
  costs about 11 us per candidate and fault on a desktop GPU, so pricing
  all of MPPI's 1000 samples against 36 faults takes 0.4 s a cycle."""


REPRESENTATIVE_FAULTS: tuple[str, ...] = (
  "FL_hip_joint:limp",
  "FR_hip_joint:limp",
  "FL_calf_joint:lock",
  "FR_calf_joint:lock",
  "RL_thigh_joint:limp",
  "RR_thigh_joint:limp",
  "RL_thigh_joint:lock",
  "RR_thigh_joint:lock",
)
"""The eight faults the critic imagines (chosen 26 Oct from the fault data:
two per leg, the worst for falls): front hips limp and front calves
seized, rear thighs limp and seized."""


def imagined_faults(
  cfg: RiskCriticCfg, num_joints: int
) -> tuple[torch.Tensor, torch.Tensor]:
  """The faults imagined each cycle, ``[F, J, 2]``, and their probability
  given a fault ``[F]``: per joint, seized, limp and weak, with the
  mixture's odds."""
  mix = cfg.mixture
  _, lock_p, weak_p = mix.probs
  joints = range(num_joints) if cfg.joints is None else cfg.joints
  kinds = (
    (LOCK, 1.0, lock_p),
    (TORQUE, 0.0, weak_p * mix.limp_prob),
    (TORQUE, cfg.weak_torque, weak_p * (1 - mix.limp_prob)),
  )
  params, prior = [], []
  for j in joints:
    for index, value, p in kinds:
      if p <= 0:
        continue
      x = torch.tensor(NOMINAL).repeat(num_joints, 1)
      x[j, index] = value
      params.append(x)
      prior.append(p)
  stacked, prior = torch.stack(params), torch.tensor(prior)
  if cfg.faults is not None:
    names = fault_names(stacked)
    unknown = set(cfg.faults) - set(names)
    if unknown:
      raise ValueError(f"Unknown imagined faults {sorted(unknown)}.")
    keep = torch.tensor([n in cfg.faults for n in names])
    stacked, prior = stacked[keep], prior[keep]
  return stacked, prior / prior.sum()


def fault_names(params: torch.Tensor) -> list[str]:
  """``<joint>:<lock|limp|weak>`` per imagined fault ``[F, J, 2]``."""
  names = []
  for p in params:
    j = int((p != torch.tensor(NOMINAL)).any(-1).nonzero()[0])
    kind = "lock" if p[j, LOCK] > 0.5 else ("limp" if p[j, TORQUE] == 0 else "weak")
    names.append(f"{JOINT_NAMES[j]}:{kind}")
  return names


def blank_obstacles(model_map: torch.Tensor, obstacles: torch.Tensor) -> torch.Tensor:
  """The model's map ``[n, 2, H, W]`` (height, known) with ``obstacles [n,
  H, W]`` flattened to the floor: the median height of the known cells
  that aren't obstacles."""
  height, known = model_map[:, 0], model_map[:, 1] > 0.5
  ground = torch.where(known & ~obstacles, height, float("nan"))
  floor = ground.flatten(1).nanmedian(-1).values.nan_to_num(0.0)
  height = torch.where(obstacles, floor[:, None, None], height)
  return torch.stack([height, model_map[:, 1]], 1)


class RiskCritic:
  """``(data) -> cost [n, K]`` for ``Nav2Mppi``'s and ``Dwb``'s extra
  critics. ``data`` carries the planner's ``commands [n, K, T, 3]`` and
  ``resolution`` and, from the runner, the model ``inputs`` (obstacles
  blanked), the severity ``risk_fields`` (``cbf.severity_fields``) and
  optionally ``exposed [n]`` (robots not yet faulted)."""

  __name__ = "fault_risk"

  def __init__(self, cfg: RiskCriticCfg, model: DynamicsModel):
    self.cfg = cfg
    self.model = model
    self.last: dict[str, torch.Tensor] = {}
    """The last call's per-fault cost ``[n, K, F]``."""

  @torch.no_grad()
  def __call__(self, data: dict) -> torch.Tensor:
    c = self.cfg
    cmd = deadband(data["commands"], c.zero_below)
    n, k = cmd.shape[:2]
    if not data.get("risk_fields"):
      return cmd.new_zeros(n, k)
    if c.top_m is None or c.top_m >= k or "stock_cost" not in data:
      risk = self._price(cmd, data)
    else:
      best = data["stock_cost"].topk(c.top_m, -1, largest=False).indices
      picked = cmd.gather(1, best[..., None, None].expand(-1, -1, *cmd.shape[2:]))
      priced = self._price(picked, data)
      risk = priced.amax(-1, keepdim=True).expand(n, k).scatter(1, best, priced)
    exposed = data.get("exposed")
    return risk if exposed is None else risk * exposed[:, None]

  @torch.no_grad()
  def throws(self, cmd: torch.Tensor, data: dict) -> dict:
    """For one plan per robot, ``cmd [n, T, 3]`` (as it would be sent): each
    imagined fault's predicted trunk pose ``pose [n, F, T, 3]`` and spread
    ``std``, its weighted share of the critic's cost ``cost [n, F]`` (the
    shares add up to the plan's cost) and the faults' ``names``."""
    c = self.cfg
    cmd = deadband(cmd, c.zero_below)[:, None]
    n = cmd.shape[0]
    params, prior = imagined_faults(c, data["inputs"]["fault"].shape[-1] // 2)
    rep, path = self._rows(cmd, data, params)
    nf = len(params)
    seq, cp = cmd.repeat_interleave(nf, 0), path.repeat_interleave(nf, 0)
    pred = self.model.predict(rep, seq, cp)
    cost, touch = self._damage_touch(pred, cp, data["risk_fields"], data["resolution"])
    cost, touch = cost[:, 0].unflatten(0, (n, nf)), touch[:, 0].unflatten(0, (n, nf))
    return {
      "pose": (cp + pred["drift"])[:, 0].unflatten(0, (n, nf)),
      "std": pred["drift_std"][:, 0].unflatten(0, (n, nf)),
      "cost": c.weight * self._shares(cost, prior),
      "damage": cost,
      "touch": touch,
      "names": fault_names(params),
    }

  def _rows(
    self, cmd: torch.Tensor, data: dict, params: torch.Tensor
  ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """The model inputs with every robot once per imagined fault (rows
    ``[n * F]``) and the commanded paths ``[n, K, T, 3]`` of ``cmd``."""
    c = self.cfg
    inputs = data["inputs"]
    rep = {key: v.repeat_interleave(len(params), 0) for key, v in inputs.items()}
    faults = encode_fault(params).to(cmd)
    rep["fault"] = faults.repeat(cmd.shape[0], 1).to(inputs["fault"].dtype)
    return rep, integrate_plan(cmd, c.control_dt, c.substeps)

  def _price(self, cmd: torch.Tensor, data: dict) -> torch.Tensor:
    """The weighted expected damage ``[n, K]`` of commands ``[n, K, T, 3]``."""
    c = self.cfg
    n, k = cmd.shape[:2]
    fields = data["risk_fields"]
    params, prior = imagined_faults(c, data["inputs"]["fault"].shape[-1] // 2)
    prior = prior.to(cmd)
    nf = len(params)
    rep, path = self._rows(cmd, data, params)
    per = max(1, c.chunk // (n * nf))
    parts = []
    for k0 in range(0, k, per):
      ks = slice(k0, k0 + per)
      seq = cmd[:, ks].repeat_interleave(nf, 0)
      cp = path[:, ks].repeat_interleave(nf, 0)
      pred = self.model.predict(rep, seq, cp)
      parts.append(self._damage(pred, cp, fields, data["resolution"]))
    cost = torch.cat(parts, 1).unflatten(0, (n, nf)).permute(0, 2, 1)
    self.last = {"per_fault": cost}
    return c.weight * self._shares(cost, prior).sum(-1)

  def _shares(self, cost: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
    """Each fault's share ``[..., F]`` of the combined damage of per-fault
    damages ``cost [..., F]`` under ``prior [F]`` (``measure``)."""
    prior = prior.to(cost)
    if self.cfg.measure == "mean":
      return cost * prior
    if self.cfg.measure == "max":
      # The worst fault alone, however unlikely.
      worst = cost.argmax(-1, keepdim=True)
      return cost * torch.zeros_like(cost).scatter(-1, worst, 1.0)
    # CVaR: the worst faults first, each weighted by the part of its prior
    # inside the worst alpha.
    alpha = self.cfg.cvar_alpha
    order = cost.argsort(-1, descending=True)
    p = prior.expand_as(cost).gather(-1, order)
    before = p.cumsum(-1) - p
    inside = (alpha - before).clamp(min=0).minimum(p) / alpha
    weight = torch.zeros_like(cost).scatter(-1, order, inside)
    return cost * weight

  def _damage(
    self,
    pred: dict[str, torch.Tensor],
    path: torch.Tensor,
    fields: list[tuple[float, torch.Tensor]],
    resolution: float,
  ) -> torch.Tensor:
    """Expected damage ``[n * F, K]`` of predicted throws off ``path``."""
    return self._damage_touch(pred, path, fields, resolution)[0]

  def _damage_touch(
    self,
    pred: dict[str, torch.Tensor],
    path: torch.Tensor,
    fields: list[tuple[float, torch.Tensor]],
    resolution: float,
  ) -> tuple[torch.Tensor, torch.Tensor]:
    """``_damage`` and the worst P(touching any object) over the steps."""
    c = self.cfg
    pose = path + pred["drift"]
    if not fields:  # Nothing in view.
      zero = pose.new_zeros(pose.shape[:2])
      return zero, zero
    if "vel" in pred:
      speed2 = pred["vel"].square().sum(-1)
    else:
      start = torch.zeros_like(pose[..., :1, :2])
      step = torch.diff(pose[..., :2], dim=-2, prepend=start)
      speed2 = (step / (c.control_dt * c.substeps)).square().sum(-1)
    energy = 0.5 * c.mass_kg * speed2
    nf = pose.shape[0] // fields[0][1].shape[0]
    cost = pose.new_zeros(pose.shape[:2])
    worst = pose.new_zeros(pose.shape[:2])
    for severity, dist in fields:
      touch = hit_prob(
        dist.repeat_interleave(nf, 0), pose, pred["drift_std"], resolution, c.footprint
      )
      scale = c.energy_scale.get(severity, DEFAULT_ENERGY_SCALE)
      cost = cost + (touch * damage(severity, energy, scale)).amax(-1)
      worst = worst.maximum(touch.amax(-1))
    return cost, worst
