"""The fault-risk critic: FailBench's method, as one more critic of a stock
Nav2 planner (``Nav2Mppi``, ``Dwb``), its weight the one knob.

Each cycle it asks the impact model, for every candidate and every imagined
fault: if this fault hit now and the robot, not noticing, kept sending this
candidate's commands, where would the trunk go? The candidate pays, per
fault, the worst over its steps of P(the footprint touches an object) times
the object's severity times the trunk's kinetic energy there, summed over
the map's severity levels; the faults are weighted by how often the
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

import torch

from failbench.faults import LOCK, NOMINAL, TORQUE, FaultMixture, encode_fault
from failbench.mppi.costmap import FootprintCfg, hit_prob
from failbench.mppi.inputs import deadband, integrate_plan
from failbench.mppi.models import DynamicsModel


@dataclass(frozen=True)
class RiskCriticCfg:
  weight: float = 0.01
  """The knob: cost per unit of expected damage given a fault (severity
  times joules). A glancing 1 m/s hit on glass (25) is 190 units; Nav2's
  critics differ between samples by a few units."""
  mixture: FaultMixture = field(default_factory=FaultMixture)
  """The evaluation's fault distribution, which the imagined faults stand
  for."""
  joints: tuple[int, ...] | None = None
  """The joints imagined faulting (None: all); each stands for its share."""
  weak_torque: float = 0.25
  """The torque scale standing for a weak (not limp) joint: the mean of
  ``mixture.weak_torque``."""
  mass_kg: float = 15.0
  """The Go2's mass, for the kinetic energy."""
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
  prior = torch.tensor(prior)
  return torch.stack(params), prior / prior.sum()


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

  def _price(self, cmd: torch.Tensor, data: dict) -> torch.Tensor:
    """The weighted expected damage ``[n, K]`` of commands ``[n, K, T, 3]``."""
    c = self.cfg
    n, k = cmd.shape[:2]
    fields = data["risk_fields"]
    inputs = data["inputs"]
    params, prior = imagined_faults(c, inputs["fault"].shape[-1] // 2)
    faults = encode_fault(params).to(cmd)
    prior = prior.to(cmd)
    nf = len(faults)
    path = integrate_plan(cmd, c.control_dt, c.substeps)
    # Every robot once per imagined fault: rows [n * F].
    rep = {key: v.repeat_interleave(nf, 0) for key, v in inputs.items()}
    rep["fault"] = faults.repeat(n, 1).to(inputs["fault"].dtype)
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
    return c.weight * (cost * prior).sum(-1)

  def _damage(
    self,
    pred: dict[str, torch.Tensor],
    path: torch.Tensor,
    fields: list[tuple[float, torch.Tensor]],
    resolution: float,
  ) -> torch.Tensor:
    """Expected damage ``[n * F, K]`` of predicted throws off ``path``."""
    c = self.cfg
    pose = path + pred["drift"]
    if "vel" in pred:
      speed2 = pred["vel"].square().sum(-1)
    else:
      start = torch.zeros_like(pose[..., :1, :2])
      step = torch.diff(pose[..., :2], dim=-2, prepend=start)
      speed2 = (step / (c.control_dt * c.substeps)).square().sum(-1)
    energy = 0.5 * c.mass_kg * speed2
    nf = pose.shape[0] // fields[0][1].shape[0]
    cost = pose.new_zeros(pose.shape[:2])
    for severity, dist in fields:
      touch = hit_prob(
        dist.repeat_interleave(nf, 0), pose, pred["drift_std"], resolution, c.footprint
      )
      cost = cost + severity * (touch * energy).amax(-1)
    return cost
