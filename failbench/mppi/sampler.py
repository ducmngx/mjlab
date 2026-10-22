"""Candidate plans for MPPI, kept close to the plans the model trained on.

Per robot and cycle, ``k`` plans of ``steps`` twists (``step_s`` each):

- the previous cycle's plan shifted one step (the warm start), a hold of
  the current command, a stop, and straight cruises at ``cruise_vx``
  (ramped at the plans' ``max_accel``), which also probe why MPPI won't
  move when it doesn't;
- ``segment_share`` fresh ``SegmentPlan`` draws (``failbench/plans.py``,
  the data's post-fault plans, lane plans on the course), started from the
  current command;
- the rest: the warm start plus smooth noise (knots every ``knot_s``,
  linearly interpolated).

All are clipped to the plan limits with vx never negative, and the first
``fixed_steps`` steps are the commands already committed (``delay``: a
plan computed now starts executing a cycle late).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch

from failbench.plans import PlanCfg, SegmentPlan

CRUISE_VX = (0.5, 0.8, 1.1)
FIXED_PLANS = 3 + len(CRUISE_VX)
"""Warm start, hold, stop, cruises."""


@dataclass(frozen=True)
class SamplerCfg:
  num_plans: int = 256
  steps: int = 50
  step_s: float = 0.1
  segment_share: float = 0.5
  plan: PlanCfg = field(default_factory=lambda: PlanCfg(hold_prob=0.0))
  """No random holds: the hold is always one of the fixed plans."""
  noise_std: tuple[float, float, float] = (0.25, 0.08, 0.35)
  knot_s: float = 1.0
  max_vx: float = 1.2
  """Top speed of the data's plans (``PlanCfg.speed``)."""
  compile: bool = True
  """``torch.compile`` the segment-plan step: about 5x faster, since a
  rollout is 50 steps of small ops."""


class PlanSampler:
  def __init__(self, cfg: SamplerCfg, device):
    self.cfg = cfg
    self.device = device
    self.num_segment = round(cfg.segment_share * (cfg.num_plans - FIXED_PLANS))
    self.num_noise = cfg.num_plans - FIXED_PLANS - self.num_segment
    if self.num_noise < 0:
      raise ValueError("num_plans is too small.")
    self.plan_cfg = replace(cfg.plan, limits=(cfg.max_vx, *cfg.plan.limits[1:]))
    self._plan: SegmentPlan | None = None
    self._step = None

  def _clip(self, plans: torch.Tensor) -> torch.Tensor:
    lim = torch.tensor(self.plan_cfg.limits, device=plans.device)
    plans = torch.maximum(torch.minimum(plans, lim), -lim)
    return torch.cat([plans[..., :1].clamp(min=0.0), plans[..., 1:]], dim=-1)

  def _segment_plans(
    self,
    c0: torch.Tensor,
    goal: torch.Tensor,
    lane_bearing: torch.Tensor | None,
  ) -> torch.Tensor:
    n, m = c0.shape[0], self.num_segment
    c, dev = self.cfg, self.device
    if self._plan is None or self._plan.cmd.shape[0] != n * m:
      self._plan = SegmentPlan(n * m, self.plan_cfg, dev)
      step = self._plan.step
      self._step = torch.compile(step) if c.compile else step
    plan, plan_step = self._plan, self._step
    assert plan_step is not None
    ids = torch.arange(n * m, device=dev)
    lane = bearing = None
    if lane_bearing is not None:
      lane = torch.ones(n * m, dtype=torch.bool, device=dev)
      bearing = lane_bearing.repeat_interleave(m)
    plan.start(
      ids,
      c0.repeat_interleave(m, dim=0),
      torch.zeros(n * m, 3, device=dev),
      goal.repeat_interleave(m, dim=0),
      hold=torch.zeros(n * m, dtype=torch.bool, device=dev),
      lane=lane,
      bearing=bearing,
    )
    out = []
    for _ in range(c.steps):
      out.append(plan.cmd.clone())
      plan_step(None, c.step_s)
    return torch.stack(out, dim=1).view(n, m, c.steps, 3)

  def _noise(self, n: int) -> torch.Tensor:
    c = self.cfg
    knots = round(c.steps * c.step_s / c.knot_s) + 1
    std = torch.tensor(c.noise_std, device=self.device)
    eps = torch.randn(n, self.num_noise, knots, 3, device=self.device) * std
    u = torch.linspace(0, knots - 1, c.steps, device=self.device)
    lo = u.floor().long().clamp(max=knots - 2)
    w = (u - lo)[:, None]
    return eps[:, :, lo] * (1 - w) + eps[:, :, lo + 1] * w

  def sample(
    self,
    warm: torch.Tensor,
    c_now: torch.Tensor,
    goal: torch.Tensor,
    lane_bearing: torch.Tensor | None = None,
    fixed_steps: int = 0,
  ) -> torch.Tensor:
    """``[n, k, steps, 3]`` plans. ``warm [n, steps, 3]``: last cycle's plan
    shifted; ``c_now [n, 3]``: the command plans start from (the committed
    one with a delay); ``goal [n, 2]`` and ``lane_bearing [n]`` (lane plans,
    the course) in the robot's yaw frame."""
    c = self.cfg
    n = warm.shape[0]
    hold = c_now[:, None].expand(n, c.steps, 3)
    ramp = (1 - torch.arange(c.steps, device=self.device) / 10).clamp(min=0.0)
    stop = c_now[:, None] * ramp[None, :, None]
    k = torch.arange(1, c.steps + 1, device=self.device)
    cruises = []
    for vx in CRUISE_VX:
      step = self.plan_cfg.max_accel * c.step_s
      ramp_vx = c_now[:, :1] + (vx - c_now[:, :1]).clamp(-step * k, step * k)
      cruises.append(torch.stack([ramp_vx, 0 * ramp_vx, 0 * ramp_vx], dim=-1))
    fixed = torch.stack([warm, hold, stop, *cruises], dim=1)
    groups = [fixed]
    if self.num_segment:
      groups.append(self._segment_plans(c_now, goal, lane_bearing))
    if self.num_noise:
      groups.append(warm[:, None] + self._noise(n))
    plans = self._clip(torch.cat(groups, dim=1))
    if fixed_steps:
      plans[:, :, :fixed_steps] = warm[:, None, :fixed_steps]
    return plans
