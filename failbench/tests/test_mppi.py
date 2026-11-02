"""Tests for the MPPI planner: input parity with the data builder, the
obstacle sweep and the planner on a synthetic map."""

import math
from dataclasses import replace
from typing import cast

import pytest
import torch
from failbench.data.build import _integrate
from failbench.faults import NOMINAL
from failbench.mppi.cbf import CbfCfg, CbfFilter, severity_fields
from failbench.mppi.costmap import (
  FootprintCfg,
  distance_field,
  drop_cells,
  obstacle_cells,
  sweep,
)
from failbench.mppi.inputs import HistoryBuffer, crop_center, integrate_plan
from failbench.mppi.models import ZeroDriftModel
from failbench.mppi.planner import (
  CostWeights,
  FaultRiskCfg,
  InflationCfg,
  Mppi,
  MppiCfg,
  risk_weights,
)
from failbench.mppi.sampler import PlanSampler, SamplerCfg
from failbench.sensing import cell_centres

RES = 0.05


def test_integrate_plan_matches_the_builders_cmd_path():
  torch.manual_seed(0)
  plan = torch.rand(4, 50, 3) - 0.5
  # build.py integrates the 50 Hz command and reads every 5th step.
  ref = _integrate(plan.repeat_interleave(5, dim=1), 0.02)[:, 4::5]
  assert torch.allclose(integrate_plan(plan, 0.02, 5), ref, atol=1e-5)


def test_history_matches_the_builders_window():
  # build.py: history at step s is steps s - 19 r .. s (stride r), oldest first.
  hist = HistoryBuffer(1, 20, 1, "cpu")
  r, s = 5, 200
  for step in range(r, s + 1, r):
    hist.push(torch.tensor([[float(step)]]))
  expected = s - r * torch.arange(19, -1, -1)
  assert torch.equal(hist.data[0, :, 0], expected.float())


def test_centre_crop_shares_the_model_grid():
  big, small = cell_centres(160, RES, "cpu"), cell_centres(64, RES, "cpu")
  crop = crop_center(big.permute(2, 0, 1), 64).permute(1, 2, 0)
  assert torch.allclose(crop, small)


def test_stairs_are_not_walls_but_crates_and_poles_are():
  x = torch.arange(160) * RES
  stairs = (torch.clamp((x - 2.0) // 0.3 + 1, min=0) * 0.14).expand(160, 160)
  assert not obstacle_cells(stairs[None], 0.18).any()
  crate = torch.zeros(1, 160, 160)
  crate[:, 70:76, 70:76] = 0.2
  pole = torch.zeros(1, 160, 160)
  pole[:, 80, 80] = 1.0
  assert obstacle_cells(crate, 0.18).any()
  assert obstacle_cells(pole, 0.18)[0, 79:82, 79:82].any()


def test_a_ledge_is_a_drop_but_a_wall_or_stairs_down_are_not():
  cfg = FootprintCfg()
  h = torch.zeros(1, 160, 160)
  h[:, 37:40, :] = 0.8  # A 0.15 m wall 2 m to the right (-y).
  h[:, 110:, :] = -0.6  # A ledge 1.5 m to the left (+y).
  drop = drop_cells(h, cfg, RES)
  assert drop[0, 110:116].all() and not drop[0, 116:].any()
  assert not drop[0, :110].any()
  # Unknown ground (beyond the LiDAR's view) isn't a plateau over a drop.
  h[:, :20] = float("nan")
  assert not drop_cells(h, cfg, RES)[0, :110].any()
  x = torch.arange(160) * RES
  stairs = -(torch.clamp((x - 2.0) // 0.3 + 1, min=0) * 0.14).expand(1, 160, 160)
  assert not drop_cells(stairs, cfg, RES).any()


def test_sweep_charges_walls_but_not_open_ground():
  cfg = FootprintCfg()
  h = torch.zeros(1, 160, 160)
  h[:, 100:104, :] = 0.8  # A wall 1 m to the left (+y), along x.
  dist = distance_field(obstacle_cells(h, cfg.wall_jump_m), RES, cfg.max_dist_m)
  known = torch.ones_like(h, dtype=torch.bool)
  pose = torch.zeros(1, 2, 1, 3)
  pose[0, 1, 0, 1] = 0.9  # Second plan ends against the wall.
  std = torch.full_like(pose, 0.01)
  pen, unknown = sweep(dist, known, pose, std, RES, cfg)
  assert pen[0, 0].max() == 0 and pen[0, 1].max() > 0
  assert not unknown.any()
  # However uncertain, a plan in open ground isn't charged.
  free = distance_field(torch.zeros_like(h, dtype=torch.bool), RES, cfg.max_dist_m)
  pen, _ = sweep(free, known, pose[:, :1], torch.full_like(pose[:, :1], 5.0), RES, cfg)
  assert pen.max() == 0


def test_sampled_plans_respect_limits_and_committed_steps():
  sampler = PlanSampler(SamplerCfg(num_plans=32, compile=False), "cpu")
  warm = torch.rand(2, 50, 3)
  plans = sampler.sample(warm, warm[:, 0], torch.tensor([[5.0, 0.0]] * 2), None, 1)
  assert plans.shape == (2, 32, 50, 3)
  assert (plans[..., 0] >= 0).all() and (plans[..., 0] <= 1.2 + 1e-6).all()
  assert torch.equal(plans[:, :, 0], warm[:, None, 0].expand(2, 32, 3))


def test_log_mppi_noise_keeps_the_variance_with_heavier_tails():
  torch.manual_seed(0)
  draws = {}
  for s in (0.0, 0.8):
    cfg = SamplerCfg(num_plans=4000, compile=False, log_normal_std=s)
    eps = PlanSampler(cfg, "cpu")._noise(4)[..., 0].flatten() / cfg.noise_std[0]
    draws[s] = eps
  gauss, log_mppi = draws[0.0], draws[0.8]
  assert abs(log_mppi.var() / gauss.var() - 1) < 0.1
  kurt = [(e**4).mean() / e.var() ** 2 for e in (gauss, log_mppi)]
  assert kurt[1] > 2 * kurt[0]


def _plan(h: torch.Tensor, goal: torch.Tensor, cycles: int = 10) -> torch.Tensor:
  torch.manual_seed(0)
  cfg = MppiCfg(sampler=SamplerCfg(num_plans=128, compile=False), delay_steps=0)
  planner = Mppi(cfg, ZeroDriftModel(), 1, "cpu")
  known = torch.ones_like(h, dtype=torch.bool)
  out = planner.step({}, h, known, RES, goal)
  for _ in range(cycles - 1):
    out = planner.step({}, h, known, RES, goal)
  return out["pred_path"][0]


def test_mppi_walks_toward_the_goal_in_the_open():
  goal = torch.tensor([[3.0, 0.5]])
  path = _plan(torch.zeros(1, 160, 160), goal)
  assert (path[-1, :2] - goal[0]).norm() < 1.0


def test_mppi_stops_short_of_a_wall_across_the_way():
  h = torch.zeros(1, 160, 160)
  h[:, :, 110:114] = 0.8  # A wall across the whole map 1.5 m ahead.
  path = _plan(h, torch.tensor([[3.0, 0.0]]))
  # Front circle (0.22 m ahead, radius 0.2) stays off the wall's face.
  assert path[:, 0].max() < 1.5 - 0.22 - 0.2


def test_obstacle_cost_is_a_hit_chance_and_a_crash_costs_more_than_a_graze():
  cfg = MppiCfg(sampler=SamplerCfg(num_plans=64, compile=False), delay_steps=0)
  w = cfg.weights
  planner = Mppi(cfg, ZeroDriftModel(), 1, "cpu")
  h = torch.zeros(1, 160, 160)
  h[:, :, 110:114] = 0.8  # A wall across the whole map 1.5 m ahead.
  dist = distance_field(obstacle_cells(h, cfg.footprint.wall_jump_m), RES, 1.5)
  known = torch.ones_like(h, dtype=torch.bool)
  # Stand, walk up to the wall (front circle 0.08 m in at 5 s), walk into it.
  plans = torch.zeros(1, 3, 50, 3)
  plans[0, 1, :, 0], plans[0, 2, :, 0] = 0.24, 0.5
  terms = planner.score({}, plans, dist, known, RES, torch.tensor([[0.0, 0.0]]))[
    "terms"
  ]
  hit, depth = terms["obstacle"][0], terms["obstacle_depth"][0]
  assert hit[0] < 0.01 * w.obstacle and depth[0] == 0
  # Both moving plans are near-certain hits, which the probability can't
  # tell apart; the mean footprint's depth does.
  assert (hit[1:] > 0.5 * w.obstacle).all() and (hit <= w.obstacle).all()
  assert 0 < depth[1] < depth[2]


class _RiskyToMove(ZeroDriftModel):
  """Any plan that walks has a 60% chance of trunk contact."""

  def predict(self, inputs, cmd_seq, cmd_path, contact_maps=False):
    out = super().predict(inputs, cmd_seq, cmd_path, contact_maps)
    walks = (cmd_seq[..., :2].norm(dim=-1) > 0.2).any(-1, keepdim=True)
    out["p_collision"] = (0.6 * walks.float()).expand_as(out["p_fallen"])
    return out


def test_idle_cost_gets_a_robot_that_has_stood_too_long_moving():
  torch.manual_seed(0)
  cfg = MppiCfg(
    sampler=SamplerCfg(num_plans=128, compile=False),
    weights=CostWeights(collision=5.0),
    delay_steps=0,
  )
  h = torch.zeros(1, 160, 160)
  known = torch.ones_like(h, dtype=torch.bool)
  goal = torch.tensor([[0.8, 0.0]])
  planner = Mppi(cfg, _RiskyToMove(), 1, "cpu")
  pos = torch.zeros(1, 2)
  planner.track(pos, 0.1)
  out = planner.step({}, h, known, RES, goal)
  for _ in range(2):
    planner.track(pos, 0.1)
    out = planner.step({}, h, known, RES, goal)
  # Walking costs more than the goal is worth.
  assert out["plan"][0, :10, :2].norm(dim=-1).max() == 0
  for _ in range(100):  # 10 s still: standing now costs the full idle weight.
    planner.track(pos, 0.1)
  out = planner.step({}, h, known, RES, goal)
  assert out["plan"][0, :10, 0].max() > 0.2  # Ramping up from standing.
  planner.track(pos + 0.5, 0.1)
  assert planner.still_s[0] == 0


class _Thrown(ZeroDriftModel):
  """A fault on an even joint throws the robot 0.8 m to its right (-y)
  within a second, one on an odd joint (``side=-1``: also even) to its
  left."""

  def __init__(self, side: float = 1.0):
    super().__init__()
    self.side = side

  def predict(self, inputs, cmd_seq, cmd_path, contact_maps=False):
    out = super().predict(inputs, cmd_seq, cmd_path, contact_maps)
    hurt = inputs["fault"].view(-1, 12, 2) != torch.tensor(NOMINAL)
    joint = hurt.any(-1).float().argmax(-1)
    faulty = hurt.any((-1, -2))
    sign = torch.where(joint % 2 == 0, -1.0, self.side)
    t = torch.arange(1, cmd_seq.shape[2] + 1) * self.step_s
    throw = 0.8 * (t / 1.0).clamp(max=1.0)
    out["drift"][..., 1] = (faulty * sign)[:, None, None] * throw
    return out


def _side(model, h, severity=None, p=0.1, cycles=10):
  """Mean lateral offset (m, +y) of the plan's first 2 s after ``cycles``."""
  torch.manual_seed(0)
  cfg = MppiCfg(
    sampler=SamplerCfg(num_plans=128, compile=False),
    delay_steps=0,
    gap_quantile=0.1,  # Fault risk differences are small next to a crash.
    fault_risk=FaultRiskCfg(p=p, modes=("free",)),
  )
  planner = Mppi(cfg, model, 1, "cpu")
  known = torch.ones_like(h, dtype=torch.bool)
  inputs = {"fault": torch.tensor(NOMINAL).repeat(12)[None]}
  goal = torch.tensor([[3.0, 0.0]])
  kw = {"severity": severity}
  out = planner.step(inputs, h, known, RES, goal, **kw)
  for _ in range(cycles - 1):
    out = planner.step(inputs, h, known, RES, goal, **kw)
  assert ("cost_fault_risk" in out) == (p > 0)
  return out["pred_path"][0, :20, 1].mean(), out


def test_fault_risk_keeps_a_healthy_robot_away_from_a_ledge():
  h = torch.zeros(1, 160, 160)
  h[:, :66] = -0.6  # A ledge 0.7 m to the right (-y), along x.
  thrown_right = _Thrown(side=-1.0)
  stays, _ = _side(thrown_right, h, p=0.0)
  keeps_off, _ = _side(thrown_right, h, p=0.1)
  # Thrown right if a joint fails, so it keeps left once that is priced.
  assert keeps_off > stays + 0.15
  # In open ground nothing is in reach: no risk.
  _, out = _side(thrown_right, torch.zeros(1, 160, 160))
  assert out["cost_fault_risk"].abs().max() == 0


def test_fault_risk_keeps_further_from_the_more_severe_side():
  h = torch.zeros(1, 160, 160)
  h[:, 55:59] = 0.5  # Boxes 1.08 m right...
  h[:, 101:105] = 0.5  # ...and left, the same.
  right_worse = torch.where(torch.arange(160)[:, None] < 80, 25.0, 5.0)
  right_worse = right_worse.expand(1, 160, 160)
  left_worse = right_worse.flip(1)
  away_from_right, _ = _side(_Thrown(), h, right_worse)
  away_from_left, _ = _side(_Thrown(), h, left_worse)
  # The same boxes; only what they are differs, and the choice mirrors.
  # (Small: the goal is on the centre line, and the cost-to-go knows the
  # robot must come back to it.)
  assert away_from_right > 0.02 and away_from_left < -0.02


def _wall_costs(at_x: float, weights: CostWeights) -> dict[str, torch.Tensor]:
  """Cost terms of walking straight into a wall ``at_x`` m ahead."""
  h = torch.zeros(1, 160, 160)
  cx = 80 + round(at_x / RES)
  h[:, :, cx : cx + 3] = 0.8
  cfg = MppiCfg(
    sampler=SamplerCfg(num_plans=16, compile=False),
    weights=weights,
    fault_risk=FaultRiskCfg(p=0.05, modes=("free",)),
  )
  planner = Mppi(cfg, _Thrown(), 1, "cpu")
  fp = cfg.footprint
  obst = obstacle_cells(h, fp.wall_jump_m)
  plans = torch.zeros(1, 1, 50, 3)
  plans[..., 0] = 0.8
  inputs = {"fault": torch.tensor(NOMINAL).repeat(12)[None]}
  sc = planner.score(
    inputs,
    plans,
    distance_field(obst, RES, fp.max_dist_m),
    torch.ones_like(h, dtype=torch.bool),
    RES,
    torch.tensor([[6.0, 0.0]]),
    risk_fields=planner._risk_fields(obst, obst & False, None, RES),
  )
  return sc["terms"]


def test_a_nearer_hit_costs_more_when_discounted():
  ratio = {}
  for gamma in (1.0, 0.7):
    w = CostWeights(discount_per_s=gamma)
    near, far = _wall_costs(1.0, w), _wall_costs(3.0, w)
    ratio[gamma] = {k: near[k] / far[k] for k in ("obstacle", "fault_risk")}
  # Discounting moves the weight toward the nearer wall, by about
  # 0.7^-2.5 (2.5 s further at 0.8 m/s).
  for k in ("obstacle", "fault_risk"):
    assert ratio[0.7][k] > 2 * ratio[1.0][k]
  near = _wall_costs(1.0, CostWeights(discount_per_s=0.7))
  # The weight scales the fault risk and nothing else.
  heavy = _wall_costs(1.0, CostWeights(discount_per_s=0.7, fault_risk=5.0))
  assert torch.isclose(heavy["fault_risk"], 5 * near["fault_risk"])
  assert torch.equal(heavy["obstacle"], near["obstacle"])


def test_fault_risk_prices_only_the_named_faults():
  cfg = MppiCfg(fault_risk=FaultRiskCfg(p=0.2, mode_share=(3.0, 1.0)))
  branches, prior = Mppi(cfg, _Thrown(), 1, "cpu")._branches(12, "cpu")
  params = branches.view(-1, 12, 2)
  assert len(branches) == 8 and torch.isclose(prior.sum(), torch.tensor(0.2))
  # Front hips go limp (torque 0); the limp branches share 3/4 of p.
  assert params[0, 0, 0] == 0 and params[1, 3, 0] == 0
  assert torch.allclose(prior[:4], torch.tensor(0.15 / 4))
  limp_only = MppiCfg(fault_risk=FaultRiskCfg(p=0.2, modes=("free",)))
  branches, prior = Mppi(limp_only, _Thrown(), 1, "cpu")._branches(12, "cpu")
  assert len(branches) == 4 and torch.allclose(prior, torch.tensor(0.05))


def test_fault_risk_reports_each_branch():
  h = torch.zeros(1, 160, 160)
  h[:, :66] = -0.6  # A ledge 0.7 m to the right.
  _, out = _side(_Thrown(), h, p=0.1, cycles=2)
  names = Mppi(MppiCfg(fault_risk=FaultRiskCfg(modes=("free",))), _Thrown(), 1, "cpu")
  assert out["risk_branch"].shape == (1, 4)
  assert names.branch_names()[0] == "FL_hip_joint:free"
  # FL hip (even joint) is thrown right, onto the ledge; FR hip left (only
  # the route's last turn toward the goal brings the ledge near).
  branch = out["risk_branch"][0]
  assert branch[0] > 0 and branch[1] < 0.05 * branch[0]
  assert out["risk_thrown"].shape[:3] == (1, 16, 4)


def test_risk_measures_span_mean_to_worst():
  cost = torch.tensor([[4.0, 0.0, 10.0, 1.0]])
  prob = torch.full((4,), 0.025)  # p = 0.1 over four outcomes.

  def measure(**kw) -> float:
    fr = FaultRiskCfg(p=0.1, **kw)
    return float((risk_weights(cost, prob, fr) * cost).sum())

  mean = measure(measure="mean")
  assert mean == pytest.approx(0.375)
  # CVaR over everything is the mean; over the fault mass, the mean given
  # a fault; over the worst outcome's mass or less, the worst.
  assert measure(measure="cvar", cvar_alpha=1.0) == pytest.approx(mean)
  assert measure(measure="cvar", cvar_alpha=0.1) == pytest.approx(3.75)
  assert measure(measure="cvar", cvar_alpha=0.05) == pytest.approx(7.0)
  assert measure(measure="cvar", cvar_alpha=0.01) == pytest.approx(10.0)
  assert measure(measure="max") == pytest.approx(10.0)


def test_tail_measures_price_the_ledge_more():
  h = torch.zeros(1, 160, 160)
  h[:, :66] = -0.6  # A ledge 0.7 m to the right.
  costs = {}
  for m in ("mean", "cvar", "max"):
    torch.manual_seed(0)
    cfg = MppiCfg(
      sampler=SamplerCfg(num_plans=16, compile=False),
      delay_steps=0,
      fault_risk=FaultRiskCfg(p=0.1, modes=("free",), measure=m),
    )
    planner = Mppi(cfg, _Thrown(), 1, "cpu")
    inputs = {"fault": torch.tensor(NOMINAL).repeat(12)[None]}
    known = torch.ones_like(h, dtype=torch.bool)
    out = planner.step(inputs, h, known, RES, torch.tensor([[3.0, 0.0]]))
    costs[m] = float(out["risk_branch"].sum())
  assert 0 < costs["mean"] < costs["cvar"] <= costs["max"] * (1 + 1e-4)


class _Falls(ZeroDriftModel):
  """Every plan falls within its first second."""

  def predict(self, inputs, cmd_seq, cmd_path, contact_maps=False):
    out = super().predict(inputs, cmd_seq, cmd_path, contact_maps)
    out["p_fallen"] = (
      (torch.arange(cmd_seq.shape[2]) >= 10).float().expand_as(out["p_fallen"])
    )
    return out


def test_a_fall_costs_only_what_it_lands_on():
  cfg = MppiCfg(sampler=SamplerCfg(num_plans=16, compile=False), delay_steps=0)
  planner = Mppi(cfg, _Falls(), 1, "cpu")
  plans = torch.zeros(1, 1, 50, 3)  # Standing, and falling.
  fp = cfg.footprint
  fall = {}
  for name, wall_y in (("open", None), ("glass", 0.3)):
    h = torch.zeros(1, 160, 160)
    if wall_y is not None:
      cy = 80 + round(wall_y / RES)
      h[:, cy : cy + 3] = 0.8  # Beside the robot, along x.
    obst = obstacle_cells(h, fp.wall_jump_m)
    severity = torch.full_like(h, 25.0)
    sc = planner.score(
      {},
      plans,
      distance_field(obst, RES, fp.max_dist_m),
      torch.ones_like(h, dtype=torch.bool),
      RES,
      torch.tensor([[0.0, 0.0]]),
      risk_fields=planner._risk_fields(obst, obst & False, severity, RES),
    )
    fall[name] = float(sc["terms"]["fall"][0, 0])
  assert fall["open"] == 0
  assert fall["glass"] > 0.5 * 25


def test_the_margin_grows_with_severity_and_speed_only_when_asked():
  h = torch.zeros(1, 160, 160)
  h[:, 92:95] = 0.8  # A wall along x, 0.6 m left: 0.4 m past the footprint.
  plans = torch.zeros(1, 2, 50, 3)
  plans[0, 1, :, 0] = 0.8  # Stand, or walk past it at cruise speed.

  def reach(inflation: InflationCfg, severity: float) -> list[float]:
    cfg = MppiCfg(sampler=SamplerCfg(num_plans=16, compile=False), inflation=inflation)
    planner = Mppi(cfg, ZeroDriftModel(), 1, "cpu")
    fp = cfg.footprint
    obst = obstacle_cells(h, fp.wall_jump_m)
    sc = planner.score(
      {},
      plans,
      distance_field(obst, RES, fp.max_dist_m),
      torch.ones_like(h, dtype=torch.bool),
      RES,
      torch.tensor([[3.0, 0.0]]),
      risk_fields=planner._risk_fields(
        obst, obst & False, torch.full_like(h, severity), RES
      ),
    )
    if inflation.margin == 0:
      assert "inflation" not in sc["terms"]
      return [0.0, 0.0]
    return (sc["terms"]["inflation"][0] / cfg.weights.inflation).tolist()

  uniform = InflationCfg(margin=0.5)
  semantic = InflationCfg(margin=0.5, severity_ref=25.0, speed_ref=0.8)
  # Uniform: 0.1 m into the margin, whatever the wall is and however fast.
  for severity in (5.0, 25.0):
    assert reach(uniform, severity) == pytest.approx([0.1, 0.1], abs=0.03)
  # Semantic: glass keeps the full margin, crates a fifth of it (clear);
  # walking at cruise doubles it.
  assert reach(semantic, 25.0) == pytest.approx([0.1, 0.6], abs=0.03)
  assert reach(semantic, 5.0) == [0.0, 0.0]
  assert reach(InflationCfg(), 25.0) == [0.0, 0.0]


def test_standing_still_pays_the_fault_risk_of_the_ground_ahead():
  h = torch.zeros(1, 160, 160)
  h[:, :66, 112:] = -0.6  # A ledge to the right (-y), from 1.6 m ahead on.
  costs = {}
  base = FaultRiskCfg(p=0.1, modes=("free",))
  layouts = {
    "to_goal": base,
    "per_m": replace(base, fire_step_m=None),
    "per_s": replace(base, fire_step_m=None, fire_m=None),
  }
  for name, layout in layouts.items():
    cfg = MppiCfg(sampler=SamplerCfg(num_plans=16, compile=False), fault_risk=layout)
    planner = Mppi(cfg, _Thrown(side=-1.0), 1, "cpu")
    fp = cfg.footprint
    obst, drops = obstacle_cells(h, fp.wall_jump_m), drop_cells(h, fp, RES)
    plans = torch.zeros(1, 2, 50, 3)
    # Stand, or walk on at the cruise speed the ground ahead is priced at.
    plans[0, 1, :, 0] = layout.cruise_vx or 0.3
    sc = planner.score(
      {"fault": torch.tensor(NOMINAL).repeat(12)[None]},
      plans,
      distance_field(obst, RES, fp.max_dist_m),
      torch.ones_like(h, dtype=torch.bool),
      RES,
      torch.tensor([[6.0, 0.0]]),
      drop_dist=distance_field(drops, RES, fp.max_dist_m),
      risk_fields=planner._risk_fields(obst, drops, None, RES),
    )
    costs[name] = sc["terms"]["fault_risk"][0]
  # Faults by the metre: standing only puts the ledge off, at the same risk.
  for name in ("to_goal", "per_m"):
    assert costs[name][1] > 0
    assert torch.isclose(costs[name][0], costs[name][1], rtol=0.01)
  # Faults by the second: standing where nothing is in reach looks safe.
  per_s = costs["per_s"]
  assert per_s[0] < 0.05 * per_s[1]


def test_speed_share_zero_prices_only_where_a_plan_walks():
  h = torch.zeros(1, 160, 160)
  h[:, :66, 112:] = -0.6  # A ledge to the right (-y), from 1.6 m ahead on.
  costs = {}
  for share in (0.0, 1.0):
    fr = FaultRiskCfg(p=0.1, modes=("free",), speed_share=share)
    cfg = MppiCfg(sampler=SamplerCfg(num_plans=16, compile=False), fault_risk=fr)
    planner = Mppi(cfg, _Thrown(side=-1.0), 1, "cpu")
    fp = cfg.footprint
    obst, drops = obstacle_cells(h, fp.wall_jump_m), drop_cells(h, fp, RES)
    plans = torch.zeros(1, 2, 50, 3)
    plans[0, 0, :, 0] = 0.4  # The same line, slow or fast.
    plans[0, 1, :, 0] = 1.1
    pose = planner._path(plans)
    risk = planner._fault_risk(
      planner._throws({"fault": torch.tensor(NOMINAL).repeat(12)[None]}),
      plans,
      pose,
      planner._risk_fields(obst, drops, None, RES),
      RES,
      torch.tensor([[6.0, 0.0]]),
    )
    costs[share] = cast(torch.Tensor, risk)[0]
  # Steering only: the same ground costs the same at any speed.
  assert torch.isclose(costs[0.0][0], costs[0.0][1], rtol=0.01)
  # The full price: the faster plan's throws reach further along the ledge.
  assert costs[1.0][1] > costs[1.0][0]


def test_turning_on_the_spot_does_not_dodge_the_ground_ahead():
  h = torch.zeros(1, 160, 160)
  h[:, :66, 112:] = -0.6  # A ledge to the right (-y), from 1.6 m ahead on.
  cfg = MppiCfg(
    sampler=SamplerCfg(num_plans=16, compile=False),
    fault_risk=FaultRiskCfg(p=0.1, modes=("free",)),
  )
  planner = Mppi(cfg, _Thrown(side=-1.0), 1, "cpu")
  fp = cfg.footprint
  obst, drops = obstacle_cells(h, fp.wall_jump_m), drop_cells(h, fp, RES)
  plans = torch.zeros(1, 2, 50, 3)
  plans[0, 1, :, 2] = 0.3  # Stand, or turn left on the spot, away from it.
  sc = planner.score(
    {"fault": torch.tensor(NOMINAL).repeat(12)[None]},
    plans,
    distance_field(obst, RES, fp.max_dist_m),
    torch.ones_like(h, dtype=torch.bool),
    RES,
    torch.tensor([[6.0, 0.0]]),
    drop_dist=distance_field(drops, RES, fp.max_dist_m),
    risk_fields=planner._risk_fields(obst, drops, None, RES),
  )
  # The ground still to walk lies toward the goal, whichever way it faces.
  risk = sc["terms"]["fault_risk"][0]
  assert risk[0] > 0 and torch.isclose(risk[0], risk[1], rtol=0.01)


def test_looping_back_through_the_open_does_not_dodge_the_hazard_ahead():
  h = torch.zeros(1, 160, 160)
  h[:, 60:72, 120:150] = 0.8  # Glass 0.4-1.0 m right (-y), 2.0-3.5 m ahead.
  severity = torch.full_like(h, 25.0)
  plans = torch.zeros(1, 2, 50, 3)
  plans[0, :, :, 0] = 0.6  # Walk on past it, or...
  plans[0, 1, :, 2] = 2 * math.pi / 5.0  # ...loop left, back to the start.
  risk = {}
  base = FaultRiskCfg(p=0.1, modes=("free",))
  for name, layout in (("next_3m", replace(base, fire_step_m=None)), ("to_goal", base)):
    cfg = MppiCfg(sampler=SamplerCfg(num_plans=16, compile=False), fault_risk=layout)
    planner = Mppi(cfg, _Thrown(side=-1.0), 1, "cpu")  # Thrown right.
    fp = cfg.footprint
    obst = obstacle_cells(h, fp.wall_jump_m)
    sc = planner.score(
      {"fault": torch.tensor(NOMINAL).repeat(12)[None]},
      plans,
      distance_field(obst, RES, fp.max_dist_m),
      torch.ones_like(h, dtype=torch.bool),
      RES,
      torch.tensor([[6.0, 0.0]]),
      risk_fields=planner._risk_fields(obst, obst & False, severity, RES),
    )
    risk[name] = sc["terms"]["fault_risk"][0]
  # Pricing the next 3 m only, the loop spends them in the open and looks
  # far safer; with the route to the goal priced, it still owes the glass.
  # (A loop that ended on a line clear of it would rightly cost less.)
  assert risk["next_3m"][1] < 0.5 * risk["next_3m"][0]
  assert risk["to_goal"][1] >= 0.95 * risk["to_goal"][0]


def _filtered(cmd, rows=None, severity=25.0, margin=0.3, **kw):
  """The CBF's command for ``cmd`` beside a wall filling map ``rows`` (+y
  is up the rows) or ``cols`` (+x), and its logs."""
  h = torch.zeros(1, 160, 160)
  if rows is not None:
    h[:, rows] = 0.8
  if "cols" in kw:
    h[:, :, kw.pop("cols")] = 0.8
  cfg = CbfCfg(margin=margin, **kw)
  fields = severity_fields(h, torch.full_like(h, severity), RES, cfg.footprint)
  safe, log = CbfFilter(cfg).filter(torch.tensor([cmd]), fields, torch.zeros(1), RES)
  return safe[0], log


def test_the_cbf_passes_commands_through_in_open_ground():
  safe, log = _filtered((0.8, 0.1, 0.3))
  assert torch.allclose(safe, torch.tensor([0.8, 0.1, 0.3]))
  assert log["active"][0] == 0


def test_the_cbf_slows_the_approach_to_a_wall_but_not_the_walk_along_it():
  # A wall 1.03 m ahead: the front circle is 0.81 m off, 0.31 m outside a
  # 0.3 m margin, so it may close at gamma h / dt = 0.61 m/s.
  safe, log = _filtered((1.0, 0.3, 0.0), cols=slice(100, 103))
  assert safe[0] == pytest.approx(0.61, abs=0.03)
  assert safe[1] == pytest.approx(0.3, abs=0.01)
  assert log["active"][0] >= 1


def test_the_semantic_cbf_keeps_off_glass_but_not_crates():
  toward = (0.0, 0.5, 0.0)  # Strafing at a wall 0.62 m left.
  wall = slice(92, 95)
  glass, _ = _filtered(toward, wall, 25.0, severity_ref=25.0)
  crates, _ = _filtered(toward, wall, 5.0, severity_ref=25.0)
  uniform_crates, _ = _filtered(toward, wall, 5.0)
  assert glass[1] < 0.35  # 0.12 m from glass's margin: at most ~0.25 m/s.
  assert crates[1] == pytest.approx(0.5, abs=0.01)  # A 0.06 m margin.
  assert uniform_crates[1] < 0.35  # Uniform: crates as glass.


def test_the_cbf_pushes_out_of_the_margin_and_limits_turning_into_it():
  # 0.42 m from a wall with a 0.3 m margin: 0.08 m inside, standing still.
  inside, _ = _filtered((0.0, 0.0, 0.0), slice(88, 91))
  assert inside[1] < -0.1
  # Turning toward a wall swings the front circle at it (0.22 m/s per
  # rad/s): with 0.075 m to the margin it may close at 0.15 m/s.
  turn, _ = _filtered((0.0, 0.0, 1.0), slice(89, 92), margin=0.2)
  assert turn[2] < 0.9
  assert turn[1] + 0.22 * turn[2] <= 0.16
