"""Tests for the MPPI planner: input parity with the data builder, the
obstacle sweep and the planner on a synthetic map."""

import torch
from failbench.data.build import _integrate
from failbench.faults import NOMINAL
from failbench.mppi.costmap import (
  FootprintCfg,
  distance_field,
  drop_cells,
  obstacle_cells,
  sweep,
)
from failbench.mppi.inputs import HistoryBuffer, crop_center, integrate_plan
from failbench.mppi.models import ZeroDriftModel
from failbench.mppi.planner import CostWeights, FaultRiskCfg, Mppi, MppiCfg
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


class _RiskyToMove(ZeroDriftModel):
  """Any plan that walks has a 15% chance of a fall."""

  def predict(self, inputs, cmd_seq, cmd_path, contact_maps=False):
    out = super().predict(inputs, cmd_seq, cmd_path, contact_maps)
    walks = (cmd_seq[..., :2].norm(dim=-1) > 0.2).any(-1, keepdim=True)
    out["p_fallen"] = (0.15 * walks.float()).expand_as(out["p_fallen"])
    return out


def test_idle_cost_gets_a_robot_that_has_stood_too_long_moving():
  torch.manual_seed(0)
  cfg = MppiCfg(sampler=SamplerCfg(num_plans=128, compile=False), delay_steps=0)
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
  assert away_from_right > 0.05 and away_from_left < -0.05


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
  # FL hip (even joint) is thrown right, onto the ledge; FR hip left.
  assert out["risk_branch"][0, 0] > 0 and out["risk_branch"][0, 1] == 0
  assert out["risk_thrown"].shape[:3] == (1, 6, 4)
