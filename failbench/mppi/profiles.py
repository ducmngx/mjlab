"""Named Nav2 MPPI settings, and the record of what differs from stock.

``stock`` is Nav2's defaults with the Go2's limits (``Nav2MppiCfg()``).
Every other profile lists each parameter it changes, the new value and why,
so the paper can report them. ``changes_from_stock`` diffs any config
against stock (profile and command-line overrides alike); ``run_mppi``
writes it into every run's JSON.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass, replace
from typing import Any

from failbench.mppi.nav2 import Nav2MppiCfg


@dataclass(frozen=True)
class Change:
  param: str
  """Dotted field path in ``Nav2MppiCfg``: ``vy_std``,
  ``critics.path_align.weight``."""
  value: Any
  reason: str


STEADY_GRID = tuple(
  (vx, vy, wz)
  for vx in (0.6, 0.9, 1.2)
  for vy in (-0.3, -0.15, 0.0, 0.15, 0.3)
  for wz in (-0.2, 0.0, 0.2)
)
"""45 constant commands: three speeds, five sidesteps, three turn rates."""

PROFILES: dict[str, tuple[Change, ...]] = {
  "stock": (),
  # Tried and not adopted: +2-3 cm toward the crates on captured gate
  # cycles, but no change in closed loop on the four gate scenes.
  "sidestep": (
    Change(
      "noise_corr",
      (0.0, 0.97, 0.0),
      "Independent per-step noise averages a sample's sideways speed out over "
      "the horizon, so the batch holds almost no full-speed sidesteps. On "
      "captured gate cycles a steady 0.1 m/s strafe toward the crates cut the "
      "critic's risk about as much as slowing to 0.8x, at a tenth of the "
      "stock cost, but MPPI never sampled one.",
    ),
    Change(
      "wz_std",
      0.1,
      "With 0.4 rad/s of yaw noise most samples move sideways by turning, "
      "which costs path-align and heading; less yaw noise leaves the "
      "sideways spread to strafing (best single change in the offline "
      "screen).",
    ),
  ),
  "loose_path": (
    Change(
      "critics.path_align.weight",
      3.0,
      "At Nav2's 14, path align charges every step a plan spends off the "
      "path, so stepping away from a hazard costs more than the risk it "
      "saves and the cheaper answer is to slow down. At 3, on the lone glass "
      "case at risk weight 10 (8 robots, speed free), robots passed 9.5 / "
      "26.7 cm further away (glass left / right; stock -2.7 / +12.9) and "
      "lost 0.14 / 0.09 s to the goal (stock 0.25 / 0.28 s).",
    ),
  ),
  # Tried and not adopted: steady candidates and 3 passes per cycle didn't
  # move the robot off the glass with speed free (-2.4 cm, glass left).
  "steer": (
    Change(
      "steady_candidates",
      STEADY_GRID,
      "Noise about the mean almost never holds one sideways course for the "
      "whole 3 s, so a plan that steps off a hazard and stays off is rarely "
      "in the batch even when the critics price it lowest. A small grid of "
      "constant commands (as DWB samples) always offers it.",
    ),
    Change(
      "iteration_count",
      3,
      "One pass sends the average of a few good samples straight away; "
      "re-sampling about that average twice more lets the plan settle "
      "toward what the critics prefer before anything is sent. Nav2 "
      "parameter (default 1).",
    ),
  ),
  "legacy": (
    Change(
      "gap_quantile",
      0.5,
      "FailBench's earlier planner (failbench/mppi/planner.py, which moved "
      "off hazards) weighed samples by their cost relative to the batch's "
      "gap to the median, not on Nav2's absolute scale.",
    ),
    Change("temperature", 0.25, "The earlier planner's temperature."),
    Change(
      "offset_m",
      (-0.4, -0.2, 0.2, 0.4),
      "The earlier planner always offered the warm start shifted sideways "
      "by these (its OFFSET_M), so keeping further from one side was a plan "
      "it could find.",
    ),
    Change(
      "noise_corr",
      (0.9, 0.9, 0.9),
      "The earlier planner's noise was smooth (knots every 1 s), so a "
      "sample held one course instead of averaging it out.",
    ),
    Change(
      "critics.path_align.weight",
      3.0,
      "The earlier planner charged 0.5 per m^2 off the lane (20 cm: 0.02); "
      "Nav2's path align at 14 makes any sideways course expensive.",
    ),
  ),
}


def _set(cfg: Any, path: list[str], value: Any) -> Any:
  head, *rest = path
  if head not in {f.name for f in fields(cfg)}:
    raise AttributeError(f"{type(cfg).__name__} has no field {head!r}")
  inner = _set(getattr(cfg, head), rest, value) if rest else value
  return replace(cfg, **{head: inner})


def apply_changes(cfg: Nav2MppiCfg, changes: tuple[Change, ...]) -> Nav2MppiCfg:
  for c in changes:
    cfg = _set(cfg, c.param.split("."), c.value)
  return cfg


def _diff(stock: Any, cfg: Any, prefix: str) -> list[dict]:
  out = []
  for f in fields(cfg):
    a, b = getattr(stock, f.name), getattr(cfg, f.name)
    name = prefix + f.name
    if is_dataclass(b):
      out += _diff(a, b, name + ".")
    elif a != b:
      out.append({"param": name, "stock": a, "value": b})
  return out


def changes_from_stock(cfg: Nav2MppiCfg) -> list[dict]:
  """Every field of ``cfg`` that differs from ``Nav2MppiCfg()``, in field
  order: ``{"param", "stock", "value"}``."""
  return _diff(Nav2MppiCfg(), cfg, "")
