"""The FailBench data-generation env: a Go2 HIM task with joint faults.

Built on the HIM task's play config (no pushes or observation noise) with
its domain randomization removed, so fault labels are relative to the
nominal robot. Adds a per-trial goal (``failbench/goals.py``) the robot
spawns facing, ``inject_kinodynamic_fault``, the goal-directed command term
with its post-fault segment plan (``failbench/commands.py``), heat_bench's whole-body contact sensor and a
trunk-centred, yaw-aligned elevation map, strips every termination, and
picks a heat_bench fault terrain (``FAULT_TERRAINS``).

On the stair presets (pyramids: ``stairs`` spawns on the top platform,
``stairs_up`` in a pit) the fault fires by position, somewhere between just
before the first step and the last one (``STAIR_FIRE_RADIUS``), so nearly
every stair trial fails on the flight; ``stairs`` also loses its flat
column. Rough and course trials fault at a sampled time.
"""

from __future__ import annotations

from dataclasses import replace

import go2_him.him.tasks  # noqa: F401  (registers the Go2 HIM tasks)
from heat_bench.envs_mjlab.fault_impact import fault_elevation_sensor_cfg
from heat_bench.envs_mjlab.go2_eval_env_cfg import use_fault_terrain
from heat_bench.envs_mjlab.joint_fault_injection import fault_contact_sensor_cfg

from failbench.commands import GOAL_TERM, GoalVelocityCommandCfg
from failbench.faults import FAULT_TERM, FaultMixture, inject_kinodynamic_fault
from failbench.goals import GoalMode, sample_goal
from failbench.plans import PlanCfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import pyramid_stairs, pyramid_stairs_inv
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

DEFAULT_TASK = "Go2-HIM-Rough-Original"
RANDOMIZATION_EVENTS = ("payload", "base_com", "pd_gains", "foot_friction")

MAP_CELLS = 64
MAP_RESOLUTION = 0.05
MAP_SIZE = (MAP_CELLS - 1) * MAP_RESOLUTION
"""Side passed to the ray grid: its inclusive arange gives ``MAP_CELLS``."""

TALL_STAIR_STEPS = dict(
  proportion=1.0,
  step_height_range=(0.08, 0.18),
  step_width=0.25,
  platform_width=2.0,
  border_width=0.5,
)
"""Ten 0.25 m treads from 1.0 to 3.5 m out (Chebyshev) of the centre,
risers 8-18 cm down the difficulty rows: 18-36 deg flights 0.8-1.8 m tall.
The reference policy trained on risers up to 10 cm."""


def _tall_stairs(kind) -> TerrainGeneratorCfg:
  return TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=10,
    num_cols=1,
    curriculum=True,
    sub_terrains={"tall_stairs": kind(**TALL_STAIR_STEPS)},
    add_lights=True,
  )


TALL_STAIR_TERRAINS = {
  "stairs_tall": _tall_stairs(pyramid_stairs),
  "stairs_up_tall": _tall_stairs(pyramid_stairs_inv),
}
"""FailBench's own presets beside heat_bench's ``FAULT_TERRAINS``: a raised
pyramid to walk down from its top platform, and a pit to climb out of."""

GOAL_MODES: dict[str, GoalMode] = {
  "rough": "free",
  "stairs": "pyramid",
  "stairs_up": "pyramid",
  "stairs_tall": "pyramid",
  "stairs_up_tall": "pyramid",
  "course": "lane",
}
STAIR_FIRE_RADIUS = {
  "stairs": (1.6, 3.2),
  "stairs_up": (1.4, 2.9),
  "stairs_tall": (0.9, 3.3),
  "stairs_up_tall": (0.9, 3.3),
}
"""Chebyshev distance (m) from the pyramid centre at which the fault fires.
``stairs`` platforms end 1.1-1.8 m out and its steps 3.0-3.5 m;
``stairs_up``'s pit ends 1.5 m out and its steps 3.0 m; the tall flights
run from 1.0 to 3.5 m."""
STAIR_SPAWN_XY = (-0.3, 0.3)
"""Spawn jitter about the pyramid centre, so the robot starts well inside
the firing radius."""
STAIR_MIN_TIME_S = 2.5
"""Earliest stair fault: HIM's history has settled and the 2 s pre-fault
log fits."""
STAIR_TIMEOUT_S = (10.0, 10.0)
"""A stair fault fires by this time even if the robot never got there."""


def failbench_env_cfg(
  task: str = DEFAULT_TASK,
  terrain: str = "rough",
  terrain_seed: int | None = None,
  fault_time_s: tuple[float, float] = (3.0, 5.0),
  plan: PlanCfg | None = None,
  mixture: FaultMixture | None = None,
) -> ManagerBasedRlEnvCfg:
  """``fault_time_s`` applies to rough and course; stair trials fault by
  position (``STAIR_FIRE_RADIUS``)."""
  cfg = load_env_cfg(task, play=True)
  if "actuator_health" in cfg.events:
    raise ValueError("FailBench faults write the force range themselves.")
  for name in RANDOMIZATION_EVENTS:
    cfg.events.pop(name, None)
  # The play config re-picks each env's tile after reset_base has placed the
  # robot, leaving env_origins on a tile it isn't on; pick it first instead,
  # since goals and the stair fault trigger are measured from env_origins.
  if "randomize_terrain" in cfg.events:
    cfg.events = {"randomize_terrain": cfg.events.pop("randomize_terrain")} | (
      cfg.events
    )
  # Trials start from rest.
  if "reset_base" in cfg.events:
    cfg.events["reset_base"].params["velocity_range"] = {}
  if terrain in TALL_STAIR_TERRAINS:
    # As use_fault_terrain: fresh layout per seed, spawns on every row.
    assert cfg.scene.terrain is not None
    gen = TALL_STAIR_TERRAINS[terrain]
    if terrain_seed is not None:
      gen = replace(gen, seed=terrain_seed)
    cfg.scene.terrain.terrain_generator = gen
    cfg.scene.terrain.max_init_terrain_level = None
  else:
    use_fault_terrain(cfg, terrain, terrain_seed)
  if terrain in STAIR_FIRE_RADIUS:
    params = cfg.events["reset_base"].params
    params["pose_range"] = {
      **params["pose_range"],
      "x": STAIR_SPAWN_XY,
      "y": STAIR_SPAWN_XY,
    }
  if terrain == "stairs":
    gen = cfg.scene.terrain.terrain_generator  # type: ignore[union-attr]
    assert gen is not None
    subs = {k: v for k, v in gen.sub_terrains.items() if k != "flat"}
    cfg.scene.terrain.terrain_generator = replace(  # type: ignore[union-attr]
      gen, sub_terrains=subs, num_cols=len(subs)
    )

  # After reset_base (insertion order), so it can turn the spawned robot.
  cfg.events[GOAL_TERM] = EventTermCfg(
    func=sample_goal,
    mode="reset",
    params={"mode": GOAL_MODES[terrain], "asset_cfg": SceneEntityCfg("robot")},
  )
  fault_params: dict = {
    "fault_time_s": tuple(fault_time_s),
    "mixture": mixture or FaultMixture(),
    "asset_cfg": SceneEntityCfg("robot"),
  }
  if terrain in STAIR_FIRE_RADIUS:
    fault_params |= {
      "fault_time_s": STAIR_TIMEOUT_S,
      "fire_radius": STAIR_FIRE_RADIUS[terrain],
      "min_time_s": STAIR_MIN_TIME_S,
    }
  cfg.events[FAULT_TERM] = EventTermCfg(
    func=inject_kinodynamic_fault, mode="step", params=fault_params
  )

  cfg.commands["twist"] = GoalVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    heading_command=False,
    rel_heading_envs=0.0,
    debug_vis=True,
    ranges=UniformVelocityCommandCfg.Ranges(
      lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)
    ),
    plan=plan or PlanCfg(),
    # Course lanes run along +x through the goal at the lane end.
    lane_heading=0.0 if GOAL_MODES[terrain] == "lane" else None,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    fault_contact_sensor_cfg(),
    fault_elevation_sensor_cfg(size=(MAP_SIZE, MAP_SIZE), resolution=MAP_RESOLUTION),
  )
  cfg.terminations = {}
  cfg.episode_length_s = int(1e9)
  cfg.curriculum = {}
  return cfg
