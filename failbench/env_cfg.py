"""The FailBench data-generation env: a Go2 HIM task with kinodynamic faults.

Built on the HIM task's play config (no pushes or observation noise) with
its domain randomization removed, so fault labels are relative to the
nominal robot. Adds ``inject_kinodynamic_fault``, the knot command term,
heat_bench's whole-body contact sensor and a trunk-centred, yaw-aligned
elevation map, strips every termination, and picks a heat_bench fault
terrain (``FAULT_TERRAINS``).
"""

from __future__ import annotations

import go2_him.him.tasks  # noqa: F401  (registers the Go2 HIM tasks)
from heat_bench.envs_mjlab.fault_impact import fault_elevation_sensor_cfg
from heat_bench.envs_mjlab.go2_eval_env_cfg import use_fault_terrain
from heat_bench.envs_mjlab.joint_fault_injection import fault_contact_sensor_cfg

from failbench.commands import KnotVelocityCommandCfg
from failbench.faults import FAULT_TERM, FaultMixture, inject_kinodynamic_fault
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

DEFAULT_TASK = "Go2-HIM-Rough-Original"
RANDOMIZATION_EVENTS = ("payload", "base_com", "pd_gains", "foot_friction")

MAP_CELLS = 64
MAP_RESOLUTION = 0.05
MAP_SIZE = (MAP_CELLS - 1) * MAP_RESOLUTION
"""Side passed to the ray grid: its inclusive arange gives ``MAP_CELLS``."""


def failbench_env_cfg(
  task: str = DEFAULT_TASK,
  terrain: str = "rough",
  terrain_seed: int | None = None,
  fault_time_s: tuple[float, float] = (3.0, 5.0),
  pre_fault_ranges: tuple[tuple[float, float], ...] = (
    (0.3, 1.0),
    (-0.3, 0.3),
    (-0.5, 0.5),
  ),
  post_fault_ranges: tuple[tuple[float, float], ...] = (
    (-1.0, 1.5),
    (-1.0, 1.0),
    (-1.0, 1.0),
  ),
  num_knots: int = 7,
  knot_s: float = 1.0,
  style_probs: tuple[float, float, float] = (0.3, 0.2, 0.5),
  mixture: FaultMixture | None = None,
) -> ManagerBasedRlEnvCfg:
  cfg = load_env_cfg(task, play=True)
  if "actuator_health" in cfg.events:
    raise ValueError("FailBench faults write the force range themselves.")
  for name in RANDOMIZATION_EVENTS:
    cfg.events.pop(name, None)
  # Trials start from rest.
  if "reset_base" in cfg.events:
    cfg.events["reset_base"].params["velocity_range"] = {}
  cfg.events[FAULT_TERM] = EventTermCfg(
    func=inject_kinodynamic_fault,
    mode="step",
    params={
      "fault_time_s": tuple(fault_time_s),
      "mixture": mixture or FaultMixture(),
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )

  def ranges(r: tuple[tuple[float, float], ...]) -> UniformVelocityCommandCfg.Ranges:
    return UniformVelocityCommandCfg.Ranges(
      lin_vel_x=r[0], lin_vel_y=r[1], ang_vel_z=r[2]
    )

  cfg.commands["twist"] = KnotVelocityCommandCfg(
    entity_name="robot",
    resampling_time_range=(1e9, 1e9),
    heading_command=False,
    rel_heading_envs=0.0,
    debug_vis=True,
    ranges=ranges(pre_fault_ranges),
    post_fault_ranges=ranges(post_fault_ranges),
    num_knots=num_knots,
    knot_s=knot_s,
    style_probs=style_probs,
  )

  cfg.scene.sensors = (cfg.scene.sensors or ()) + (
    fault_contact_sensor_cfg(),
    fault_elevation_sensor_cfg(size=(MAP_SIZE, MAP_SIZE), resolution=MAP_RESOLUTION),
  )
  cfg.terminations = {}
  cfg.episode_length_s = int(1e9)
  cfg.curriculum = {}
  use_fault_terrain(cfg, terrain, terrain_seed)
  return cfg
