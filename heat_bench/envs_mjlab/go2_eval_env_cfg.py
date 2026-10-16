"""Passive thermal/battery evaluation config, built on the Go1 rough-terrain task.

mjlab has no Go2 MJCF asset yet, so this config simulates the existing Go1
robot body while modeling Go2's electro-thermal hardware constants (see
``heat_bench/configs/go2_eval_config.yaml``). It adds a read-only "thermal"
observation group and episode-summary metrics on top of the unmodified Go1
rough-terrain task; no reward or action terms are touched.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import torch
import yaml

from heat_bench.envs_mjlab.actuator_health import (
  apply_actuator_health,
  enabled_gear_break,
  scripted_joint_fault,
)
from heat_bench.envs_mjlab.eval_observations import (
  ThermalEnergyObservation,
  battery_capacity_loss_pct,
  battery_cumulative_energy_wh,
  battery_energy_wh_step,
  battery_soc,
  thermal_max_joint_temp,
  thermal_mean_joint_temp,
)
from heat_bench.envs_mjlab.fault_impact import fault_elevation_sensor_cfg
from heat_bench.envs_mjlab.gear_load import GearLoadObservation, gear_load_peak_ratio
from heat_bench.envs_mjlab.joint_fault_injection import (
  fault_contact_sensor_cfg,
  inject_joint_fault,
  parse_fault_modes,
)
from heat_bench.envs_mjlab.payload import DEFAULT_OFFSET_RANGE, attach_payload
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.metrics_manager import MetricsTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.config.go1.env_cfgs import unitree_go1_rough_env_cfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains.config import (
  ROUGH_TERRAINS_CFG,
  STAIRS_TERRAINS_CFG,
  pyramid_stairs_inv,
)
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg

STAIRS_UP_TERRAINS_CFG = TerrainGeneratorCfg(
  size=(8.0, 8.0),
  border_width=10.0,
  num_rows=10,
  num_cols=1,
  curriculum=True,
  sub_terrains={
    # A pit: the robot spawns on the low 3 m centre platform, so walking out
    # in any direction climbs 5 steps of 0.3 m tread (1.5-3.0 m from the
    # centre, Chebyshev distance). Riser height rises with the difficulty
    # row; the reference policy trained on risers up to 0.10 m, so the top
    # rows (to 0.14 m) are beyond what it has seen.
    "stairs_up": pyramid_stairs_inv(
      proportion=1.0,
      step_height_range=(0.02, 0.14),
      step_width=0.3,
      platform_width=3.0,
      border_width=1.0,
    ),
  },
  add_lights=True,
)

FAULT_TERRAINS = {
  "rough": ROUGH_TERRAINS_CFG,
  "stairs": STAIRS_TERRAINS_CFG,
  "stairs_up": STAIRS_UP_TERRAINS_CFG,
}
"""Terrain presets for fault trials, generated in curriculum layout so each
tile's row is its difficulty and its column its sub-terrain type."""

CONFIG_PATH = Path(__file__).parents[1] / "configs" / "go2_eval_config.yaml"


def load_heat_bench_config() -> dict:
  with open(CONFIG_PATH) as f:
    return yaml.safe_load(f)


def distance_traveled(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  """Per-step distance delta. Reduced with ``sum`` for the episode-end total."""
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_lin_vel_b[:, :2].norm(dim=-1) * env.step_dt


class CumulativeDistanceTraveled:
  """Running distance total since the last reset, for live monitoring.

  A plain ``sum``-reduce metric (like ``distance_traveled`` above) only
  exposes its running total at episode end, not per step -- this keeps its
  own persistent buffer so an interactive viewer can plot the trend live."""

  def __init__(self, cfg: MetricsTermCfg, env):
    del cfg
    self._cum_distance = torch.zeros(env.num_envs, device=env.device)

  def reset(self, env_ids) -> None:
    self._cum_distance[env_ids] = 0.0

  def __call__(self, env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    step_distance = asset.data.root_link_lin_vel_b[:, :2].norm(dim=-1) * env.step_dt
    self._cum_distance += step_distance
    return self._cum_distance


def go2_eval_env_cfg(
  play: bool = True, battery_model: str | None = None, gear_break: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create the passive Go2-hardware-modeled thermal/battery eval config.

  Args:
    play: Forwarded to unitree_go1_rough_env_cfg.
    battery_model: If given, overrides the yaml's `battery.model` ("rint"
      or "rint_soc_aging") -- lets scripts A/B compare battery models
      without editing configs/go2_eval_config.yaml.
    gear_break: Turn on the yaml's `actuator_health.gear_break` gearbox
      overload break (off by default). Requires gear_load.
  """
  cfg = unitree_go1_rough_env_cfg(play=play)
  hb_cfg = load_heat_bench_config()
  if battery_model is not None:
    hb_cfg["battery"]["model"] = battery_model
  if gear_break:
    hb_cfg["actuator_health"]["gear_break"]["enabled"] = True
  robot_asset_cfg = SceneEntityCfg("robot", joint_names=(".*",))

  cfg.observations["thermal"] = ObservationGroupCfg(
    terms={
      "thermal_energy": ObservationTermCfg(
        func=ThermalEnergyObservation,
        params={"asset_cfg": robot_asset_cfg, "config": hb_cfg},
      ),
    },
    concatenate_terms=True,
    enable_corruption=False,
  )

  cfg.metrics["thermal_max_joint_temp"] = MetricsTermCfg(
    func=thermal_max_joint_temp,
    reduce="max",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["thermal_mean_joint_temp"] = MetricsTermCfg(
    func=thermal_mean_joint_temp,
    reduce="mean",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["battery_energy_wh"] = MetricsTermCfg(
    func=battery_energy_wh_step,
    reduce="sum",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["battery_final_soc"] = MetricsTermCfg(
    func=battery_soc,
    reduce="last",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["distance_traveled"] = MetricsTermCfg(
    func=distance_traveled,
    reduce="sum",
    params={"asset_cfg": robot_asset_cfg},
  )
  cfg.metrics["battery_cumulative_energy_wh"] = MetricsTermCfg(
    func=battery_cumulative_energy_wh,
    reduce="last",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )
  cfg.metrics["distance_traveled_cumulative"] = MetricsTermCfg(
    func=CumulativeDistanceTraveled,
    reduce="last",
    params={"asset_cfg": robot_asset_cfg},
  )
  cfg.metrics["battery_capacity_loss_pct"] = MetricsTermCfg(
    func=battery_capacity_loss_pct,
    reduce="last",
    params={"obs_group": "thermal", "obs_term": "thermal_energy"},
  )

  if hb_cfg.get("gear_load", {}).get("enabled", False):
    # Separate group so the thermal observation vector stays unchanged.
    # Registered after "thermal", so its sim.step wrap chains onto the
    # thermal term's.
    cfg.observations["gear_load"] = ObservationGroupCfg(
      terms={
        "gear_load": ObservationTermCfg(
          func=GearLoadObservation, params={"asset_cfg": robot_asset_cfg}
        ),
      },
      concatenate_terms=True,
      enable_corruption=False,
    )
    gear_params = {"obs_group": "gear_load", "obs_term": "gear_load"}
    cfg.metrics["gear_load_max_ratio"] = MetricsTermCfg(
      func=gear_load_peak_ratio, reduce="max", params=gear_params
    )
    cfg.metrics["gear_load_mean_peak_ratio"] = MetricsTermCfg(
      func=gear_load_peak_ratio, reduce="mean", params=gear_params
    )

  health_cfg = hb_cfg.get("actuator_health", {})
  if health_cfg.get("enabled", False):
    health_params = {
      "asset_cfg": robot_asset_cfg,
      "obs_group": "thermal",
      "obs_term": "thermal_energy",
      "config": hb_cfg,
    }
    if enabled_gear_break(hb_cfg) is not None:
      if "gear_load" not in cfg.observations:
        raise ValueError("actuator_health.gear_break requires gear_load.enabled.")
      health_params["gear_obs_group"] = "gear_load"
      health_params["gear_obs_term"] = "gear_load"
    cfg.events["actuator_health"] = EventTermCfg(
      func=apply_actuator_health, mode="step", params=health_params
    )

  payload_cfg = hb_cfg.get("payload", {})
  if payload_cfg.get("enabled", False):
    cfg.events["payload_mass"] = EventTermCfg(
      func=dr.pseudo_inertia,
      mode=payload_cfg.get("mode", "reset"),
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
        "alpha_range": tuple(payload_cfg.get("alpha_range", (0.0, 0.0))),
      },
    )

  return cfg


def add_push_disturbance(cfg: ManagerBasedRlEnvCfg) -> None:
  """Re-add the standard velocity-kick push event, mutating ``cfg`` in place.

  ``unitree_go1_rough_env_cfg(play=True)`` strips "push_robot"
  (src/mjlab/tasks/velocity/config/go1/env_cfgs.py) so interactive viewing
  is deterministic by default. Call this after ``go2_eval_env_cfg(play=True)``
  to opt back into it for watching how the tracked thermal/battery/torque
  state reacts to an external disturbance -- same params mjlab's own
  training config uses (src/mjlab/tasks/velocity/velocity_env_cfg.py).
  """
  cfg.events["push_robot"] = EventTermCfg(
    func=envs_mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.4, 0.4),
        "roll": (-0.52, 0.52),
        "pitch": (-0.52, 0.52),
        "yaw": (-0.78, 0.78),
      },
    },
  )


def add_impulse_disturbance(
  cfg: ManagerBasedRlEnvCfg, hb_cfg: dict | None = None
) -> None:
  """Add a real force-based push/kick event, mutating ``cfg`` in place.

  Unlike ``add_push_disturbance`` (an instantaneous, mass-independent qvel
  overwrite -- no force is ever computed), this uses
  ``mjlab.envs.mdp.apply_body_impulse``: a real force+torque wrench written
  to ``xfrc_applied`` and held for a sampled duration, so it respects the
  robot's own mass/inertia/contacts and renders as a visible arrow in the
  Viser/native debug-vis overlay (``apply_body_impulse.debug_vis``,
  auto-discovered by ``EventManager.debug_vis``, no extra viewer code
  needed). Force/torque/duration/cooldown magnitudes come from
  ``configs/go2_eval_config.yaml``'s ``impulse_disturbance`` section.
  """
  hb_cfg = hb_cfg if hb_cfg is not None else load_heat_bench_config()
  impulse_cfg = hb_cfg["impulse_disturbance"]
  cfg.events["impulse_disturbance"] = EventTermCfg(
    func=envs_mdp.apply_body_impulse,
    mode="step",
    params={
      "force_range": tuple(impulse_cfg["force_range"]),
      "torque_range": tuple(impulse_cfg["torque_range"]),
      "duration_s": tuple(impulse_cfg["duration_s"]),
      "cooldown_s": tuple(impulse_cfg["cooldown_s"]),
      "asset_cfg": SceneEntityCfg("robot", body_names=tuple(impulse_cfg["body_names"])),
    },
  )


def add_scripted_joint_fault(
  cfg: ManagerBasedRlEnvCfg,
  joint_names: tuple[str, ...],
  derate_start_s: float = 3.0,
  derate_end_s: float = 6.0,
  derate_floor: float = 0.3,
  dead_at_s: float = 10.0,
) -> None:
  """Script joints through healthy -> derated -> dead, mutating ``cfg``.

  Demo of the ``actuator_health`` write path (see ``scripted_joint_fault``).
  All selected joints share one schedule; every other joint stays at full
  health. Requires ``actuator_health`` to be enabled. Re-inserts the health
  term after the fault event so the scripted derate applies in the same
  control step.
  """
  if "actuator_health" not in cfg.events:
    raise ValueError("add_scripted_joint_fault requires actuator_health enabled.")
  health_event = cfg.events.pop("actuator_health")
  cfg.events["joint_fault"] = EventTermCfg(
    func=scripted_joint_fault,
    mode="step",
    params={
      "joint_names": tuple(joint_names),
      "health_term": "actuator_health",
      "derate_start_s": derate_start_s,
      "derate_end_s": derate_end_s,
      "derate_floor": derate_floor,
      "dead_at_s": dead_at_s,
    },
  )
  cfg.events["actuator_health"] = health_event


def use_fault_terrain(cfg: ManagerBasedRlEnvCfg, terrain: str) -> None:
  """Swap in a ``FAULT_TERRAINS`` preset, mutating ``cfg`` in place."""
  if terrain not in FAULT_TERRAINS:
    raise ValueError(f"Unknown terrain {terrain!r}; expected {list(FAULT_TERRAINS)}.")
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_generator = replace(
    FAULT_TERRAINS[terrain], curriculum=True, border_width=10.0
  )
  # Fault trials never reset, so the initial spawn is the only draw: cover
  # every difficulty row, not just the training curriculum's starting levels.
  cfg.scene.terrain.max_init_terrain_level = None


def add_joint_fault_injection(
  cfg: ManagerBasedRlEnvCfg,
  joint_names: tuple[str, ...],
  modes: tuple[str, ...] = ("limit", "lock", "free"),
  fault_time_s: tuple[float, float] = (4.0, 4.0),
  num_fault_joints: int = 1,
  limit_half_width_rad: float = 0.15,
  command_vel: tuple[float, float, float] | None = (1.0, 0.0, 0.0),
  command_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
  | None = None,
  terrain: str | None = None,
  elevation_map: bool = False,
) -> None:
  """Fail joints mid-run without resetting, mutating ``cfg`` in place.

  Adds ``inject_joint_fault`` (before ``actuator_health``, like
  ``add_scripted_joint_fault``) and the whole-body ``fault_contacts``
  sensor. Strips every termination and makes episodes effectively infinite,
  so a fall -- e.g. a thigh touching the ground, normally
  ``illegal_contact`` -- doesn't reset the env and the post-fault contacts
  can be watched or recorded. ``command_vel`` pins the twist command to
  (lin_vel_x, lin_vel_y, ang_vel_z) so runs compare; ``None`` keeps the
  task's random commands. ``command_ranges`` instead samples one twist per
  env at reset from ((vx lo, hi), (vy lo, hi), (wz lo, hi)) and holds it
  for the whole run, so velocity errors are measured against a constant
  command; the two are mutually exclusive. ``terrain`` picks a
  ``FAULT_TERRAINS`` preset (``None`` keeps the task's random tiles).
  ``elevation_map`` adds the ``fault_elevation_map`` raycast grid (see
  ``fault_impact.py``). The gearbox break is turned off so only the
  scripted fault fires.
  """
  if command_vel is not None and command_ranges is not None:
    raise ValueError("Pass command_vel or command_ranges, not both.")
  if terrain is not None and terrain not in FAULT_TERRAINS:
    raise ValueError(f"Unknown terrain {terrain!r}; expected {list(FAULT_TERRAINS)}.")
  if "actuator_health" not in cfg.events:
    raise ValueError("add_joint_fault_injection requires actuator_health enabled.")
  parse_fault_modes(modes)  # Fail fast on a typo.
  health_event = cfg.events.pop("actuator_health")
  health_cfg = health_event.params["config"]["actuator_health"]
  if health_cfg.get("gear_break") is not None:
    health_cfg["gear_break"]["enabled"] = False
  health_event.params.pop("gear_obs_group", None)
  health_event.params.pop("gear_obs_term", None)
  cfg.events["joint_fault_injection"] = EventTermCfg(
    func=inject_joint_fault,
    mode="step",
    params={
      "joint_names": tuple(joint_names),
      "modes": tuple(modes),
      "fault_time_s": tuple(fault_time_s),
      "health_term": "actuator_health",
      "num_fault_joints": num_fault_joints,
      "limit_half_width_rad": limit_half_width_rad,
      "asset_cfg": SceneEntityCfg("robot"),
    },
  )
  cfg.events["actuator_health"] = health_event

  sensors = (fault_contact_sensor_cfg(),)
  if elevation_map:
    sensors += (fault_elevation_sensor_cfg(),)
  cfg.scene.sensors = (cfg.scene.sensors or ()) + sensors
  cfg.terminations = {}
  cfg.episode_length_s = int(1e9)
  cfg.curriculum = {}

  if terrain is not None:
    use_fault_terrain(cfg, terrain)

  if command_vel is not None:
    vx, vy, wz = command_vel
    command_ranges = ((vx, vx), (vy, vy), (wz, wz))
  if command_ranges is not None:
    twist = cfg.commands["twist"]
    assert isinstance(twist, UniformVelocityCommandCfg)
    rx, ry, rz = command_ranges
    twist.heading_command = False
    twist.rel_standing_envs = 0.0
    twist.rel_heading_envs = 0.0
    twist.rel_world_envs = 0.0
    twist.rel_forward_envs = 0.0
    twist.init_velocity_prob = 0.0
    # Sampled at reset only; never resampled mid-run.
    twist.resampling_time_range = (1e9, 1e9)
    twist.ranges = UniformVelocityCommandCfg.Ranges(
      lin_vel_x=rx, lin_vel_y=ry, ang_vel_z=rz
    )


def add_box_payload(
  cfg: ManagerBasedRlEnvCfg,
  mass_kg_range: tuple[float, float],
  offset_range: tuple[tuple[float, float], ...] = DEFAULT_OFFSET_RANGE,
) -> None:
  """Strap a box payload of sampled mass onto the trunk, mutating ``cfg``.

  New sample at every reset (see ``attach_payload``). The default mount sits
  on top of the trunk with a few cm of fore-aft/lateral jitter, so the COM
  rises and shifts like a real backpack's. Go1 weighs ~12.7 kg, so 5 kg is
  ~40% extra. The reference policy was trained without a payload; pair with
  fault mode ``none`` to see what the payload alone does.
  """
  cfg.events["box_payload"] = EventTermCfg(
    func=attach_payload,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("robot", body_names=("trunk",)),
      "mass_kg_range": tuple(mass_kg_range),
      "offset_range": tuple(tuple(r) for r in offset_range),
    },
  )
