"""Run the MPPI planner in closed loop on FailBench trials.

The sim stands in for the robot, and the planner (``failbench/mppi``) only
gets what the Go2 itself would have:

- HIM's velocity estimate and latent, the IMU and the joint encoders;
- an elevation map: in sim a trunk-centred, yaw-aligned ray-cast grid
  (``map_cells`` at 0.05 m), masked by what the head LiDAR saw over the
  last 1.5 s (``failbench.data.build.lidar_known``) and noised as in
  training. On the robot an elevation mapper replaces this. The impact model
  sees the centre 64x64 crop, the footprint sweep all of it
  (``sweep_map="oracle"`` gives the sweep the full clean map instead, for
  debugging);
- a pose estimate (odometry) with random-walk drift (``odom_*_std``), used
  for the goal, the lane and the map fusion;
- the fault, known once it fires (``--no-fault-aware``: never told, for the
  fault-blind comparison).

Each env stands for ``warmup_s`` (the command term's), walks toward its
goal with the data's goal follower until ``start_s`` (the history is then
full), and from there MPPI drives, re-planning every 0.1 s with a one-cycle
delay. A planner command takes effect one control step after it is written
(the command term updates before the observation in
``ManagerBasedRlEnv.step``). An env stops (zero command) once it reaches its
goal or falls.

Writes ``<output>.npz`` (per-step robot state and per-cycle plans, for
``failbench/scripts/play_mppi.py``) and ``<output>.json`` (per-trial and
per-family results).
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from go2_him.him.env_cfg import LIN_VEL_SCALE
from heat_bench.envs_mjlab.fault_impact import (
  TRUNK_GROUPS,
  body_group,
  body_part,
  fault_elevation_sensor_cfg,
  heading,
  tilt_angle,
)
from heat_bench.envs_mjlab.joint_fault_injection import (
  FAULT_CONTACT_SENSOR,
  contacts_in_base_frame,
)
from heat_bench.scripts.policy_loading import load_runner

from failbench.commands import GOAL_TERM, GoalVelocityCommand
from failbench.data.build import _yaw_rotate_quat, fallen, lidar_known
from failbench.env_cfg import (
  DEFAULT_TASK,
  GOAL_MODES,
  MAP_CELLS,
  MAP_RESOLUTION,
  failbench_env_cfg,
)
from failbench.faults import (
  FAULT_TERM,
  NOMINAL,
  FaultFamily,
  FaultMixture,
  encode_fault,
  faulted_joint,
  inject_kinodynamic_fault,
)
from failbench.goals import sample_goal
from failbench.mppi.costmap import distance_field, obstacle_cells
from failbench.mppi.inputs import HistoryBuffer, crop_center
from failbench.mppi.models import DynamicsModel, ImpactDynamics, ZeroDriftModel
from failbench.mppi.planner import Mppi, MppiCfg
from failbench.scripts.generate import policy_estimates
from failbench.sensing import LidarCfg, NoiseCfg, corrupt_map
from failbench.severity import DROP_SEVERITY, geom_severity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import RayCastSensor
from mjlab.sensor.raycast_sensor import RayCastData
from mjlab.utils.lab_api.math import wrap_to_pi
from mjlab.utils.torch import configure_torch_backends

MPPI_MAP_SENSOR = "mppi_map"
CLOSE_CALL_M = 0.3
FELL_OFF_M = 0.35
"""On ``cliff``, the trunk this far below its starting height has gone over
the ledge (the corridor is flat; the pit is 0.6 m down)."""
"""A trunk centre this close to an obstacle (about 0.1 m of body clearance
beside the trunk) counts as a close call."""
FAMILIES = {
  "mixture": FaultMixture(),
  "nominal": FaultMixture(probs=(1.0, 0.0, 0.0)),
  "lock": FaultMixture(probs=(0.0, 1.0, 0.0)),
  "weak": FaultMixture(probs=(0.0, 0.0, 1.0)),
  "free": FaultMixture(probs=(0.0, 0.0, 1.0), limp_prob=1.0),
}
"""Fault mixture per ``--family``; ``free`` is WEAK at zero torque only (a
limp joint, heat_bench's FREE)."""


@dataclass(frozen=True)
class RunMppiConfig:
  checkpoint_file: str
  """The frozen HIM policy."""
  output: str
  """Output path without extension (``.npz`` and ``.json`` are appended, so
  dots in the name are kept)."""
  controller: Literal["mppi", "follower"] = "mppi"
  """``follower``: no planner; the data's goal follower (turn toward the
  goal, walk at the trial's cruise speed) drives from ``start_s``, steering
  on the same odometry. The baseline MPPI has to beat."""
  model: str | None = None
  """Impact model checkpoint (``best.pt``); none = the zero-drift stub."""
  replan_every_s: float = 0.1
  """Replan this often; in between, the robot plays the current plan out
  open loop (goal and fall checks still run every 0.1 s). Longer commitments
  show how good each plan is, since feedback can't fix it as fast."""
  fault_aware: bool = True
  """Tell the dynamics model the fault once it fires; off, it is always given
  the healthy vector (a planner that ignores the fault, same model)."""
  task: str = DEFAULT_TASK
  terrain: str = "course"
  terrain_seed: int | None = None
  num_envs: int = 16
  duration_s: float = 30.0
  fault_time_s: tuple[float, float] = (4.0, 8.0)
  family: Literal["mixture", "nominal", "lock", "weak", "free"] = "mixture"
  start_s: float = 2.5
  """MPPI takes over from the goal follower (history and fused poses full)."""
  map_cells: int = 160
  map_log_every: int = 10
  explain: bool = False
  """On the map-log cycles, also log every candidate plan scored with both
  the fault the planner is told and the other choice (healthy vs the true
  fault), and the plan MPPI would pick with each, for
  ``failbench/scripts/explain_mppi.py``."""
  """Log the planning map and its obstacle cells every n cycles (0: never)."""
  """Planning map side (cells, 0.05 m): 8 m, about a 5 s plan at 0.8 m/s."""
  sweep_map: Literal["lidar", "oracle"] = "lidar"
  odom_xy_std: float = 0.02
  """Odometry position random walk (m per sqrt(s))."""
  odom_yaw_std: float = 0.005
  """Odometry heading random walk (rad per sqrt(s))."""
  arrive_radius: float = 0.5
  waypoint_radius: float = 1.0
  """On waypoint terrains (``l_course``, ``doors``), move on to the next waypoint
  within this distance of the current one (or once past it)."""
  mppi: MppiCfg = field(default_factory=MppiCfg)
  lidar: LidarCfg = field(default_factory=LidarCfg)
  noise: NoiseCfg = field(default_factory=NoiseCfg)
  fuse_back_s: tuple[float, ...] = (0.5, 1.0, 1.5)
  history_s: float = 2.0
  lidar_chunk: int = 4
  device: str | None = None
  seed: int = 0


def _touched_severity(
  touch: torch.Tensor,
  contact_w: torch.Tensor,
  rays: RayCastData,
  geom_sev: torch.Tensor,
  reach_m: float = 0.08,
) -> torch.Tensor:
  """Worst severity per env ``[n]`` of what its touching contacts
  (``touch [n, N]`` at ``contact_w [n, N, 3]``) are on: the most severe
  geom the map's rays hit within ``reach_m`` (xy) of each contact."""
  out = torch.zeros(touch.shape[0], device=touch.device)
  sev = geom_sev[rays.hit_geom_id.long()]  # [n, R]
  for b in touch.any(-1).nonzero()[:, 0].tolist():
    pts = contact_w[b, touch[b], :2]
    near = torch.cdist(pts, rays.hit_pos_w[b, :, :2]) < reach_m  # [M, R]
    out[b] = torch.where(near, sev[b], 0.0).amax()
  return out


def _yaw_quat_mul(quat: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
  return _yaw_rotate_quat(quat[:, None], yaw)[:, 0]


def _to_yaw_frame_2d(xy: torch.Tensor, origin: torch.Tensor, yaw: torch.Tensor):
  d = xy - origin
  c, s = torch.cos(yaw), torch.sin(yaw)
  return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], -1)


def _to_world(path: torch.Tensor, pos: torch.Tensor, yaw: torch.Tensor):
  """``path [n, T, 3]`` in each robot's yaw frame -> world (x, y)."""
  c, s = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
  x, y = path[..., 0], path[..., 1]
  return torch.stack(
    [pos[:, None, 0] + c * x - s * y, pos[:, None, 1] + s * x + c * y], -1
  )


def run(cfg: RunMppiConfig) -> dict:
  configure_torch_backends()
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  torch.manual_seed(cfg.seed)
  mixture = FAMILIES[cfg.family]
  env_cfg = failbench_env_cfg(
    cfg.task,
    cfg.terrain,
    cfg.terrain_seed,
    fault_time_s=cfg.fault_time_s,
    mixture=mixture,
  )
  env_cfg.scene.num_envs = cfg.num_envs
  if (cfg.map_cells - MAP_CELLS) % 2:
    raise ValueError("map_cells - 64 must be even to share the model's grid.")
  side = (cfg.map_cells - 1) * MAP_RESOLUTION
  big = fault_elevation_sensor_cfg(size=(side, side), resolution=MAP_RESOLUTION)
  big.name = MPPI_MAP_SENSOR
  env_cfg.scene.sensors = (env_cfg.scene.sensors or ()) + (big,)
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped, runner = load_runner(
    env, device, checkpoint_file=cfg.checkpoint_file, task_id=cfg.task
  )
  policy = runner.get_inference_policy(device=device)
  actor = runner.alg.get_policy()
  model: DynamicsModel = (
    ImpactDynamics.from_checkpoint(cfg.model, device) if cfg.model else ZeroDriftModel()
  )
  print(f"[INFO] Dynamics: {cfg.model or 'zero-drift stub'} (map: {model.map_mode})")

  n, step_dt = env.num_envs, env.step_dt
  robot = env.scene["robot"]
  fault = env.event_manager.get_term_cfg(FAULT_TERM).func
  assert isinstance(fault, inject_kinodynamic_fault)
  command = env.command_manager.get_term("twist")
  goal_term = env.event_manager.get_term_cfg(GOAL_TERM).func
  assert isinstance(goal_term, sample_goal)
  assert isinstance(command, GoalVelocityCommand)
  elev = env.scene[MPPI_MAP_SENSOR]
  geom_sev = geom_severity(env.sim.mj_model, device=device)
  assert isinstance(elev, RayCastSensor)
  sensor = env.scene[FAULT_CONTACT_SENSOR]
  cells = cfg.map_cells
  mid = slice(cells // 2 - 2, cells // 2 + 2)
  geom_part = [body_group(body_part(g)) for g in sensor.primary_names]
  slot_trunk = torch.tensor(
    [g in TRUNK_GROUPS for g in geom_part], device=device
  ).repeat_interleave(sensor.cfg.num_slots)
  slot_body = torch.tensor(
    [g != "foot" for g in geom_part], device=device
  ).repeat_interleave(sensor.cfg.num_slots)

  planner = Mppi(cfg.mppi, model, n, device)
  r = planner.substeps
  hist = HistoryBuffer(n, round(cfg.history_s / cfg.mppi.sampler.step_s), 36, device)
  back = [round(b / step_dt) for b in cfg.fuse_back_s]
  ring_len = max(back) + 1
  ring_pos = torch.zeros(n, ring_len, 3, device=device)
  ring_quat = torch.zeros(n, ring_len, 4, device=device)
  odom_xy = torch.zeros(n, 2, device=device)
  odom_yaw = torch.zeros(n, device=device)
  nominal = torch.tensor(NOMINAL, device=device).expand(n, len(fault.joint_names), 2)
  lane = GOAL_MODES[cfg.terrain] == "lane"

  num_steps = round(cfg.duration_s / step_dt)
  num_cycles = num_steps // r + 1
  T = cfg.mppi.sampler.steps
  log: dict[str, np.ndarray] = {
    k: np.zeros((n, num_steps, *shape), np.float32)
    for k, shape in {
      "base_pos_w": (3,),
      "base_quat_w": (4,),
      "joint_pos": (len(fault.joint_names),),
      "cmd": (3,),
      "lin_vel_b": (3,),
      "ang_vel_b": (3,),
      "tilt": (),
      "base_height": (),
    }.items()
  }
  log["trunk_contact"] = np.zeros((n, num_steps), bool)
  log["faulted"] = np.zeros((n, num_steps), bool)
  cyc: dict[str, np.ndarray] = {
    k: np.full((n, num_cycles, *shape), np.nan, np.float32)
    for k, shape in {
      "path_w": (T, 2),
      "pred_w": (T, 2),
      "pred_std": (T, 3),
      "pred_yaw_w": (T,),
      "pred_p_fallen": (T,),
      "pred_p_collision": (T,),
      "p_fall_soon": (),
      "drift_end": (3,),
      "origin_w": (4,),
      "p_fallen": (),
      "ess": (),
      "true_pred_w": (T, 2),
      "true_pred_std": (T, 3),
      "true_p_fallen": (),
    }.items()
  }
  # The chosen plan's predicted body-contact maps (per second, on the
  # model's 64x64 grid around the trunk at the cycle) as uint8 probability,
  # and that grid's clean terrain height, to draw them on.
  num_secs = round(T * cfg.mppi.sampler.step_s)
  contact = np.zeros((n, num_cycles, num_secs, MAP_CELLS, MAP_CELLS), np.uint8)
  contact_height = np.full((n, num_cycles, MAP_CELLS, MAP_CELLS), np.nan, np.float16)
  cyc["step"] = np.full(num_cycles, -1, np.int64)
  every = cfg.map_log_every
  num_maps = num_cycles // every + 1 if every else 0
  maps = np.full((n, num_maps, cells, cells), np.nan, np.float16)
  obstacles = np.zeros((n, num_maps, cells, cells), bool)
  ex: dict[str, np.ndarray] = {}
  if cfg.explain:
    K, sub = cfg.mppi.sampler.num_plans, T // 5
    shapes = {
      "path": ((K, sub, 2), np.float16),
      "pose": ((K, sub, 2), np.float16),
      "alt_pose": ((K, sub, 2), np.float16),
      "cost": ((K,), np.float32),
      "alt_cost": ((K,), np.float32),
      "weight": ((K,), np.float32),
      "alt_weight": ((K,), np.float32),
      "plan_path": ((T, 2), np.float32),
      "plan_pred": ((T, 2), np.float32),
      "plan_pred_alt": ((T, 2), np.float32),
      "alt_plan_path": ((T, 2), np.float32),
      "alt_plan_pred": ((T, 2), np.float32),
      "alt_plan_pred_here": ((T, 2), np.float32),
      "goal": ((2,), np.float32),
    }
    ex = {k: np.full((n, num_maps, *sh), np.nan, dt) for k, (sh, dt) in shapes.items()}
  cost_logs: dict[str, np.ndarray] = {}
  clearance = np.full((n, num_cycles), np.nan, np.float32)
  replan_k = max(1, round(cfg.replan_every_s / cfg.mppi.sampler.step_s))

  active = torch.zeros(n, dtype=torch.bool, device=device)
  reached = torch.zeros(n, dtype=torch.bool, device=device)
  down = torch.zeros(n, dtype=torch.bool, device=device)
  t_reach = torch.full((n,), float("nan"), device=device)
  collided = torch.zeros(n, dtype=torch.bool, device=device)
  fell_off = torch.zeros(n, dtype=torch.bool, device=device)
  incident = torch.zeros(n, device=device)
  """Worst severity a non-foot body part has touched (or a drop's, gone
  over one) from MPPI's start until arrival."""
  start_z = torch.full((n,), float("nan"), device=device)
  plan_time = []

  obs, _ = wrapped.reset()
  for i in range(num_steps):
    with torch.no_grad():
      obs, _, _, _ = wrapped.step(policy(obs))
      vel_est, latent = policy_estimates(actor, obs, n, device)
    step = i + 1
    t = step * step_dt
    data = robot.data
    pos, quat = data.root_link_pos_w, data.root_link_quat_w

    odom_xy += torch.randn_like(odom_xy) * cfg.odom_xy_std * math.sqrt(step_dt)
    odom_yaw += torch.randn_like(odom_yaw) * cfg.odom_yaw_std * math.sqrt(step_dt)
    est_pos = pos + torch.cat([odom_xy, torch.zeros_like(odom_yaw)[:, None]], -1)
    est_quat = _yaw_quat_mul(quat, odom_yaw)
    ring_pos = torch.cat([ring_pos[:, 1:], est_pos[:, None]], 1)
    ring_quat = torch.cat([ring_quat[:, 1:], est_quat[:, None]], 1)

    z = elev.data.hit_pos_w[..., 2] - pos[:, 2:3]
    height = torch.where(elev.data.distances < 0, float("nan"), z).view(n, cells, cells)
    severity = geom_sev[elev.data.hit_geom_id.long()].view(n, cells, cells)
    base_height = -height[:, mid, mid].flatten(1).nanmedian(dim=-1).values
    tilt = tilt_angle(data.projected_gravity_b)
    _, _, contact_w, force_w, valid = contacts_in_base_frame(sensor, robot)
    trunk_hit = (valid & slot_trunk & (force_w.norm(dim=-1) >= 1.0)).any(-1)
    started = t >= cfg.start_s
    down |= fallen(tilt, base_height) & started
    collided |= trunk_hit & started & ~reached
    if i == 0:
      start_z = pos[:, 2].clone()
    if cfg.terrain == "cliff":
      fell_off |= (pos[:, 2] < start_z - FELL_OFF_M) & started & ~reached
      down |= fell_off
      incident = torch.where(fell_off, incident.clamp(min=DROP_SEVERITY), incident)
    touch = (
      valid & slot_body & (force_w.norm(dim=-1) >= 1.0) & (started & ~reached)[:, None]
    )
    if touch.any():
      incident = torch.maximum(
        incident, _touched_severity(touch, contact_w, elev.data, geom_sev)
      )

    state = torch.cat(
      [
        vel_est / LIN_VEL_SCALE,
        data.root_link_ang_vel_b,
        data.projected_gravity_b,
        data.joint_pos,
        data.joint_vel,
      ],
      dim=-1,
    )
    for k, v in {
      "base_pos_w": pos,
      "base_quat_w": quat,
      "joint_pos": data.joint_pos,
      "cmd": command.command,
      "lin_vel_b": data.root_link_lin_vel_b,
      "ang_vel_b": data.root_link_ang_vel_b,
      "tilt": tilt,
      "base_height": base_height,
      "trunk_contact": trunk_hit,
      "faulted": fault.faulted,
    }.items():
      log[k][:, i] = v.detach().cpu().numpy()

    if step % r:
      continue
    hist.push(torch.cat([state, command.command], dim=-1))
    if t < cfg.start_s:
      continue

    est_yaw = heading(est_quat)
    goal_term.advance(est_pos[:, :2], cfg.waypoint_radius)
    goal = _to_yaw_frame_2d(command.goal_w[:, :2], est_pos[:, :2], est_yaw)
    near = goal_term.final & (goal.norm(dim=-1) < cfg.arrive_radius)
    newly = (~reached & near).nonzero()[:, 0]
    reached[newly] = True
    t_reach[newly] = t
    stopped = reached | down
    fresh = ~active & ~stopped
    if fresh.any():
      planner.reset(fresh.nonzero()[:, 0], command.command[fresh])
    active = ~stopped

    c = step // r
    cyc["step"][c] = i
    # Trunk-to-obstacle distance on the clean map (an evaluation metric only).
    clear_d = distance_field(
      obstacle_cells(height, cfg.mppi.footprint.wall_jump_m), MAP_RESOLUTION, 3.0
    )
    centre = slice(cells // 2 - 1, cells // 2 + 1)
    clear_now = clear_d[:, centre, centre].amin((-1, -2))
    clearance[:, c] = np.where(active.cpu().numpy(), clear_now.cpu().numpy(), np.nan)
    if cfg.controller == "follower":
      gc = command.goal_cfg
      err = wrap_to_pi(torch.atan2(goal[:, 1], goal[:, 0]))
      follow = torch.stack(
        [
          command.speed * torch.cos(err).clamp(min=0.0),
          torch.zeros_like(err),
          (gc.heading_gain * err).clamp(-gc.max_turn, gc.max_turn),
        ],
        dim=-1,
      )
      command.external[:] = torch.where(active[:, None], follow, 0.0)
      command.use_external |= started
      continue
    if not (fresh.any() or c % replan_k == 0):
      cmd = torch.where(active[:, None], planner.advance(), 0.0)
      command.external[:] = cmd
      # The committed plan stays where it was drawn in the world.
      off = ~active.cpu().numpy()
      for k, arr in [*cyc.items(), *cost_logs.items()]:
        if k != "step":
          arr[:, c] = arr[:, c - 1]
          arr[off, c] = np.nan
      contact[:, c] = contact[:, c - 1]
      contact_height[:, c] = contact_height[:, c - 1]
      continue

    t0 = time.perf_counter()
    poses = [ring_len - 1] + [ring_len - 1 - b for b in back]
    single, fused = lidar_known(
      height,
      ring_pos[:, poses],
      ring_quat[:, poses],
      MAP_RESOLUTION,
      cfg.lidar,
      cfg.lidar_chunk,
    )
    noisy = corrupt_map(height, fused, cfg.noise)  # [n, 2, H, W]
    if model.map_mode == "clean":
      crop = crop_center(height, MAP_CELLS)
      model_map = torch.stack([torch.nan_to_num(crop), (~crop.isnan()).float()], 1)
    elif model.map_mode == "single":
      model_map = crop_center(corrupt_map(height, single, cfg.noise), MAP_CELLS)
    else:
      model_map = crop_center(noisy, MAP_CELLS)
    if cfg.sweep_map == "oracle":
      plan_h, plan_known = height, ~height.isnan()
    else:
      plan_known = noisy[:, 1] > 0.5
      plan_h = torch.where(plan_known, noisy[:, 0], float("nan"))
    told = fault.faulted if cfg.fault_aware else torch.zeros_like(fault.faulted)
    params = torch.where(told[:, None, None], fault.params, nominal)
    inputs = {
      "state": state,
      "him_latent": latent,
      "history": hist.data,
      "fault": encode_fault(params),
      "map": model_map,
    }
    bearing = wrap_to_pi(-est_yaw) if lane else None
    explain_now = cfg.explain and every and (step // r) % every == 0
    alt_inputs = None
    if explain_now:
      alt_told = torch.zeros_like(fault.faulted) if cfg.fault_aware else fault.faulted
      alt_params = torch.where(alt_told[:, None, None], fault.params, nominal)
      alt_inputs = inputs | {"fault": encode_fault(alt_params)}
    route = None
    if goal_term.waypoints.shape[1] > 1:
      # The waypoint after the current one (itself at the last), so plans
      # are scored on the remaining route, corner then goal.
      nxt = (goal_term.index + 1).clamp(max=goal_term.waypoints.shape[1] - 1)
      route_w = goal_term.waypoints[torch.arange(n, device=device), nxt, :2]
      route = _to_yaw_frame_2d(route_w, est_pos[:, :2], est_yaw)
    planner.track(est_pos[:, :2], replan_k * cfg.mppi.sampler.step_s)
    out = planner.step(
      inputs,
      plan_h,
      plan_known,
      MAP_RESOLUTION,
      goal,
      bearing,
      alt_inputs,
      route,
      exposed=~told,  # The fault risk prices faults still to come.
      severity=severity,
    )
    if explain_now:
      e, m = out["explain"], (step // r) // every
      xy = lambda v: v[..., :2].float().cpu().numpy()  # noqa: E731
      for k in ("path", "pose", "alt_pose"):
        ex[k][:, m] = xy(e[k][:, :, 4::5])
      for k in ("cost", "alt_cost", "weight", "alt_weight"):
        ex[k][:, m] = e[k].cpu().numpy()
      for k, v in (("terms", e["terms"]), ("alt_terms", e["alt_terms"])):
        for name, t in v.items():
          key = f"{k}_{name}"
          if key not in ex:
            ex[key] = np.full((n, num_maps, t.shape[-1]), np.nan, np.float32)
          ex[key][:, m] = t.cpu().numpy()
      ex["plan_path"][:, m] = xy(out["path"])
      ex["plan_pred"][:, m] = xy(out["pred_path"])
      ex["plan_pred_alt"][:, m] = xy(e["plan_pred_alt"])
      ex["alt_plan_path"][:, m] = xy(e["alt_plan_path"])
      ex["alt_plan_pred"][:, m] = xy(e["alt_plan_pred"])
      ex["alt_plan_pred_here"][:, m] = xy(e["alt_plan_pred_here"])
      ex["goal"][:, m] = goal.cpu().numpy()
    # What the model predicts for the chosen plan given the true fault (the
    # same as the plan's own prediction for a fault-aware planner).
    true_params = torch.where(fault.faulted[:, None, None], fault.params, nominal)
    true_pred = model.predict(
      inputs | {"fault": encode_fault(true_params)},
      out["plan"][:, None],
      out["path"][:, None],
    )
    if device.startswith("cuda"):
      torch.cuda.synchronize()
    plan_time.append(time.perf_counter() - t0)
    if every and (step // r) % every == 0:
      m = (step // r) // every
      maps[:, m] = plan_h.cpu().numpy()
      obstacles[:, m] = (
        obstacle_cells(plan_h, cfg.mppi.footprint.wall_jump_m).cpu().numpy()
      )

    cmd = torch.where(active[:, None], out["cmd"], 0.0)
    command.external[:] = cmd
    command.use_external |= started
    yaw = heading(quat)
    for k, v in {
      "path_w": _to_world(out["path"], pos, yaw),
      "pred_w": _to_world(out["pred_path"], pos, yaw),
      "pred_std": out["pred_std"],
      "pred_yaw_w": wrap_to_pi(out["pred_path"][..., 2] + yaw[:, None]),
      "pred_p_fallen": out["pred_p_fallen"],
      "pred_p_collision": out["pred_p_collision"],
      "p_fall_soon": out["p_fall_soon"],
      "drift_end": out["pred_path"][:, -1] - out["path"][:, -1],
      "origin_w": torch.cat([pos, yaw[:, None]], -1),
      "true_pred_w": _to_world(out["path"] + true_pred["drift"][:, 0], pos, yaw),
      "true_pred_std": true_pred["drift_std"][:, 0],
      "true_p_fallen": true_pred["p_fallen"][:, 0, -1],
      "p_fallen": out["p_fallen"],
      "ess": out["ess"],
    }.items():
      cyc[k][:, c] = np.where(
        active.cpu().numpy().reshape(-1, *[1] * (v.dim() - 1)),
        v.detach().cpu().numpy(),
        np.nan,
      )
    on = active.cpu().numpy()
    if "pred_contact" in out:
      probs = (out["pred_contact"][active].clamp(0, 1) * 255).round().byte()
      contact[on, c] = probs.cpu().numpy()
      contact_height[on, c] = crop_center(height[active], MAP_CELLS).cpu().numpy()
    for k, v in out.items():
      if k.startswith(("cost_", "probe_")):
        arr = cost_logs.setdefault(k, np.full((n, num_cycles), np.nan, np.float32))
        arr[:, c] = np.where(active.cpu().numpy(), v.cpu().numpy(), np.nan)

  fam = fault.family.cpu().numpy()
  joint = faulted_joint(fault.params).cpu().numpy()
  # Closest approach to an obstacle from the fault on (from MPPI's start
  # for robots without a fault) until arrival or a fall.
  from_step = np.maximum(fault.fault_step.cpu().numpy(), round(cfg.start_s / step_dt))
  after_fault = np.where(cyc["step"][None, :] >= from_step[:, None], clearance, np.nan)
  trials = [
    {
      "env": b,
      "family": FaultFamily(int(fam[b])).name,
      "joint": fault.joint_names[joint[b]] if joint[b] >= 0 else None,
      "torque": float(fault.params[b, joint[b], 0]) if joint[b] >= 0 else 1.0,
      "fault_time_s": float(fault.fault_step[b]) * step_dt
      if fault.faulted[b]
      else None,
      "reached": bool(reached[b]),
      "time_to_goal_s": None if torch.isnan(t_reach[b]) else float(t_reach[b]),
      "fell": bool(down[b]),
      "trunk_contact": bool(collided[b]),
      "fell_off": bool(fell_off[b]),
      "incident_severity": float(incident[b]),
      "min_clearance_m": float(np.nanmin(after_fault[b]))
      if np.isfinite(after_fault[b]).any()
      else None,
    }
    for b in range(n)
  ]
  summary: dict[str, dict | float | None] = {}
  outcome = {
    k: v.cpu().numpy()
    for k, v in (
      ("reached", reached),
      ("fell", down),
      ("trunk_contact", collided),
      ("fell_off", fell_off),
    )
  }
  t_goal = t_reach.cpu().numpy()
  incident_np = incident.cpu().numpy()
  with np.errstate(all="ignore"):
    min_clear = np.nanmin(np.where(np.isfinite(after_fault), after_fault, np.inf), 1)
  min_clear[~np.isfinite(min_clear)] = np.nan
  for name in ["ALL", *(f.name for f in FaultFamily)]:
    sel = np.ones(n, bool) if name == "ALL" else fam == FaultFamily[name]
    if not sel.any():
      continue
    times = t_goal[sel & outcome["reached"]]
    summary[name] = {
      "trials": int(sel.sum()),
      **{k: float(v[sel].mean()) for k, v in outcome.items()},
      "mean_time_to_goal_s": float(times.mean()) if len(times) else None,
      "mean_min_clearance_m": float(np.nanmean(min_clear[sel]))
      if np.isfinite(min_clear[sel]).any()
      else None,
      "close_calls": float((min_clear[sel] < CLOSE_CALL_M).mean()),
      "mean_incident_severity": float(incident_np[sel].mean()),
    }
  summary["plan_ms_mean"] = 1e3 * float(np.mean(plan_time)) if plan_time else None

  out_path = Path(cfg.output)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  arrays: dict[str, np.ndarray] = {
    **{f"dense_{k}": v for k, v in log.items()},
    **{f"cycle_{k}": v for k, v in cyc.items()},
    **{f"cycle_{k}": v for k, v in cost_logs.items()},
  }
  np.savez_compressed(
    out_path.parent / f"{out_path.name}.npz",
    allow_pickle=False,
    **arrays,
    map_height=maps,
    map_obstacle=obstacles,
    cycle_clearance=clearance,
    contact_prob=contact,
    contact_height=contact_height,
    map_every=np.array(every),
    **{f"explain_{k}": v for k, v in ex.items()},
    fault_aware=np.array(cfg.fault_aware),
    goal_w=command.goal_w.cpu().numpy(),
    waypoints_w=goal_term.waypoints.cpu().numpy(),
    fault_params=fault.params.cpu().numpy(),
    fault_family=fam,
    fault_step=fault.fault_step.cpu().numpy(),
    joint_names=np.array(fault.joint_names),
    terrain=np.array(cfg.terrain),
    terrain_seed=np.array(-1 if cfg.terrain_seed is None else cfg.terrain_seed),
    step_dt=np.array(step_dt),
    start_s=np.array(cfg.start_s),
  )
  result = {"config": asdict(cfg), "summary": summary, "trials": trials}
  (out_path.parent / f"{out_path.name}.json").write_text(json.dumps(result, indent=2))
  env.close()
  return result


def main():
  cfg = tyro.cli(RunMppiConfig)
  if not Path(cfg.checkpoint_file).is_file():
    print(f"No checkpoint at {cfg.checkpoint_file}.")
    sys.exit(1)
  result = run(cfg)
  for name, s in result["summary"].items():
    print(f"[INFO] {name}: {s}")


if __name__ == "__main__":
  main()
