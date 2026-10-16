"""What happens to the robot after a joint fault: path drift and contact risk.

Helpers for ``heat_bench/scripts/fault_impact.py``, which logs fault trials
(``joint_fault_injection.py``) over a long horizon so trials the robot
survives are covered as well as the falls. All frames and grids here match
the robot's elevation map, so the contact "risk zone" can be laid over the
terrain the robot saw when the joint failed.

- ``*_fy``: the *fault yaw frame*. Origin at the root link at fault time,
  x/y rotated by the fault-time heading only, z up. It is gravity-aligned
  like ``terrain_scan`` (``ray_alignment="yaw"``), so pitch/roll at fault
  time don't tilt the map.
- Deviation follows the velocity-tracking view of HIM-style policies: the
  policy aims for the commanded body twist, so ``dv = v_actual - v_cmd``
  (body frame) is the fault's effect on speed. The planar path drift
  accumulated from it is the actual pose minus the commanded twist
  integrated from the fault pose.
- Contacts are grouped per body part (``FR_thigh`` covers its three
  collision capsules) and split into contact *events*: one per stretch of
  touching, with onset/end times relative to the fault.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensor, RayCastSensorCfg
from mjlab.terrains.terrain_generator import TerrainGeneratorCfg
from mjlab.utils.lab_api.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
  from mjlab.viewer.debug_visualizer import DebugVisualizer

FAULT_ELEVATION_SENSOR = "fault_elevation_map"
BODY_GROUPS = ("trunk", "head", "hip", "thigh", "calf", "foot")
TRUNK_GROUPS = ("trunk", "head")
"""The trunk body: its box plus the head sphere fixed to it. Only these
count as the robot hitting the environment -- a fall, a contact event, an
impact or the body sweep. Legs touch the terrain in normal walking (calves
graze stair edges), so leg contact is logged per step but isn't impact."""


def body_group(geom_name: str) -> str:
  for group in BODY_GROUPS:
    if group in geom_name:
      return group
  return "other"


def body_part(geom_name: str) -> str:
  """``FR_thigh_collision2`` -> ``FR_thigh``; ``trunk_collision`` -> ``trunk``."""
  return re.sub(r"_collision\d*$", "", geom_name)


@dataclass(frozen=True)
class GridSpec:
  """Cell layout of a ``GridPatternCfg``: ray ``i`` sits at the centre of
  cell ``(iy, ix) = divmod(i, nx)``, ``x = -size_x / 2 + ix * resolution``."""

  size: tuple[float, float]
  resolution: float

  def _num(self, extent: float) -> int:
    # Same arange as GridPatternCfg.generate_rays, so counts always agree.
    r = self.resolution
    return torch.arange(-extent / 2, extent / 2 + r * 0.5, r).numel()

  @property
  def nx(self) -> int:
    return self._num(self.size[0])

  @property
  def ny(self) -> int:
    return self._num(self.size[1])

  def cell_index(self, xy: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Flat cell index (``iy * nx + ix``) of each point and an in-grid mask."""
    ix = torch.round((xy[..., 0] + self.size[0] / 2) / self.resolution).long()
    iy = torch.round((xy[..., 1] + self.size[1] / 2) / self.resolution).long()
    inside = (ix >= 0) & (ix < self.nx) & (iy >= 0) & (iy < self.ny)
    return iy * self.nx + ix, inside


@dataclass
class RaisedGridPatternCfg(GridPatternCfg):
  """``GridPatternCfg`` with ray origins lifted ``origin_height`` above the
  frame. Downward rays from the trunk itself miss terrain higher than the
  trunk (a stair pyramid ahead, or anything once the robot lies on the
  ground); starting higher sees it. Hit positions are unaffected."""

  origin_height: float = 2.0

  def generate_rays(
    self, mj_model: mujoco.MjModel | None, device: str
  ) -> tuple[torch.Tensor, torch.Tensor]:
    offsets, directions = super().generate_rays(mj_model, device)
    offsets[:, 2] += self.origin_height
    return offsets, directions


# Height colour ramp, low -> high: blue, cyan, green, yellow, red.
_RAMP = np.array(
  [[0.1, 0.2, 0.9], [0.0, 0.8, 0.9], [0.1, 0.8, 0.2], [1.0, 0.9, 0.1], [0.9, 0.1, 0.1]]
)


class ElevationMapSensor(RayCastSensor):
  """The elevation-map raycast, drawn in the viewer as one flat tile per
  cell at the terrain height, coloured by height relative to the ground
  under the trunk (blue = below, red = above, saturating at
  ``viz_height_range`` m). Misses are skipped."""

  cfg: ElevationMapSensorCfg

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    if not self.cfg.debug_vis or not self._debug_vis_enabled:
      return
    data = self.data
    env_ids = list(visualizer.get_env_indices(data.distances.shape[0]))
    pattern = self.cfg.pattern
    assert isinstance(pattern, GridPatternCfg)
    grid = GridSpec(size=pattern.size, resolution=pattern.resolution)
    half = np.array([0.45 * grid.resolution, 0.45 * grid.resolution, 0.004])
    centre = (grid.ny // 2) * grid.nx + grid.nx // 2
    yaw = heading(data.quat_w[env_ids]).cpu().numpy()
    hits = data.hit_pos_w[env_ids].cpu().numpy()
    ok = (data.distances[env_ids] >= 0).cpu().numpy()
    r = self.cfg.viz_height_range
    for k in range(len(env_ids)):
      c, s = np.cos(yaw[k]), np.sin(yaw[k])
      mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
      ground = hits[k, centre, 2] if ok[k, centre] else np.nanmedian(hits[k, ok[k], 2])
      u = np.clip((hits[k, :, 2] - ground + r) / (2 * r), 0.0, 1.0) * (len(_RAMP) - 1)
      lo = np.floor(u).astype(int).clip(max=len(_RAMP) - 2)
      rgb = _RAMP[lo] + (u - lo)[:, None] * (_RAMP[lo + 1] - _RAMP[lo])
      for i in np.flatnonzero(ok[k]):
        color = (float(rgb[i, 0]), float(rgb[i, 1]), float(rgb[i, 2]), 0.85)
        visualizer.add_box(hits[k, i] + [0.0, 0.0, 0.005], half, mat, color)


@dataclass
class ElevationMapSensorCfg(RayCastSensorCfg):
  viz_height_range: float = 0.5
  """Height above/below the ground under the trunk (m) where the tile colour
  saturates."""

  def build(self) -> ElevationMapSensor:
    return ElevationMapSensor(self)


def fault_elevation_sensor_cfg(
  size: tuple[float, float] = (4.0, 4.0),
  resolution: float = 0.1,
  origin_height: float = 2.0,
  debug_vis: bool = False,
) -> ElevationMapSensorCfg:
  """A trunk-mounted, yaw-aligned terrain height grid, like ``terrain_scan``
  but large enough to cover where a falling robot lands, and cast from
  ``origin_height`` above the trunk so terrain above the trunk is seen.
  Snapshotted at fault time and at fall onset. ``debug_vis`` draws it in
  the viewer as height-coloured tiles."""
  return ElevationMapSensorCfg(
    name=FAULT_ELEVATION_SENSOR,
    frame=ObjRef(type="body", name="trunk", entity="robot"),
    ray_alignment="yaw",
    pattern=RaisedGridPatternCfg(
      size=size, resolution=resolution, origin_height=origin_height
    ),
    max_distance=origin_height + 5.0,
    exclude_parent_body=True,
    include_geom_groups=(0,),  # Terrain only.
    debug_vis=debug_vis,
  )


def heading(quat_w: torch.Tensor) -> torch.Tensor:
  """Yaw angle (rad) of ``(w, x, y, z)`` quaternions."""
  w, x, y, z = quat_w.unbind(-1)
  return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def to_yaw_frame(
  pos_w: torch.Tensor, origin_w: torch.Tensor, quat_w: torch.Tensor
) -> torch.Tensor:
  """World points ``[B, ..., 3]`` into the yaw-only frame of each env's
  ``origin_w [B, 3]`` / ``quat_w [B, 4]``."""
  extra = pos_w.dim() - 2
  shape = (pos_w.shape[0],) + (1,) * extra
  q = yaw_quat(quat_w).view(*shape, 4).expand(*pos_w.shape[:-1], 4)
  return quat_apply_inverse(q, pos_w - origin_w.view(*shape, 3))


def rotate_to_yaw_frame(vec_w: torch.Tensor, quat_w: torch.Tensor) -> torch.Tensor:
  """World vectors ``[B, ..., 3]`` (forces) into each env's yaw-only frame."""
  return to_yaw_frame(vec_w, torch.zeros_like(quat_w[:, :3]), quat_w)


def integrate_command_path(cmd: torch.Tensor, dt: float) -> torch.Tensor:
  """Planar pose ``(x, y, yaw)`` after each step of following ``cmd``.

  ``cmd [B, K, 3]`` is the body twist ``(vx, vy, wz)`` held over each of ``K``
  steps, starting from the origin with zero heading; the result is
  ``[B, K, 3]``. Midpoint heading per step, so a constant turn traces its
  circle to second order in ``wz * dt``.
  """
  vx, vy, wz = cmd.unbind(-1)
  yaw = torch.cumsum(wz * dt, dim=-1)
  mid = yaw - 0.5 * wz * dt
  dx = (vx * torch.cos(mid) - vy * torch.sin(mid)) * dt
  dy = (vx * torch.sin(mid) + vy * torch.cos(mid)) * dt
  return torch.stack([dx.cumsum(-1), dy.cumsum(-1), yaw], dim=-1)


def path_deviation(actual: torch.Tensor, expected: torch.Tensor) -> torch.Tensor:
  """``actual - expected`` planar poses ``[..., 3]``, heading error wrapped."""
  d = actual - expected
  return torch.cat([d[..., :2], wrap_to_pi(d[..., 2:])], dim=-1)


def tilt_angle(projected_gravity_b: torch.Tensor) -> torch.Tensor:
  """Angle (rad) between the trunk's up axis and world up."""
  g = projected_gravity_b / projected_gravity_b.norm(dim=-1, keepdim=True)
  return torch.acos((-g[..., 2]).clamp(-1.0, 1.0))


def fire_step(fault_time_s: torch.Tensor, step_dt: float) -> torch.Tensor:
  """Episode step at which ``inject_joint_fault`` fires: the first ``k``
  with ``float(k) * step_dt >= fault_time_s``, computed the same way."""
  k = torch.ceil(fault_time_s / step_dt)
  k = torch.where((k - 1) * step_dt >= fault_time_s, k - 1, k)
  k = torch.where(k * step_dt < fault_time_s, k + 1, k)
  return k.long()


def terrain_tile(
  pos_w: torch.Tensor, gen: TerrainGeneratorCfg
) -> tuple[torch.Tensor, torch.Tensor]:
  """``(row, col)`` of the generator tile under each point, -1 on the border.

  In curriculum mode the row is the difficulty level and the column the
  sub-terrain type (``list(gen.sub_terrains)[col]``).
  """
  num_cols = len(gen.sub_terrains) if gen.curriculum else gen.num_cols
  sx, sy = gen.size
  row = torch.floor((pos_w[..., 0] + gen.num_rows * sx / 2) / sx).long()
  col = torch.floor((pos_w[..., 1] + num_cols * sy / 2) / sy).long()
  inside = (row >= 0) & (row < gen.num_rows) & (col >= 0) & (col < num_cols)
  return torch.where(inside, row, -1), torch.where(inside, col, -1)


def rasterize_contacts(
  env_id: torch.Tensor,
  pos_fy: torch.Tensor,
  force: torch.Tensor,
  dt: torch.Tensor,
  num_envs: int,
  grid: GridSpec,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Contacts onto each env's elevation-map grid.

  Takes one row per contact (slot and step): ``env_id [N]``, ``pos_fy
  [N, 3]``, force magnitude ``[N]`` and time since the fault ``[N]``.
  Returns ``risk [B, 3, ny, nx]`` -- contact-step count, max force (N), and
  earliest time since the fault (s, NaN where never touched) -- and the
  per-env count of contacts that fell outside the grid.
  """
  device = pos_fy.device
  cells = grid.ny * grid.nx
  cell, inside = grid.cell_index(pos_fy[:, :2])
  outside = torch.zeros(num_envs, dtype=torch.long, device=device)
  outside.index_add_(0, env_id[~inside], torch.ones_like(env_id[~inside]))
  flat = env_id[inside] * cells + cell[inside]
  f, t = force[inside], dt[inside]

  count = torch.zeros(num_envs * cells, device=device)
  count.index_add_(0, flat, torch.ones_like(f))
  peak = torch.zeros(num_envs * cells, device=device)
  peak.scatter_reduce_(0, flat, f, reduce="amax")
  first = torch.full((num_envs * cells,), float("inf"), device=device)
  first.scatter_reduce_(0, flat, t, reduce="amin")
  first[first.isinf()] = float("nan")
  risk = torch.stack([count, peak, first]).view(3, num_envs, grid.ny, grid.nx)
  return risk.transpose(0, 1).contiguous(), outside


def geom_sample_points(
  geom_type: np.ndarray, geom_size: np.ndarray, geom_rbound: np.ndarray
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Spheres covering each collision geom, for the body-sweep footprint.

  Capsules and cylinders get 5 spheres along their axis, boxes a 3x3 grid
  in their local xy plane (radius = half-height), spheres one; anything
  else one sphere of its bounding radius. Returns ``(geom_index [S],
  local_offset [S, 3], radius [S])``.
  """
  idx, off, rad = [], [], []
  for g, (t, size, rb) in enumerate(
    zip(geom_type, geom_size, geom_rbound, strict=True)
  ):
    if t == mujoco.mjtGeom.mjGEOM_SPHERE:
      pts, r = [(0.0, 0.0, 0.0)], size[0]
    elif t in (mujoco.mjtGeom.mjGEOM_CAPSULE, mujoco.mjtGeom.mjGEOM_CYLINDER):
      pts = [(0.0, 0.0, z) for z in np.linspace(-size[1], size[1], 5)]
      r = size[0]
    elif t == mujoco.mjtGeom.mjGEOM_BOX:
      pts = [
        (x, y, 0.0) for x in (-size[0], 0.0, size[0]) for y in (-size[1], 0.0, size[1])
      ]
      r = size[2]
    else:
      pts, r = [(0.0, 0.0, 0.0)], rb
    idx += [g] * len(pts)
    off += pts
    rad += [r] * len(pts)
  return (
    torch.tensor(idx, dtype=torch.long),
    torch.tensor(off, dtype=torch.float32),
    torch.tensor(rad, dtype=torch.float32),
  )


def sweep_cells(
  points: torch.Tensor, radius: torch.Tensor, grid: GridSpec
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
  """Grid cells under body spheres.

  ``points [B, S, 3]`` are sphere centres in the grid's frame and ``radius
  [S]`` their radii. A cell counts when its centre is within ``radius +
  resolution / 2`` of the centre in xy. Returns flat ``(env_id, cell,
  bottom_z)`` entries, ``bottom_z`` being the sphere's lowest point.
  """
  res = grid.resolution
  reach = int(np.ceil(float(radius.max()) / res)) + 1
  k = torch.arange(-reach, reach + 1, device=points.device)
  dy, dx = torch.meshgrid(k, k, indexing="ij")
  dx, dy = dx.flatten(), dy.flatten()
  ix0 = torch.round((points[..., 0] + grid.size[0] / 2) / res).long()
  iy0 = torch.round((points[..., 1] + grid.size[1] / 2) / res).long()
  ix = ix0[..., None] + dx  # [B, S, K]
  iy = iy0[..., None] + dy
  cx = -grid.size[0] / 2 + ix * res
  cy = -grid.size[1] / 2 + iy * res
  dist = torch.hypot(cx - points[..., 0:1], cy - points[..., 1:2])
  r = radius.to(points.device)[None, :, None]
  hit = (
    (dist <= r + res / 2 + 1e-6)
    & (ix >= 0)
    & (ix < grid.nx)
    & (iy >= 0)
    & (iy < grid.ny)
  )
  b, si, ki = hit.nonzero(as_tuple=True)
  cell = iy[b, si, ki] * grid.nx + ix[b, si, ki]
  bottom = points[b, si, 2] - r[0, si, 0]
  return b, cell, bottom


class SweepMap:
  """Per-env footprint of the robot's body on an elevation-map grid,
  accumulated over steps: whether each cell was ever under the body, when
  first (s since the fault), and the smallest clearance between the body's
  bottom and that cell's terrain (m; <= 0 means touching)."""

  def __init__(self, num_envs: int, grid: GridSpec, device: str):
    self.grid = grid
    cells = grid.ny * grid.nx
    self.occupied = torch.zeros(num_envs * cells, dtype=torch.bool, device=device)
    self.first_dt = torch.full((num_envs * cells,), float("inf"), device=device)
    self.min_clearance = torch.full((num_envs * cells,), float("inf"), device=device)

  def update(
    self,
    rows: torch.Tensor,
    points: torch.Tensor,
    radius: torch.Tensor,
    terrain: torch.Tensor,
    dt: torch.Tensor,
  ) -> None:
    """Add one step for envs ``rows``: ``points [R, S, 3]`` in the map
    frame, ``terrain [R, ny, nx]`` the map's heights in that frame (NaN =
    unknown), ``dt [R]`` the time since the fault."""
    b, cell, bottom = sweep_cells(points, radius, self.grid)
    flat = rows[b] * (self.grid.ny * self.grid.nx) + cell
    self.occupied[flat] = True
    self.first_dt.scatter_reduce_(0, flat, dt[b], reduce="amin")
    clear = bottom - terrain.flatten(1)[b, cell]
    ok = ~clear.isnan()
    self.min_clearance.scatter_reduce_(0, flat[ok], clear[ok], reduce="amin")

  def result(self) -> torch.Tensor:
    """``[B, 3, ny, nx]``: occupied (0/1), first time (NaN if never), min
    clearance (NaN if never or terrain unknown)."""
    n = self.occupied.numel() // (self.grid.ny * self.grid.nx)
    first = self.first_dt.masked_fill(self.first_dt.isinf(), float("nan"))
    clear = self.min_clearance.masked_fill(self.min_clearance.isinf(), float("nan"))
    out = torch.stack([self.occupied.float(), first, clear])
    return out.view(3, n, self.grid.ny, self.grid.nx).transpose(0, 1).contiguous()


RISK_CHANNELS = (
  "touched",
  "peak_force_n",
  "first_contact_dt_s",
  "impacts",
  "contact_steps",
)
"""Channels of a contact risk grid: any contact (0/1), the largest contact
force (N), the earliest contact time after the fault (s, NaN if never),
the number of contact events that *started* in the cell (impacts, not time
spent resting), and contact-steps (20 ms steps x contact points; dominated
by a robot lying still)."""
SWEEP_CHANNELS = ("occupied", "first_dt_s", "min_clearance_m")


def assemble_risk(contacts: torch.Tensor, impacts: torch.Tensor) -> torch.Tensor:
  """``RISK_CHANNELS`` grid ``[B, 5, ny, nx]`` from a ``rasterize_contacts``
  grid of contact rows ``[B, 3, ny, nx]`` and per-cell event onset counts
  ``[B, ny, nx]``."""
  count, peak, first = contacts.unbind(1)
  touched = (count > 0).float()
  return torch.stack([touched, peak, first, impacts, count], dim=1)


def window_risk(
  contacts: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
  events: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
  edges: torch.Tensor,
  num_envs: int,
  grid: GridSpec,
) -> torch.Tensor:
  """``RISK_CHANNELS`` grids per time window after the fault.

  ``contacts`` and ``events`` are ``(env_id [N], pos [N, 3], force [N],
  dt [N])`` in the grid's frame -- contact rows, and event onsets with
  their peak force. Window ``w`` holds rows with ``edges[w] <= dt <
  edges[w + 1]``. Returns ``[B, W, 5, ny, nx]``.
  """
  out = []
  for lo, hi in zip(edges[:-1].tolist(), edges[1:].tolist(), strict=True):
    c_in = (contacts[3] >= lo) & (contacts[3] < hi)
    e_in = (events[3] >= lo) & (events[3] < hi)
    c_env, c_pos, c_force, c_dt = (x[c_in] for x in contacts)
    e_env, e_pos, e_force, e_dt = (x[e_in] for x in events)
    base, _ = rasterize_contacts(c_env, c_pos, c_force, c_dt, num_envs, grid)
    impacts, _ = rasterize_contacts(e_env, e_pos, e_force, e_dt, num_envs, grid)
    out.append(assemble_risk(base, impacts[:, 0]))
  return torch.stack(out, dim=1)


def extract_contact_events(
  active: np.ndarray,
  force: np.ndarray,
  dt: np.ndarray,
  step_dt: float,
  debounce_steps: int = 2,
) -> dict[str, np.ndarray]:
  """Split per-step contact into events.

  ``active [B, T, P]`` and ``force [B, T, P]`` (summed force magnitude, N)
  are per env, logged step and body part; ``dt [B, T]`` is each step's time
  since the fault (NaN for unlogged steps). Gaps of at most
  ``debounce_steps`` contact-free steps don't end an event, so a bounce
  stays one impact. Returns flat arrays, one entry per event:

  - ``env_id``, ``part``, ``order`` (0 = the env's first event by onset,
    ties broken by the stronger hit), ``onset_step`` / ``end_step``
    (logged-step index),
  - ``onset_dt`` / ``end_dt`` (s since the fault), ``duration_s``,
  - ``peak_force_n``, ``impulse_ns`` (sum of force * step_dt).
  """
  cols: dict[str, list] = {
    k: []
    for k in ("env_id", "part", "onset_step", "end_step", "peak_force_n", "impulse_ns")
  }
  for b, p in zip(*np.nonzero(active.any(axis=1)), strict=True):
    steps = np.flatnonzero(active[b, :, p])
    breaks = np.flatnonzero(np.diff(steps) > debounce_steps + 1)
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks, [len(steps) - 1]])
    for s, e in zip(steps[starts], steps[ends], strict=True):
      f = force[b, s : e + 1, p]
      cols["env_id"].append(b)
      cols["part"].append(p)
      cols["onset_step"].append(s)
      cols["end_step"].append(e)
      cols["peak_force_n"].append(f.max())
      cols["impulse_ns"].append(f.sum() * step_dt)
  ev: dict[str, np.ndarray] = {
    k: np.asarray(v, dtype=np.float32 if "force" in k or "impulse" in k else np.int64)
    for k, v in cols.items()
  }
  env_id = ev["env_id"].astype(np.int64)
  ev["onset_dt"] = np.asarray(dt)[env_id, ev["onset_step"]].astype(np.float32)
  ev["end_dt"] = np.asarray(dt)[env_id, ev["end_step"]].astype(np.float32)
  ev["duration_s"] = ((ev["end_step"] - ev["onset_step"] + 1) * step_dt).astype(
    np.float32
  )
  # Rank within each env: by onset, then stronger hit first.
  idx = np.lexsort((-ev["peak_force_n"], ev["onset_step"], ev["env_id"]))
  order = np.zeros(len(idx), dtype=np.int64)
  sorted_env = ev["env_id"][idx]
  first_of_env = np.searchsorted(sorted_env, sorted_env, side="left")
  order[idx] = np.arange(len(idx)) - first_of_env
  ev = {k: v[idx] for k, v in ev.items()}
  ev["order"] = order[idx]
  return ev
