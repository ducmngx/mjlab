"""Unitree Go2 constants.

The MJCF is Unitree's own Go2 model from unitree_mujoco
(github.com/unitreerobotics/unitree_mujoco, ``unitree_robots/go2``, commit
1eb6642, BSD-3-Clause, see ``LICENSE``), copied unchanged. ``get_spec``
adapts it to mjlab's conventions in code, mirroring mjlab's Go1 asset
(``src/mjlab/asset_zoo/robots/unitree_go1``):

- the torque motors, sensors and keyframe are removed (mjlab adds its own
  actuators, and the IMU sensors below), and geom margins are zeroed;
- the base body ``base_link`` is renamed ``trunk``;
- collision geoms get names (``trunk_collision``, ``{leg}_thigh_collision``,
  ``{leg}_foot_collision``, ...) and each foot gets a ``{leg}`` site;
- with ``capsules=True`` (the default) the head, hip, thigh and calf
  cylinders/boxes become capsules (or spheres, for the squat head and hip
  cylinders) of the same radius and length (see ``_to_capsule``); the
  trunk box and the chin and foot spheres stay.
"""

from functools import partial
from pathlib import Path
from typing import Literal

import mujoco
import numpy as np

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

GO2_XML: Path = Path(__file__).parent / "go2.xml"
assert GO2_XML.exists()

LEGS = ("FR", "FL", "RR", "RL")
"""Leg order of the MJCF and of the Unitree SDK motor indices."""

# Collision geoms of the base body, in MJCF order: box, head cylinder, chin
# sphere.
_TRUNK_GEOM_NAMES = ("trunk_collision", "head_collision", "head_lower_collision")
# Foot sphere centre in the calf frame (the ``foot`` default class).
_FOOT_POS = (-0.002, 0.0, -0.213)


def _collision_geoms(body: mujoco.MjsBody) -> list[mujoco.MjsGeom]:
  return [g for g in body.geoms if g.group == 3]


def _quat_rotate(q, v: np.ndarray) -> np.ndarray:
  out = np.zeros(3)
  mujoco.mju_rotVecQuat(out, v, np.asarray(q, dtype=float))
  return out


def _to_capsule(geom: mujoco.MjsGeom) -> None:
  """Replace a cylinder or box collision geom with a capsule of matching
  radius and end-to-end length (mjlab's Go1 uses capsules for the same
  parts). mujoco_warp's convex narrowphase allocates large per-pair buffers
  for heightfield-vs-cylinder/box contacts, enough to exhaust GPU memory
  with a few thousand envs; capsule-vs-heightfield is cheap."""
  pos = np.asarray(geom.pos, dtype=float).copy()
  if geom.type == mujoco.mjtGeom.mjGEOM_CYLINDER:
    radius, half = float(geom.size[0]), float(geom.size[1])
    axis = np.array([0.0, 0.0, 1.0])
  elif geom.type == mujoco.mjtGeom.mjGEOM_BOX:
    sizes = np.asarray(geom.size, dtype=float)
    long_axis = int(np.argmax(sizes))
    half = float(sizes[long_axis])
    radius = float(np.max(np.delete(sizes, long_axis)))
    axis = np.eye(3)[long_axis]
  else:
    raise ValueError(f"Cannot convert geom type {geom.type} to a capsule.")
  direction = _quat_rotate(geom.quat, axis)
  core = half - radius
  if core <= 1e-3:  # Squat cylinder (a disc): a sphere of its radius.
    geom.type = mujoco.mjtGeom.mjGEOM_SPHERE
    geom.size = np.array([radius, 0.0, 0.0])
    return
  geom.type = mujoco.mjtGeom.mjGEOM_CAPSULE
  geom.size = np.array([radius, core, 0.0])
  # Capsules lie along their local z axis.
  quat = np.zeros(4)
  mujoco.mju_quatZ2Vec(quat, direction)
  geom.quat = quat
  geom.pos = pos


def get_spec(capsules: bool = True) -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  for element in [*spec.actuators, *spec.sensors, *spec.keys]:
    spec.delete(element)
  # Solver options belong to the task (MujocoCfg), not the robot; reset the
  # MJCF's cone/impratio to defaults so attaching doesn't warn.
  default_opt = mujoco.MjSpec().option
  spec.option.cone = default_opt.cone
  spec.option.impratio = default_opt.impratio
  # The MJCF default gives every geom a 1 mm margin; mujoco_warp's multi-CCD
  # needs zero margins.
  for geom in spec.geoms:
    geom.margin = 0.0

  trunk = spec.body("base_link")
  trunk.name = "trunk"
  for geom, name in zip(_collision_geoms(trunk), _TRUNK_GEOM_NAMES, strict=True):
    geom.name = name
  if capsules:
    _to_capsule(spec.geom("head_collision"))
  trunk.add_site(name="head", pos=(0.3, 0.0, 0.0), size=(0.02, 0, 0), group=5)

  for leg in LEGS:
    (hip_geom,) = _collision_geoms(spec.body(f"{leg}_hip"))
    hip_geom.name = f"{leg}_hip_collision"
    (thigh_geom,) = _collision_geoms(spec.body(f"{leg}_thigh"))
    thigh_geom.name = f"{leg}_thigh_collision"
    calf = spec.body(f"{leg}_calf")
    shin_upper, shin_lower, foot = _collision_geoms(calf)
    shin_upper.name = f"{leg}_calf_collision1"
    shin_lower.name = f"{leg}_calf_collision2"
    foot.name = f"{leg}_foot_collision"
    if capsules:
      for geom in (hip_geom, thigh_geom, shin_upper, shin_lower):
        _to_capsule(geom)
    calf.add_site(
      name=leg,
      pos=_FOOT_POS,
      type=mujoco.mjtGeom.mjGEOM_SPHERE,
      size=(0.023, 0, 0),
      group=5,
    )

  # The IMU sensors mjlab's velocity task reads (same as its Go1 MJCF).
  spec.add_sensor(
    name="imu_ang_vel",
    type=mujoco.mjtSensor.mjSENS_GYRO,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_vel",
    type=mujoco.mjtSensor.mjSENS_VELOCIMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_lin_acc",
    type=mujoco.mjtSensor.mjSENS_ACCELEROMETER,
    objtype=mujoco.mjtObj.mjOBJ_SITE,
    objname="imu",
  )
  spec.add_sensor(
    name="imu_upvector",
    type=mujoco.mjtSensor.mjSENS_FRAMEZAXIS,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="world",
    reftype=mujoco.mjtObj.mjOBJ_SITE,
    refname="imu",
  )
  spec.add_sensor(
    name="root_angmom",
    type=mujoco.mjtSensor.mjSENS_SUBTREEANGMOM,
    objtype=mujoco.mjtObj.mjOBJ_BODY,
    objname="trunk",
  )
  return spec


##
# Actuators.
##

# PD gains. "unitree": Unitree's own Go2 RL configs (unitree_rl_gym go2:
# stiffness 20, damping 0.5). "himloco": HIMLoco's (go1_config: stiffness
# 40, damping 1). Action scale is 0.25 in both. Torque limits are the MJCF
# motor ctrlranges. Armature (0.01), joint damping (0.1) and friction loss
# (0.2) stay as set in the MJCF.
GAINS = {"unitree": (20.0, 0.5), "himloco": (40.0, 1.0)}
Gains = Literal["unitree", "himloco"]
HIP_EFFORT_LIMIT = 23.7
KNEE_EFFORT_LIMIT = 45.43
ACTION_SCALE = 0.25


def go2_articulation(gains: Gains = "unitree") -> EntityArticulationInfoCfg:
  """Fresh hip/thigh and knee actuator configs with the chosen PD gains."""
  kp, kd = GAINS[gains]
  return EntityArticulationInfoCfg(
    actuators=(
      BuiltinPositionActuatorCfg(
        target_names_expr=(".*_hip_joint", ".*_thigh_joint"),
        stiffness=kp,
        damping=kd,
        effort_limit=HIP_EFFORT_LIMIT,
      ),
      BuiltinPositionActuatorCfg(
        target_names_expr=(".*_calf_joint",),
        stiffness=kp,
        damping=kd,
        effort_limit=KNEE_EFFORT_LIMIT,
      ),
    ),
    soft_joint_pos_limit_factor=0.9,
  )


##
# Initial state: Unitree / HIMLoco default stance.
##

INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.42),
  joint_pos={
    ".*L_hip_joint": 0.1,
    ".*R_hip_joint": -0.1,
    "F[LR]_thigh_joint": 0.8,
    "R[LR]_thigh_joint": 1.0,
    ".*_calf_joint": -1.5,
  },
  joint_vel={".*": 0.0},
)

##
# Collisions (same scheme as mjlab's Go1).
##

_foot_regex = "^[FR][LR]_foot_collision$"
_collision_regex = r".*_collision\d*"

FULL_COLLISION = CollisionCfg(
  geom_names_expr=(_collision_regex,),
  contype=1,
  conaffinity=1,
  solref=(0.01, 1),
  # Feet get condim 6 and rolling/torsional friction; the rest are frictionless.
  condim={_foot_regex: 6, _collision_regex: 1},
  priority={_foot_regex: 1, ".*": 0},
  friction={_foot_regex: (1, 5e-3, 5e-4)},
)


Collision = Literal["capsule", "original"]


def get_go2_robot_cfg(
  collision: Collision = "capsule", gains: Gains = "unitree"
) -> EntityCfg:
  """A fresh Go2 robot configuration (new instance each call).

  ``collision="original"`` keeps the MJCF's cylinder and box collision
  shapes; ``"capsule"`` swaps them for capsules/spheres (see ``get_spec``).
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=partial(get_spec, capsules=collision == "capsule"),
    articulation=go2_articulation(gains),
  )
