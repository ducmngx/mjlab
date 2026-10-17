"""Registers the Go2 HIMLoco tasks with mjlab. Import this module before
using mjlab's train/play entry points (``go2_him/scripts`` does).

- ``Go2-HIM-Rough``: HIMLoco's setup with capsule collision shapes, 100 CCD
  iterations and Unitree's Go2 gains; fits 2048 envs in an 8 GB GPU.
- ``Go2-HIM-Rough-Original``: as close to HIMLoco as this port gets: the Go2
  MJCF's own collision shapes, mjlab's 500 CCD iterations and HIMLoco's PD
  gains (Kp 40, Kd 1). Needs a larger GPU.
"""

from go2_him.him.env_cfg import go2_him_env_cfg
from go2_him.him.rl_cfg import go2_him_runner_cfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task
from mjlab.tasks.velocity.rl import VelocityOnPolicyRunner

TASK_ID = "Go2-HIM-Rough"
ORIGINAL_TASK_ID = "Go2-HIM-Rough-Original"


def _original_cfg(play: bool = False):
  return go2_him_env_cfg(
    play=play, collision="original", gains="himloco", ccd_iterations=500
  )


if TASK_ID not in list_tasks():
  register_mjlab_task(
    task_id=TASK_ID,
    env_cfg=go2_him_env_cfg(),
    play_env_cfg=go2_him_env_cfg(play=True),
    rl_cfg=go2_him_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
if ORIGINAL_TASK_ID not in list_tasks():
  register_mjlab_task(
    task_id=ORIGINAL_TASK_ID,
    env_cfg=_original_cfg(),
    play_env_cfg=_original_cfg(play=True),
    rl_cfg=go2_him_runner_cfg(),
    runner_cls=VelocityOnPolicyRunner,
  )
