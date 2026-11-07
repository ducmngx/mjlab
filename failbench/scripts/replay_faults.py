"""Fork saved runs and replay faults from them: how much damage a fault at
each moment of a route would do, under the plan the robot was running.

For each ``run_mppi.py`` log saved with ``--save-state-every-s``, every
snapshot from ``start_s`` until the robot reached its goal (every
``every_s``) is restored into a fresh env (``failbench/sim_state.py``), one
fault fires at once, and the robot, never told, keeps receiving the
commands it was sent then (``dense_cmd``) for ``window_s``. Its non-foot
contacts are priced against the layout's objects (``failbench/replay.py``).
With ``control`` each moment is also replayed with no fault, which checks
the fork retraces the log (``control_err_m``) and gives the no-fault
baseline.

Faults: ``representative``, the eight the critic imagines
(``REPRESENTATIVE_FAULTS``), or ``all`` 36 (every joint seized, limp or
weak); each carries its mixture odds (``imagined_faults``), so the
prior-weighted mean damage is the expected damage given a fault there.

Writes ``<output>.npz`` (one row per replay) and ``<output>.json`` (per
log: expected damage given a fault, touch rates, per fault).

  uv run python -m failbench.scripts.replay_faults \\
    --logs runs/house_crowd_w0.npz runs/house_crowd_w0.5.npz \\
    --checkpoint-file policies/.../model.pt --output replay/crowd
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import tyro
from heat_bench.envs_mjlab.fault_impact import (
  FAULT_ELEVATION_SENSOR,
  TRUNK_GROUPS,
  body_group,
  body_part,
  tilt_angle,
)
from heat_bench.envs_mjlab.joint_fault_injection import (
  FAULT_CONTACT_SENSOR,
  contacts_in_base_frame,
)
from heat_bench.scripts.policy_loading import load_runner

from failbench.commands import GoalVelocityCommand
from failbench.env_cfg import DEFAULT_TASK, failbench_env_cfg
from failbench.faults import FAULT_TERM, NOMINAL, inject_kinodynamic_fault
from failbench.house import LAYOUTS
from failbench.house_terrain import HOUSE_TERRAINS, route_table
from failbench.mppi.risk_critic import (
  REPRESENTATIVE_FAULTS,
  RiskCriticCfg,
  fault_names,
  imagined_faults,
)
from failbench.replay import ObjectBoxes, price_touches
from failbench.sim_state import restore
from mjlab.envs import ManagerBasedRlEnv


@dataclass
class ReplayConfig:
  logs: tuple[str, ...]
  """``run_mppi.py`` logs (``.npz``, with their ``.json``) saved with
  ``--save-state-every-s``, all on one house terrain."""
  checkpoint_file: str
  output: str
  faults: Literal["representative", "all"] = "representative"
  every_s: float = 0.5
  """Fault onsets this far apart (a multiple of the logs' snapshot spacing)."""
  window_s: float = 3.0
  start_s: float = 2.5
  """The first onset: once the planner has taken over."""
  control: bool = True
  """Also replay each onset with no fault."""
  batch: int = 2048
  task: str = DEFAULT_TASK
  device: str | None = None


def _jobs(log, trials, every: int, start: int, n_faults: int, control: bool):
  """``[J, 3]`` (robot, snapshot, fault; -1 none) to replay from one log:
  every ``every`` steps from ``start`` while the robot hadn't arrived."""
  steps = log["snap_step"]
  rows = []
  for b, tr in enumerate(trials):
    reach = np.inf if tr["time_to_goal_s"] is None else tr["time_to_goal_s"]
    for k, s in enumerate(steps):
      if s < start or (s - start) % every or s * log["dt"] >= reach:
        continue
      for f in range(-1 if control else 0, n_faults):
        rows.append((b, k, f))
  return np.array(rows, dtype=np.int64).reshape(-1, 3)


def main(cfg: ReplayConfig) -> None:
  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
  logs = [np.load(p) for p in cfg.logs]
  terrains = {str(lg["terrain"]) for lg in logs}
  if len(terrains) != 1 or (terrain := terrains.pop()) not in HOUSE_TERRAINS:
    raise ValueError(f"Logs must share one house terrain; got {terrains}.")
  layout = LAYOUTS[HOUSE_TERRAINS[terrain]]
  crit = RiskCriticCfg(
    faults=REPRESENTATIVE_FAULTS if cfg.faults == "representative" else None
  )
  params, prior = imagined_faults(crit, 12)
  names = fault_names(params)

  env_cfg = failbench_env_cfg(cfg.task, terrain, fault_time_s=(1e9, 1e9))
  env_cfg.scene.num_envs = cfg.batch
  # Contacts are priced from the layout: drop FailBench's elevation map
  # (the task's own sensors stay; its critic observations read them).
  env_cfg.scene.sensors = tuple(
    s for s in (env_cfg.scene.sensors or ()) if s.name != FAULT_ELEVATION_SENSOR
  )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  wrapped, runner = load_runner(
    env, device, checkpoint_file=cfg.checkpoint_file, task_id=cfg.task
  )
  policy = runner.get_inference_policy(device=device)
  robot = env.scene["robot"]
  fault = env.event_manager.get_term_cfg(FAULT_TERM).func
  assert isinstance(fault, inject_kinodynamic_fault)
  command = env.command_manager.get_term("twist")
  assert isinstance(command, GoalVelocityCommand)
  sensor = env.scene[FAULT_CONTACT_SENSOR]
  part = [body_group(body_part(g)) for g in sensor.primary_names]
  slot_trunk = torch.tensor(
    [g in TRUNK_GROUPS for g in part], device=device
  ).repeat_interleave(sensor.cfg.num_slots)
  boxes = ObjectBoxes.from_layout(layout, device)
  origin = env.scene.env_origins[:, :3]
  dt = env.step_dt
  window = round(cfg.window_s / dt)
  params_d = params.to(device)
  nominal = torch.tensor(NOMINAL, device=device)

  obs, _ = wrapped.reset()
  table, _ = route_table(HOUSE_TERRAINS[terrain])
  rows_out: dict[str, list] = {
    k: []
    for k in (
      "log",
      "robot",
      "route",
      "onset_s",
      "fault",
      "severity",
      "damage",
      "trunk_hit",
      "touch_s",
      "max_tilt",
      "control_err_m",
    )
  }
  t0 = time.time()
  for li, (path, log) in enumerate(zip(cfg.logs, logs, strict=True)):
    if "snap_step" not in log:
      raise ValueError(f"{path} has no snapshots (run with --save-state-every-s).")
    trials = json.load(open(Path(path).with_suffix(".json")))["trials"]
    spacing = int(np.diff(log["snap_step"]).min()) if len(log["snap_step"]) > 1 else 1
    every = max(spacing, round(cfg.every_s / dt) // spacing * spacing)
    logd = {"snap_step": log["snap_step"], "dt": dt}
    jobs = _jobs(logd, trials, every, round(cfg.start_s / dt), len(names), cfg.control)
    keys = [k for k in log.files if k.startswith("snap_") and k != "snap_step"]
    cmd_log = torch.as_tensor(log["dense_cmd"], device=device)  # [n, T, 3]
    pos_log = torch.as_tensor(log["dense_base_pos_w"], device=device)
    total = cmd_log.shape[1]
    for c0 in range(0, len(jobs), cfg.batch):
      chunk = jobs[c0 : c0 + cfg.batch]
      m = len(chunk)
      pad = np.concatenate([chunk, np.repeat(chunk[:1], cfg.batch - m, 0)])
      b, k, f = (torch.as_tensor(pad[:, i], device=device) for i in range(3))
      # Clear the last chunk's faults; none fires on its own.
      fault.reset(None)
      fault.fault_time_s[:] = float("inf")
      snap = {
        key[5:]: torch.as_tensor(log[key][pad[:, 0], pad[:, 1]], device=device)
        for key in keys
      }
      rows = torch.arange(cfg.batch, device=device)
      for key, v in restore(env, snap, rows).items():
        obs[key] = v
      step0 = torch.as_tensor(log["snap_step"], device=device)[k]
      faulty = f >= 0
      fault.params[:] = torch.where(
        faulty[:, None, None], params_d[f.clamp(min=0)], nominal
      )
      fault.fire_now(faulty)
      command.use_external[:] = True
      sev = torch.zeros(cfg.batch, device=device)
      dmg = torch.zeros(cfg.batch, device=device)
      trunk = torch.zeros(cfg.batch, dtype=torch.bool, device=device)
      touch_s = torch.full((cfg.batch,), float("nan"), device=device)
      tilt = torch.zeros(cfg.batch, device=device)
      err = torch.zeros(cfg.batch, device=device)
      with torch.no_grad():
        for t in range(window):
          at = (step0 + t).clamp(max=total - 1)
          command.external[:] = cmd_log[b, at]
          obs, _, _, _ = wrapped.step(policy(obs))
          d = robot.data
          _, _, contact_w, force_w, valid = contacts_in_base_frame(sensor, robot)
          strong = valid & (force_w.norm(dim=-1) >= 1.0)
          s_, d_ = price_touches(
            boxes,
            contact_w - origin[:, None],
            strong,  # Feet too: the boxes tell floor from object.
            d.root_link_lin_vel_w[:, :2].square().sum(-1),
          )
          new = (s_ > 0) & touch_s.isnan()
          touch_s = torch.where(new, (t + 1) * dt, touch_s)
          sev = torch.maximum(sev, s_)
          dmg = torch.maximum(dmg, d_)
          trunk |= (strong & slot_trunk).any(-1) & (s_ > 0)
          tilt = torch.maximum(tilt, tilt_angle(d.projected_gravity_b))
          # The no-fault fork should retrace the log (until the log ends).
          ref = pos_log[b, at]
          err = torch.maximum(err, (d.root_link_pos_w - ref).norm(dim=-1))
      take = slice(0, m)
      onset = (step0.float() * dt)[take]
      for key, v in (
        ("log", torch.full((m,), li)),
        ("robot", b[take]),
        ("route", b[take] % len(table)),
        ("onset_s", onset),
        ("fault", f[take]),
        ("severity", sev[take]),
        ("damage", dmg[take]),
        ("trunk_hit", trunk[take]),
        ("touch_s", touch_s[take]),
        ("max_tilt", tilt[take]),
        ("control_err_m", torch.where(faulty, float("nan"), err)[take]),
      ):
        rows_out[key].append(v.detach().cpu().numpy())
      print(
        f"[replay] {Path(path).name}: {min(c0 + m, len(jobs))}/{len(jobs)}"
        f" ({time.time() - t0:.0f} s)"
      )
  out = {k: np.concatenate(v) for k, v in rows_out.items()}
  p = Path(cfg.output)
  p.parent.mkdir(parents=True, exist_ok=True)
  arrays: dict[str, np.ndarray] = {
    **out,
    "fault_names": np.array(names),
    "prior": prior.numpy(),
    "logs": np.array(cfg.logs),
  }
  np.savez_compressed(p.with_suffix(".npz"), allow_pickle=False, **arrays)

  per_log: dict[str, dict] = {}
  summary = {
    "terrain": terrain,
    "faults": names,
    "prior": prior.tolist(),
    "logs": per_log,
  }
  w = prior.numpy()
  for li, path in enumerate(cfg.logs):
    sel = out["log"] == li
    fl = sel & (out["fault"] >= 0)
    per = []
    for fi, name in enumerate(names):
      s = fl & (out["fault"] == fi)
      per.append(
        {
          "fault": name,
          "n": int(s.sum()),
          "damage": float(out["damage"][s].mean()) if s.any() else None,
          "touched": float((out["severity"][s] > 0).mean()) if s.any() else None,
        }
      )
    ctl = sel & (out["fault"] < 0)
    per_log[path] = {
      "onsets": int(ctl.sum()) if cfg.control else None,
      "expected_damage_given_fault": float(
        sum(w[i] * (per[i]["damage"] or 0.0) for i in range(len(names)))
      ),
      "touched_given_fault": float(
        sum(w[i] * (per[i]["touched"] or 0.0) for i in range(len(names)))
      ),
      "no_fault_damage": float(out["damage"][ctl].mean()) if ctl.any() else None,
      "control_err_m_p95": float(np.nanpercentile(out["control_err_m"][ctl], 95))
      if ctl.any()
      else None,
      "per_fault": per,
    }
  json.dump(summary, open(p.with_suffix(".json"), "w"), indent=1)
  print(
    json.dumps(
      {
        k: {kk: vv for kk, vv in v.items() if kk != "per_fault"}
        for k, v in per_log.items()
      },
      indent=1,
    )
  )
  env.close()


if __name__ == "__main__":
  main(tyro.cli(ReplayConfig))
