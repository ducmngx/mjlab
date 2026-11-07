# Household fault experiment

Does the risk critic make a legged robot safer when a joint fault hits, and
what does it cost? Every arm drives the same routes with the same planner;
each robot suffers one fault at a place on its route, keeps replanning
without being told, and its safety is measured over the 3 s after the fault,
the critic's horizon. New baselines are new arms: they reuse every existing
arm's results, so nothing here needs rerunning (see "Adding an arm").

## Scenes and routes

Eight layouts in `failbench/house.py` (built by `failbench/house_terrain.py`),
eight routes each, 64 routes in all:

| Layout | What it tests |
|---|---|
| `apartment` | Flat flat; glass, people and furniture beside the routes |
| `loft` | Split level: an open-sided stair to a 0.6 m platform behind glass |
| `garden` | Rough lawn, a mound with ~10° slopes, a patio step and ramp |
| `townhouse` | A six-rise stair to a 0.9 m study behind glass rails, a ramp |
| `lobby` | Open hall, everything a metre or more off the routes (control) |
| `crowd` | Open hall, routes through a group of six people |
| `hall` | 11 m of open floor, then a group across the walking line |
| `side_crowd` | A group to one side of the routes, open floor on the other |

Severities (`failbench/severity.py`): person 50, glass 25, furniture 8,
wall 4, floor 0. Each route's global path is planned once, offline, by the
NavFn-style planner on the Nav2 costmap of the layout's elevation grid
(`house.plan_routes`), and is the same for every arm. Robot `i` drives route
`i % 8`, starting at its start facing along its path.

## Planner

Nav2 MPPI (`failbench/mppi/nav2.py`), profile `loose_path`: stock except
path align weight 3 (stock 14). Planning map: the true map (`--sweep-map
oracle`). The robot stands until the planner takes over at 2.5 s.

## Arms

| Arm | Flags beyond the shared ones |
|---|---|
| `normal` | `--risk.weight 0` (the critic only logs its predictions) |
| `ours_w0.5` | `--risk.weight 0.5` |
| `ours_w1` | `--risk.weight 1` |
| `inflate_<m>` (baseline) | `--risk.weight 0 --nav2.costmap.margin <m>` |
| `semantic_<m>` (baseline) | `--risk.weight 0 --nav2.costmap.margin <m> --nav2.costmap.severity-ref 25` |

Ours imagines the eight representative faults (`REPRESENTATIVE_FAULTS` in
`failbench/mppi/risk_critic.py`) at their mixture odds, with the impact model
`impact_v5` phase 1 (`/scratch/mnguy21/failbench/models/impact_v5/phase1/best.pt`).
Weights 0.5 and 1 come from the fault-free weight sweep (0.5: 4% slower on
average for 9% of robots within 50 cm of a person against 41%; 1: 11% slower,
3%).

## Faults

- One fault per robot from the eight representative faults, each equally
  often over a run's 64 robots (`--family priced`, balanced): front hips
  limp, front calves seized, rear thighs limp, rear thighs seized.
- It fires once the robot has covered a fraction of its route's straight
  start-to-goal length, stratified over each route's eight robots in
  15-85% (`--fault-place-frac 0.15 0.85`); a robot that never gets there
  faults at 30 s (`--fault-timeout-s 30`).
- The fault and its place depend only on the seed, so every arm meets the
  same fault at the same place: results pair by (layout, seed, robot).
- After the fault the robot is not told; the planner keeps replanning as
  usual (`--post-fault replan --no-fault-aware`).

## Size

4 seeds (1-4) x 64 robots = 256 faults per layout and arm, 2,048 per arm;
40 s per run (`--duration-s 40`).

## Metrics

Per robot, over the window from the fault to 3 s after it
(`failbench.replay.FaultWindow`, each trial's `window` in the run JSON).
Contacts are the robot's contact-sensor contacts (feet included) with at
least 1 N of force, matched to the layout's object boxes: within 2 cm of a
box and at least 2 cm above its base, so the floor never counts.

Reported (the paper's):

1. **Collision**: did the robot touch any object in the window
   (`collided`); also a person (`collided_person`) and glass
   (`collided_glass`).
2. **Distance**: the distance from the robot's nearest link origin to the
   nearest object's surface at the end of the window (`end_dist_m`), and to
   the nearest person (`end_dist_person_m`); also the least over the window
   (`min_dist_m`, `min_dist_person_m`). Link origins sit inside the body, so
   these read a few centimetres high next to the surface itself.
3. **Force**: the peak contact force on an object in the window
   (`peak_force_n`; on a person, `peak_force_person_n`), and the impulse
   (`impulse_ns`: the largest object force each step, summed over the
   window), which is steadier than a single step's peak.

Also recorded, recommended for the paper:

4. **Damage** (`damage`): severity x (1 - exp(-E/E0)) of the worst object
   touched, E the trunk's planar kinetic energy then, as the critic prices a
   hit; the one number that weighs a person above a sofa.
5. **Impact speed** (`impact_speed`, m/s) and **time to first contact**
   (`first_contact_s`): how hard and how soon.
6. **Fall** (`fell`): the robot went down in the window.
7. **Near miss**: `min_dist_person_m` under 0.1 m without a person
   collision (from the recorded distances).
8. **Efficiency**, from the fault-free weight sweep on the same layouts:
   time to goal against `normal`, success rate, path error.

Aggregation (per arm, pooled and per layout): rates and means over robots,
with the paired difference to `normal` (same layout, seed and robot) and its
standard error over robots; a window that didn't complete (the run ended
first; `complete` false) is left out.

## Running

From `~/mjlab` on Hopper, after syncing `failbench/`:

```
ARMS="normal ours_w0.5 ours_w1" sbatch --array=0-23 failbench/cluster/house_faults.sbatch
```

Task `i` runs layout `i % 8` with arm `i // 8` for every seed; finished
runs are skipped, so resubmitting fills gaps. Outputs:
`/scratch/mnguy21/failbench/mppi/house_faults/<arm>/<layout>_s<seed>.{json,npz}`.

## Adding an arm

Name it, give it a `case` entry in `house_faults.sbatch` if it needs new
flags, and submit only it (`ARMS="inflate_0.3" sbatch --array=0-7 ...`).
Keep every shared setting above unchanged (planner profile, map, fault
family and placement, window, seeds, robots, duration), or the pairing
breaks and the arm must be compared on its own.

## Caveats

- The GPU simulation is not bit-for-bit repeatable: the same seed gives the
  same fault at the same place, but the outcome varies run to run. Arms are
  compared over many paired faults, not trial by trial.
- The fault fires by place along the straight start-to-goal line
  (`route_progress`); on curved routes that is approximate, but the same for
  every arm.
- Falls off a level edge (drops) are not priced in the window metrics; most
  edges in these layouts are railed.
