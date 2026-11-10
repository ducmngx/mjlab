# Probe results: ours vs a fixed-margin severity safety filter

The probes (`failbench/house.py`, `PROBE_*`) are small lane layouts built to
test one claim: a safety filter with a fixed margin cannot fit every space on
a route, while ours adapts. Every arm runs the same planner (Nav2 MPPI,
`loose_path`, oracle map, HIM Go2 policy) and differs only in the added
critic or filter.

Arms:

- `normal`: stock Nav2 MPPI.
- `cbfsem_<m>`: severity-scaled CBF filter after MPPI. Margin `m·sev/25`:
  person 2m, glass m, box 0.32m, wall 0.16m.
- `ours_w<w>`: risk critic (`impact_v5` phase 1, 8 imagined faults) at
  weight `w`.

Faults: one per robot, unaware (the robot keeps replanning), placed between
15% and 85% of the route; damage over a 3 s window. Same seed and robot =
same fault at the same place across arms (paired).

Raw data, scripts and text reports: `benchmark_results/failbench/probes/`
(not in git).

## Corridor (glass on one side, wood on the other), fault-free

Local, 8 robots per cell, both twins (`probe_corridor`, `_swap`), seed 1.
Lean toward the wood inside the corridor (cm) by corridor width:

| Arm | 0.7 m | 0.9 m | 1.2 m | 1.6 m |
|---|---|---|---|---|
| normal | 0 | 0 | 0 | 0 |
| ours w0.5 | 5.0 | 4.4 | 4.5 | 4.1 |
| ours w1 | 4.7 | 5.1 | 5.8 | 5.9 |
| ours w3 | 2.5 | 4.3 | 5.0 | 6.6 |
| cbfsem 0.1 | 0.9 | 0.2 | 0.1 | 0.3 |
| cbfsem 0.2 | 4.0 | 2.1 | 0.8 | -0.7 |
| cbfsem 0.3 | blocked | 5.8 | 2.0 | -0.2 |
| cbfsem 0.4 | blocked | 12.8 | 6.0 | 0 |

All arms reach 100% except cbfsem 0.3 (0% at 0.7 m) and cbfsem 0.4 (0% at
0.7 m, 12% at 0.9 m). Ours w3 slows only in tight corridors (17.9 s at 0.7 m,
14.7 s at 0.9 m vs about 12.4 s for normal). Ours leans a little everywhere
and is never locked out; the filter is inactive or blocks. The lean (about
5 cm) is too small to headline.

## Mixed probe (door beside a person, then a cross slope with a person)

Door gap 0.75 m beside a person (needs m ≤ 0.18 to pass); then a cross slope
with a person 0.6 m off the route, downhill or uphill (needs m > 0.21 by
geometry). Hopper jobs 3359497 (faults, seeds 1-4 × 64 robots = 256 faults
per arm) and 3359498 (fault-free, seeds 1-2 × 64, 60 s).

Fault-free (128 robots per arm; E = door first, W = slope first):

| Arm | Reached | Time (s) | Closest to door person | Closest to slope person, downhill lane |
|---|---|---|---|---|
| normal | 100% | 14.1 | 34-40 cm | 71 cm |
| cbfsem 0.1 | 100% | 14.7 | 44 cm | 70 cm |
| cbfsem 0.2 | 0%, stuck at door | - | 59 cm | - |
| cbfsem 0.3 | 0%, stuck at door | - | 76 cm | - |
| ours w1 | 100% | 16.5 E / 19-21 W | 41-45 cm | 78-90 cm |
| ours w3 | 50%: all E reach, all W stuck at door | 22.5 | 44 cm | 68-93 cm |

With faults (mean ± s.e.):

| Arm | Door: damage | Door: person hit | Slope: damage | Slope: person hit |
|---|---|---|---|---|
| normal | 1.72 ± 0.84 | 5% | 1.41 ± 0.56 | 3% |
| cbfsem 0.1 | 1.65 ± 0.80 | 5% | 1.58 ± 0.57 | 3% |
| cbfsem 0.2 | 0.52 ± 0.31 | 2% | 0.98 ± 0.61 | 2% |
| cbfsem 0.3 | 0.92 ± 0.46 | 2% | 0.96 ± 0.61 | 1% |
| ours w1 | 0.21 ± 0.07 | 0% | 1.31 ± 0.45 | 3% |
| ours w3 | 0.12 ± 0.06 | 0% | 0.98 ± 0.37 | 2% |

cbfsem 0.2 and 0.3 never pass the door, so their damage comes from robots
standing still and is not comparable.

Paired, normal minus ours w1, faults at the door (88 pairs): person hits
+0.05 ± 0.02, damage +1.51 ± 0.85. Same against cbfsem 0.1: +0.05 ± 0.02 and
+1.44 ± 0.79.

Slope faults by lane (`slope_by_lane.txt`): with the person downhill, ours w1
hits the person as often as normal (6% vs 5%, damage 2.14 vs 1.81) despite
12-19 cm more fault-free room. With the person uphill ours w1 is lower (0.43
vs 1.00), but few throws go there.

Conclusions:

- Holds: no fixed margin both passes the door and helps. 0.1 m passes but is
  no safer than stock; 0.2 m and up block every robot. Ours w1 passes both
  ways and, with faults at the door, hits nobody (0% vs 5%).
- Does not hold: protection on the slope. The extra room ours gives is not
  enough against downhill throws.
- Ours w3 is too careful for this door from the slope side (stalls, a soft
  lock-out). Ours w1 costs 15-50% more time than stock here.
- The local 64-robot run (`mixed_local`) suggested the opposite on the
  slope; that was noise (2 person hits out of 42).
- Not measured: whether ours passes the door safely because it slows down
  near the person (it passes at the same distance as cbfsem 0.1).
