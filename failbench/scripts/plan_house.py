"""Plan and draw the household layouts' global routes (``failbench/house.py``).

Each layout is rasterised into the planners' elevation grid, turned into a
Nav2 costmap the way the runner does (``nav2_costmap``: height jumps over
0.18 m are obstacles, then inflation), and each route planned on it with the
NavFn-style global planner. Writes ``<out>/<layout>.png`` (ground height,
objects by class, impassable cells, routes) and prints each route's length,
climb and clearance to glass and people.

  uv run python -m failbench.scripts.plan_house --out routes/
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro
from matplotlib.patches import Rectangle

from failbench.house import CLASSES, LAYOUTS, Layout, plan_routes
from failbench.mppi.nav2 import INSCRIBED

COLORS = {"wall": "#77736d", "box": "#b88c59", "glass": "#5fb4e6", "person": "#e05555"}


@dataclass
class PlanHouseConfig:
  out: str
  layouts: tuple[str, ...] = tuple(LAYOUTS)
  resolution: float = 0.05


def clearance(path: np.ndarray, layout: Layout, cls: str) -> float:
  """Least distance (m) in the plane from the path to an object of ``cls``."""
  best = np.inf
  for it in layout.items:
    if it.cls != cls:
      continue
    dx = np.maximum(np.maximum(it.x0 - path[:, 0], 0), path[:, 0] - it.x1)
    dy = np.maximum(np.maximum(it.y0 - path[:, 1], 0), path[:, 1] - it.y1)
    best = min(best, float(np.hypot(dx, dy).min()))
  return best


def passed(path: np.ndarray, layout: Layout, within: float = 1.0) -> dict[str, int]:
  """How many objects of each class the path passes within ``within`` m."""
  out: dict[str, int] = {}
  for it in layout.items:
    if it.cls == "floor":
      continue
    dx = np.maximum(np.maximum(it.x0 - path[:, 0], 0), path[:, 0] - it.x1)
    dy = np.maximum(np.maximum(it.y0 - path[:, 1], 0), path[:, 1] - it.y1)
    if np.hypot(dx, dy).min() <= within:
      out[it.cls] = out.get(it.cls, 0) + 1
  return out


def draw(layout: Layout, res: float, out: Path) -> list[dict]:
  paths, top, cls, xs, ys, cost = plan_routes(layout, res)
  ext = (xs[0] - res / 2, xs[-1] + res / 2, ys[0] - res / 2, ys[-1] + res / 2)
  ground = np.where(cls < 0, top, np.nan)
  fig, ax = plt.subplots(figsize=(13, 10))
  im = ax.imshow(ground, origin="lower", extent=ext, cmap="YlGn_r", vmin=-0.4, vmax=1.0)
  fig.colorbar(im, ax=ax, shrink=0.6, label="ground height (m)")
  for it in layout.items:
    if it.cls == "floor":
      if it.label == "stair":
        ax.add_patch(
          Rectangle(
            (it.x0, it.y0),
            it.x1 - it.x0,
            it.y1 - it.y0,
            fill=False,
            ec="k",
            lw=0.6,
            zorder=2,
          )
        )
      continue
    ax.add_patch(
      Rectangle(
        (it.x0, it.y0), it.x1 - it.x0, it.y1 - it.y0, color=COLORS[it.cls], zorder=3
      )
    )
    if it.label:
      ax.text(
        (it.x0 + it.x1) / 2,
        (it.y0 + it.y1) / 2,
        it.label,
        ha="center",
        va="center",
        fontsize=6.5,
        zorder=4,
      )
  labelled = set()
  for it in layout.items:  # Level names, once each.
    if it.cls == "floor" and it.label not in ("stair", "") and it.label not in labelled:
      labelled.add(it.label)
      ax.text(
        it.x0 + 0.1,
        it.y1 - 0.15,
        it.label,
        fontsize=7,
        style="italic",
        va="top",
        color="#333",
        zorder=2,
      )
  # Impassable to the body (lethal or inscribed) but not an object: level
  # edges, the costmap's view of a drop.
  edge = (cost >= INSCRIBED) & (cls < 0)
  ax.imshow(
    np.where(edge, 1.0, np.nan),
    origin="lower",
    extent=ext,
    cmap="Greys",
    vmin=0,
    vmax=1.2,
    alpha=0.9,
    zorder=2,
  )
  cmap = plt.get_cmap("tab10")
  stats = []
  for k, ((a, b), p) in enumerate(zip(layout.routes, paths, strict=True)):
    name = f"{k + 1}. {a} → {b}"
    if p is None:
      print(f"  {name}: NO PATH")
      stats.append(dict(route=name, found=False))
      continue
    length = float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())
    ci = np.clip(((p[:, 0] - xs[0]) / res).round().astype(int), 0, len(xs) - 1)
    ri = np.clip(((p[:, 1] - ys[0]) / res).round().astype(int), 0, len(ys) - 1)
    z = top[ri, ci]
    s = dict(
      route=name,
      found=True,
      length=length,
      climb=float(z.max() - z.min()),
      glass=clearance(p, layout, "glass"),
      person=clearance(p, layout, "person"),
      near=passed(p, layout),
    )
    stats.append(s)
    ax.plot(
      p[:, 0],
      p[:, 1],
      color=cmap(k),
      lw=2.2,
      zorder=5,
      label=f"{name} ({length:.1f} m)",
    )
    ax.annotate(
      "",
      xy=p[-1],
      xytext=p[max(0, len(p) - 8)],
      zorder=6,
      arrowprops=dict(arrowstyle="-|>", color=cmap(k), lw=2),
    )
    ax.text(
      p[0, 0] + 0.12 * (k % 3 - 1),
      p[0, 1] + 0.12 * (k % 2),
      str(k + 1),
      color="white",
      fontsize=8,
      ha="center",
      va="center",
      zorder=7,
      bbox=dict(boxstyle="circle,pad=0.2", fc=cmap(k), ec="none"),
    )
  for x, y in layout.places.values():
    ax.plot(x, y, "k.", ms=4, zorder=6)
  w, h = layout.size
  ax.set_xlim(-0.3, w + 0.3)
  ax.set_ylim(-0.3, h + 0.3)
  ax.set_aspect("equal")
  ax.legend(
    loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=2, fontsize=9, frameon=False
  )
  ax.set_title(
    f"{layout.name}: global routes on the elevation costmap "
    "(black: level edges the body can't cross)"
  )
  fig.tight_layout()
  fig.savefig(out, dpi=100)
  plt.close(fig)
  return stats


def main(cfg: PlanHouseConfig) -> None:
  out = Path(cfg.out)
  out.mkdir(parents=True, exist_ok=True)
  assert set(CLASSES) == set(COLORS)
  for name in cfg.layouts:
    print(name)
    for s in draw(LAYOUTS[name], cfg.resolution, out / f"{name}.png"):
      if s["found"]:
        print(
          f"  {s['route']:30s} {s['length']:5.1f} m  climb {s['climb']:4.2f} m"
          f"  glass {s['glass']:4.2f} m  person {s['person']:4.2f} m"
          f"  within 1 m: {s['near'].get('glass', 0)} glass,"
          f" {s['near'].get('person', 0)} people, {s['near'].get('box', 0)} furniture"
        )


if __name__ == "__main__":
  main(tyro.cli(PlanHouseConfig))
