"""Turn the per-impulse eval tables into the result.

Reads eval_warned.csv and eval_unwarned.csv, produces the survival-vs-force
comparison and a plot. Uses only stdlib plus matplotlib so it runs anywhere.

Reports a Wilson score interval on each fall rate rather than a bare percentage:
with a few hundred impulses per cell the difference between robots can easily be noise,
and a comparison that does not say so is not a result.

Usage:
  python scripts/analyze.py --results-dir . --out survival.png
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

REQUIRED = ("warned", "unwarned")
# The optional third series is the warned robot with its warning switched off: the same
# weights, threat channel zeroed. It separates "the warning does not help" from "the robot
# never learned to use it", so it is plotted alongside whenever it exists.
PRETTY = {
  "warned": "warned",
  "unwarned": "not warned",
  "warning_removed": "warned, then warning switched off",
}
COLOURS = {
  "warned": "#d62728",
  "unwarned": "#1f77b4",
  "warning_removed": "#7f7f7f",
}


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float, float]:
  """Wilson score interval for a binomial proportion. Returns (p, lo, hi)."""
  if n == 0:
    return 0.0, 0.0, 0.0
  p = k / n
  d = 1 + z * z / n
  centre = (p + z * z / (2 * n)) / d
  half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
  return p, max(0.0, centre - half), min(1.0, centre + half)


def load(path: Path) -> dict[float, list[dict]]:
  by_force: dict[float, list[dict]] = defaultdict(list)
  with path.open() as f:
    for row in csv.DictReader(f):
      by_force[float(row["force_n"])].append(row)
  return by_force


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--results-dir", default=".")
  p.add_argument("--out", default="survival.png")
  a = p.parse_args()

  root = Path(a.results_dir)
  for robot in REQUIRED:
    if not (root / f"eval_{robot}.csv").exists():
      raise SystemExit(f"missing eval_{robot}.csv; run eval.py for the {robot} robot first")
  robots = [f.stem[len("eval_"):] for f in sorted(root.glob("eval_*.csv"))]
  data = {robot: load(root / f"eval_{robot}.csv") for robot in robots}

  forces = sorted(set().union(*(set(d) for d in data.values())))

  head = f"{'force':>7} | " + " | ".join(f"{robot:^28}" for robot in robots) + " |"
  sub = f"{'(N)':>7} | " + " | ".join(f"{'fall%  [95% CI]      n':^28}" for _ in robots) + " |"
  print("\n" + head)
  print(sub)
  print("-" * len(head))

  summary = {robot: [] for robot in robots}
  for force in forces:
    cells = []
    for robot in robots:
      rows = data[robot].get(force, [])
      n = len(rows)
      k = sum(r["fell"] == "True" for r in rows)
      prop, lo, hi = wilson(k, n)
      summary[robot].append((force, prop, lo, hi, n))
      cells.append(f"{100 * prop:5.1f}  [{100 * lo:4.1f},{100 * hi:5.1f}]  {n:6d}")
    print(f"{force:7.0f} | " + " | ".join(f"{c:^28}" for c in cells) + " |")

  # Mean peak tilt is the secondary signal: even where both robots stay up, the
  # warned one should be disturbed less if the warning is being used.
  print(f"\n{'force':>7} | mean peak tilt (deg)")
  print(f"{'(N)':>7} | " + " ".join(f"{robot:>20}" for robot in robots))
  print("-" * (10 + 21 * len(robots)))
  for force in forces:
    vals = []
    for robot in robots:
      rows = data[robot].get(force, [])
      vals.append(
        sum(float(r["peak_tilt_deg"]) for r in rows) / max(len(rows), 1)
      )
    print(f"{force:7.0f} | " + " ".join(f"{v:20.1f}" for v in vals))

  try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
  except ImportError:
    print("\nmatplotlib unavailable; table only")
    return

  fig, ax = plt.subplots(figsize=(7, 4.5))
  for robot in robots:
    colour = COLOURS.get(robot, "#2ca02c")
    xs = [s[0] for s in summary[robot]]
    ys = [100 * s[1] for s in summary[robot]]
    lo = [100 * (s[1] - s[2]) for s in summary[robot]]
    hi = [100 * (s[3] - s[1]) for s in summary[robot]]
    ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, label=PRETTY.get(robot, robot), color=colour)
  ax.set_xlabel("shove strength (N)")
  ax.set_ylabel("falls per shove (%)")
  ax.set_title("Does warning the robot help it stay up?\nbars are 95% confidence intervals")
  ax.grid(alpha=0.3)
  ax.legend()
  fig.tight_layout()
  fig.savefig(a.out, dpi=150)
  print(f"\nwrote {a.out}")


if __name__ == "__main__":
  main()
