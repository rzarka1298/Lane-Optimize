"""Generate the Week-4 hero Pareto plot from a reward-sweep CSV.

Plots collision rate (x) vs mean return (y); one point per
(reward_config, density) cell. Connects same-config points with a line
so the safety/speed *frontier* per config is visible.

Run:
    uv run python scripts/plot_pareto.py \\
        --csv results/week4_reward_sweep.csv \\
        --out docs/images/pareto_safety_speed.png

Optionally overlay the rule-based baselines for context:
    --baselines-csv results/week4_eval_matrix.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _by_agent(
    rows: list[dict[str, str]],
) -> dict[str, list[tuple[str, float, float]]]:
    """Returns {agent: [(density, collision_rate, return_total), ...]}."""
    out: dict[str, list[tuple[str, float, float]]] = defaultdict(list)
    for r in rows:
        out[r["agent"]].append((
            r["density"],
            float(r["collision_rate"]),
            float(r["return_total_mean"]),
        ))
    # Sort each list low → medium → high so plot lines go in a sensible direction.
    order = {"low": 0, "medium": 1, "high": 2}
    for agent in out:
        out[agent].sort(key=lambda t: order.get(t[0], 99))
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Plot the Pareto frontier.")
    parser.add_argument("--csv", type=Path, required=True,
                        help="sweep CSV (one row per (agent, density))")
    parser.add_argument("--baselines-csv", type=Path, default=None,
                        help="optional second CSV to overlay (e.g., rule baselines)")
    parser.add_argument("--out", type=Path, default=Path("docs/images/pareto_safety_speed.png"),
                        help="output PNG path")
    parser.add_argument("--title", type=str, default="LaneIQ — safety vs return Pareto frontier")
    args = parser.parse_args(argv)

    # matplotlib only here so the rest of the project doesn't pull it on import.
    import matplotlib.pyplot as plt

    sweep = _by_agent(_read_csv(args.csv))
    baselines = (
        _by_agent(_read_csv(args.baselines_csv)) if args.baselines_csv else {}
    )

    fig, ax = plt.subplots(figsize=(9, 6), dpi=120)

    # Marker style per density to make the density-trajectory legible.
    density_marker = {"low": "o", "medium": "s", "high": "^"}
    sweep_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                     "#8c564b"]
    baseline_palette = ["#7f7f7f", "#bcbd22", "#17becf", "#e377c2"]

    # Plot rule-based baselines first so they sit behind PPO/DQN traces.
    for i, (agent, points) in enumerate(sorted(baselines.items())):
        if agent in sweep:
            continue  # don't double-draw if also in sweep
        col = baseline_palette[i % len(baseline_palette)]
        xs = [p[1] * 100 for p in points]  # collision rate -> percentage
        ys = [p[2] for p in points]
        ax.plot(xs, ys, color=col, linewidth=1, alpha=0.5, linestyle="--")
        for density, x, y in zip(
            [p[0] for p in points], xs, ys, strict=True,
        ):
            ax.scatter(x, y, s=80, marker=density_marker.get(density, "x"),
                       color=col, alpha=0.5, edgecolors="black", linewidths=0.5,
                       label=f"{agent} ({density})" if density == "medium" else None)

    # Plot sweep agents in bold.
    for i, (agent, points) in enumerate(sweep.items()):
        col = sweep_palette[i % len(sweep_palette)]
        xs = [p[1] * 100 for p in points]
        ys = [p[2] for p in points]
        ax.plot(xs, ys, color=col, linewidth=1.5, alpha=0.85)
        for density, x, y in zip(
            [p[0] for p in points], xs, ys, strict=True,
        ):
            ax.scatter(x, y, s=140, marker=density_marker.get(density, "x"),
                       color=col, edgecolors="black", linewidths=1.0,
                       label=f"{agent} ({density})" if density == "medium" else None)

    ax.set_xlabel("Collision rate (%)", fontsize=12)
    ax.set_ylabel("Mean episode return", fontsize=12)
    ax.set_title(args.title, fontsize=13)

    # Frontier visual: top-left corner is the goal.
    ax.axvline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.axhline(0, color="gray", linewidth=0.5, alpha=0.3)
    ax.grid(True, alpha=0.2)

    # Compact legend — one entry per agent (used the medium-density point above).
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, loc="lower left", fontsize=9,
                  framealpha=0.85, title="agent (marker=density)")

    # Sub-caption text bottom-right explaining markers.
    ax.text(
        0.99, 0.02,
        "Marker: ○ low  ■ medium  ▲ high\nLine connects same-agent densities",
        transform=ax.transAxes, fontsize=8, ha="right", va="bottom",
        bbox={"boxstyle": "round,pad=0.4", "facecolor": "white",
              "edgecolor": "gray", "alpha": 0.85},
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(args.out, bbox_inches="tight")
    print(f"✓ wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
