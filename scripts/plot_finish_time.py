"""Plot time-to-drive-1-km per agent, one panel per density.

The headline chart since Week 10b. With safety-checked lane changes every
agent is collision-free, and episode return is flat in speed on a
fixed-distance road (Project-Documentation/concepts/02-reward-shaping.md),
so finish time is the comparison that means something.

Dot plot, not bars: the agents differ by 1-3 s around ~30 s, which a
zero-based bar would flatten. One emphasised series (PPO, blue) against
grey context; whiskers are 95% CIs of the mean over seeds; only PPO's
values are labelled.

Run:
    uv run python scripts/plot_finish_time.py \\
        --csv results/ctrl_eval_matrix_existing_episodes.csv \\
        --out docs/images/finish_time.png
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path

# Display order (top to bottom) and labels.
_AGENTS: tuple[tuple[str, str], ...] = (
    ("ppo", "PPO"),
    ("safe_rule", "Safe-rule"),
    ("greedy", "Greedy"),
    ("dqn", "DQN"),
    ("random", "Random"),
)
_DENSITIES: tuple[str, ...] = ("low", "medium", "high")

# Emphasis palette: one hue + de-emphasis grey (validated: blue passes
# lightness / CVD / contrast on the light surface).
_SURFACE = "#fcfcfb"
_EMPHASIS = "#2a78d6"
_CONTEXT = "#8d8b86"
_TEXT_PRIMARY = "#0b0b0b"
_TEXT_SECONDARY = "#52514e"
_GRID = "#e4e3df"


def _read_finish_times(path: Path) -> dict[tuple[str, str], list[float]]:
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open() as f:
        for r in csv.DictReader(f):
            out[(r["agent"], r["density"])].append(float(r["episode_length_s"]))
    return out


def _mean_ci95(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    m = sum(xs) / n
    if n < 2:
        return m, 0.0
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    return m, 1.96 * sd / math.sqrt(n)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--csv", type=Path,
                   default=Path("results/ctrl_eval_matrix_existing_episodes.csv"))
    p.add_argument("--out", type=Path, default=Path("docs/images/finish_time.png"))
    args = p.parse_args(argv)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    times = _read_finish_times(args.csv)
    stats = {k: _mean_ci95(v) for k, v in times.items()}
    lo = min(m - ci for m, ci in stats.values())
    hi = max(m + ci for m, ci in stats.values())
    pad = 0.4

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True,
                             facecolor=_SURFACE)
    for ax, density in zip(axes, _DENSITIES, strict=True):
        ax.set_facecolor(_SURFACE)
        for row, (key, _label) in enumerate(_AGENTS):
            m, ci = stats[(key, density)]
            y = len(_AGENTS) - 1 - row
            color = _EMPHASIS if key == "ppo" else _CONTEXT
            ax.plot([m - ci, m + ci], [y, y], color=color, linewidth=2,
                    solid_capstyle="round", zorder=2)
            ax.scatter([m], [y], s=64, color=color, edgecolors=_SURFACE,
                       linewidths=2, zorder=3)
            if key == "ppo":
                ax.annotate(f"{m:.1f} s", (m, y), xytext=(0, 9),
                            textcoords="offset points", ha="center",
                            fontsize=9, color=_TEXT_PRIMARY)
        ax.set_title(f"{density} density", fontsize=10, color=_TEXT_PRIMARY,
                     loc="left")
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(-0.6, len(_AGENTS) - 0.4)
        ax.grid(axis="x", color=_GRID, linewidth=1)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(_GRID)
        ax.tick_params(colors=_TEXT_SECONDARY, labelsize=9, length=0)
        ax.set_xlabel("seconds to drive 1 km", fontsize=9, color=_TEXT_SECONDARY)

    axes[0].set_yticks(range(len(_AGENTS)))
    axes[0].set_yticklabels([label for _k, label in reversed(_AGENTS)])
    fig.suptitle(
        "Time to drive 1 km, lower is faster  ·  mean of 20 seeds, 95% CI  ·  "
        "0% collisions for every agent",
        x=0.01, ha="left", fontsize=11, color=_TEXT_PRIMARY,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=160, facecolor=_SURFACE)
    print(f"✓ wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
