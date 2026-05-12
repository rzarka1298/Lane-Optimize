"""Plot single-agent PPO vs parameter-sharing MAPPO comparison.

Two side-by-side panels:

- LEFT: per-density mean return. Single-agent PPO (1 ego at a time) vs
  MAPPO mean per-ego (4 egos at once). MAPPO is naturally lower because
  4 egos compete for the same lanes — comparison is informative but
  NOT apples-to-apples.

- RIGHT: per-density collision rate. Same caveat: MAPPO collision rate
  counts ego-episodes (5 seeds × 4 egos = 20), so a 5% MAPPO rate means
  "1 of 20 ego-episodes ended in collision."

Run:
    uv run python scripts/plot_mappo_comparison.py \\
        --ppo-csv results/week4_eval_matrix.csv \\
        --mappo-csv results/week5_mappo_eval.csv \\
        --out docs/images/mappo_vs_ppo.png
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def _read_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def _ppo_per_density(rows: list[dict]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        if r["agent"] != "ppo":
            continue
        out[r["density"]] = {
            "return": float(r["return_total_mean"]),
            "return_std": float(r["return_total_std"]),
            "collision_rate": float(r["collision_rate"]),
        }
    return out


def _mappo_per_density(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Aggregate per-ego rows into per-density stats."""
    by_d: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_d[r["density"]].append(r)

    out: dict[str, dict[str, float]] = {}
    for density, drows in by_d.items():
        returns = [float(r["return_total"]) for r in drows]
        n_collisions = sum(1 for r in drows if r["collided"].lower() == "true")
        mean = sum(returns) / len(returns)
        var = sum((x - mean) ** 2 for x in returns) / len(returns)
        out[density] = {
            "return": mean,
            "return_std": var ** 0.5,
            "collision_rate": n_collisions / len(drows),
            "n": len(drows),
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ppo-csv", type=Path, required=True,
                        help="aggregated single-agent PPO eval (week4_eval_matrix.csv)")
    parser.add_argument("--mappo-csv", type=Path, required=True,
                        help="per-ego MAPPO eval (week5_mappo_eval.csv)")
    parser.add_argument("--out", type=Path,
                        default=Path("docs/images/mappo_vs_ppo.png"))
    args = parser.parse_args(argv)

    import matplotlib.pyplot as plt
    import numpy as np

    ppo = _ppo_per_density(_read_csv(args.ppo_csv))
    mappo = _mappo_per_density(_read_csv(args.mappo_csv))

    densities = ["low", "medium", "high"]
    x = np.arange(len(densities))
    width = 0.35

    fig, (axR, axC) = plt.subplots(1, 2, figsize=(12, 5), dpi=120)

    # ---- Returns panel
    ppo_ret = [ppo[d]["return"] for d in densities]
    ppo_ret_std = [ppo[d]["return_std"] for d in densities]
    mappo_ret = [mappo[d]["return"] for d in densities]
    mappo_ret_std = [mappo[d]["return_std"] for d in densities]

    axR.bar(x - width / 2, ppo_ret, width, yerr=ppo_ret_std, capsize=4,
            color="#1f77b4", label="Single-agent PPO (1 ego)", alpha=0.85,
            edgecolor="black", linewidth=0.7)
    axR.bar(x + width / 2, mappo_ret, width, yerr=mappo_ret_std, capsize=4,
            color="#ff7f0e", label="MAPPO mean per-ego (4 egos)", alpha=0.85,
            edgecolor="black", linewidth=0.7)
    axR.set_xticks(x)
    axR.set_xticklabels(densities)
    axR.set_ylabel("Mean episode return")
    axR.set_title("Per-density mean return")
    axR.legend(loc="upper right", fontsize=9)
    axR.grid(True, axis="y", alpha=0.3)
    axR.axhline(0, color="gray", linewidth=0.5, alpha=0.5)

    # ---- Collision rate panel
    ppo_col = [ppo[d]["collision_rate"] * 100 for d in densities]
    mappo_col = [mappo[d]["collision_rate"] * 100 for d in densities]
    axC.bar(x - width / 2, ppo_col, width,
            color="#1f77b4", label="Single-agent PPO (1 ego)", alpha=0.85,
            edgecolor="black", linewidth=0.7)
    axC.bar(x + width / 2, mappo_col, width,
            color="#ff7f0e", label="MAPPO per-ego (4 egos / scene)",
            alpha=0.85, edgecolor="black", linewidth=0.7)
    axC.set_xticks(x)
    axC.set_xticklabels(densities)
    axC.set_ylabel("Collision rate (%)")
    axC.set_title("Per-density per-ego collision rate")
    axC.legend(loc="upper left", fontsize=9)
    axC.grid(True, axis="y", alpha=0.3)
    axC.set_ylim(0, max(max(ppo_col), max(mappo_col)) * 1.4 + 5)

    fig.suptitle("Single-agent PPO vs parameter-sharing MAPPO",
                 fontsize=13, fontweight="bold")
    fig.text(
        0.5, 0.01,
        "Caveat: MAPPO has 4 egos in one scene (harder problem). The relevant "
        "MAPPO finding is that the shared policy keeps multi-agent traffic\n"
        "manageable at all 3 densities, not that it beats single-agent PPO "
        "on per-ego return.",
        ha="center", fontsize=8, style="italic", color="#444",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    fig.savefig(args.out, bbox_inches="tight")
    print(f"✓ wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
