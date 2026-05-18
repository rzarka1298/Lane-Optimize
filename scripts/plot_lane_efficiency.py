"""Lane-occupancy + ego-trajectory heatmap from one or more replay JSONs.

Per master plan §4.5 bonus: visualize how the AI uses lane capacity
over an episode. For each replay, renders a panel where:

  - x-axis = sim time (s)
  - y-axis = lane (0 = right/slow, 2 = left/fast)
  - background heatmap = vehicle count per (lane, time-bin)
  - overlay = ego's lane trajectory (gold) with markers on lane changes

Side-by-side panels make agent comparison easy: a careful policy
hugs an open lane; an aggressive one weaves through traffic; a
failed policy ends abruptly with a red collision marker.

Usage:
    uv run python -m scripts.plot_lane_efficiency \
        --in results/replays/ppo_v2_medium_seed42.json \
             results/replays/ppo_v2_high_seed0.json \
             results/replays/ppo_v1_high_seed7_collision.json \
        --out docs/images/lane_efficiency.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

N_LANES = 3
TIME_BIN_S = 0.5  # bin width for the heatmap (5 ticks)


def load_replay(path: Path) -> dict:
    return json.loads(path.read_text())


def render_panel(
    ax: plt.Axes,
    replay: dict,
    title: str,
) -> None:
    ticks = replay["ticks"]
    if not ticks:
        ax.text(0.5, 0.5, "empty replay", ha="center", va="center")
        return

    times = np.array([t["t"] for t in ticks])
    t_max = float(times[-1])
    n_bins = max(1, int(np.ceil(t_max / TIME_BIN_S)))

    # Per-(lane, time-bin) vehicle count.
    occupancy = np.zeros((N_LANES, n_bins))
    counts_per_bin = np.zeros(n_bins)

    for tick in ticks:
        bin_idx = min(n_bins - 1, int(tick["t"] / TIME_BIN_S))
        counts_per_bin[bin_idx] += 1
        for v in tick["vehicles"]:
            lane = max(0, min(N_LANES - 1, int(v["lane"])))
            occupancy[lane, bin_idx] += 1
    # Normalize to mean per-tick count (otherwise every bin scales by tick count).
    nonzero = counts_per_bin > 0
    occupancy[:, nonzero] /= counts_per_bin[nonzero]

    # Ego trajectory.
    ego_lane = []
    ego_t = []
    for tick in ticks:
        ego = next((v for v in tick["vehicles"] if v["is_ego"]), None)
        if ego is None:
            continue
        ego_t.append(tick["t"])
        ego_lane.append(ego["lane"])
    ego_t = np.array(ego_t)
    ego_lane = np.array(ego_lane)

    # Heatmap: rows = lane (with lane 2 = top to match the canvas), cols = time bin.
    # imshow's origin defaults to top-left; we want lane 2 visible at top.
    display = occupancy[::-1, :]  # flip vertically: row 0 → lane 2
    im = ax.imshow(
        display,
        aspect="auto",
        cmap="Greys",
        origin="upper",
        extent=(0.0, t_max, -0.5, N_LANES - 0.5),
        vmin=0,
        vmax=max(1.0, float(occupancy.max())),
        interpolation="nearest",
    )

    # Ego trajectory line.
    ax.plot(ego_t, ego_lane, color="#facc15", linewidth=1.8, label="ego")
    # Highlight lane changes.
    if len(ego_lane) > 1:
        changes = np.where(np.diff(ego_lane) != 0)[0] + 1
        ax.scatter(
            ego_t[changes], ego_lane[changes],
            color="#facc15", edgecolor="#451a03", s=24, zorder=3,
        )

    # Mark a collision/exit at the very end.
    if replay.get("terminated"):
        last_t = float(times[-1])
        last_l = int(ego_lane[-1]) if len(ego_lane) else 1
        # Reason: collision if backend marked terminated AND ego still in scene
        # at last tick (we can't always tell from the replay; mark as red dot
        # if there are vehicles in adjacent lane positions at that moment).
        ax.scatter(
            [last_t], [last_l],
            color="#ef4444", edgecolor="#7f1d1d", s=120, marker="X", zorder=4,
            label="terminal",
        )

    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["lane 0\n(right)", "lane 1\n(middle)", "lane 2\n(left)"],
                       fontsize=8, color="#94a3b8")
    ax.set_xlabel("sim time (s)", color="#94a3b8", fontsize=9)
    ax.set_title(title, color="#e2e8f0", fontsize=10, loc="left")
    ax.set_facecolor("#0f172a")
    for spine in ax.spines.values():
        spine.set_edgecolor("#475569")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.legend(loc="upper right", fontsize=8, frameon=False, labelcolor="#cbd5e1")
    return im


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--in", dest="inputs", nargs="+", type=Path, required=True,
        help="one or more replay JSON paths",
    )
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/images/lane_efficiency.png",
    )
    parser.add_argument("--dpi", type=int, default=140)
    args = parser.parse_args(argv)

    replays = [load_replay(p) for p in args.inputs]
    n = len(replays)

    fig, axes = plt.subplots(
        n, 1, figsize=(11, 2.8 * n + 0.8), squeeze=False,
        gridspec_kw={"hspace": 0.55},
    )
    fig.patch.set_facecolor("#0f172a")
    flat = axes.flatten()

    last_im = None
    for ax, replay, path in zip(flat, replays, args.inputs, strict=True):
        title = (
            f"{path.stem}  —  {replay['agent']} · {replay['density']} · "
            f"{replay['n_ticks']} ticks"
        )
        last_im = render_panel(ax, replay, title)

    fig.suptitle(
        "LaneIQ — lane occupancy + ego trajectory",
        color="#e2e8f0", fontsize=13, y=0.99,
    )
    fig.subplots_adjust(top=0.93, bottom=0.07, left=0.1, right=0.88)

    # Shared colorbar describing the heatmap intensity.
    if last_im is not None:
        cbar = fig.colorbar(
            last_im, ax=flat.tolist(), shrink=0.85, pad=0.02, aspect=30,
        )
        cbar.set_label("mean vehicles in lane", color="#94a3b8", fontsize=8)
        cbar.ax.tick_params(colors="#94a3b8", labelsize=7)
        cbar.outline.set_edgecolor("#475569")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=fig.get_facecolor(),
                bbox_inches="tight")
    print(f"✓ saved → {args.out}", file=sys.stderr)

    # Mirror to frontend/public/ so the dashboard can embed it.
    frontend_public = REPO_ROOT / "frontend" / "public"
    if (REPO_ROOT / "frontend").is_dir():
        frontend_public.mkdir(parents=True, exist_ok=True)
        mirror = frontend_public / args.out.name
        fig.savefig(mirror, dpi=args.dpi, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        print(f"✓ mirrored → {mirror}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
