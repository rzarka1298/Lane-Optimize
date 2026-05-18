"""Render training curves for the canonical agents into one PNG.

Reads TensorBoard event files under runs/ and produces a 2x2 grid of
"learning over time" plots — one per canonical agent. Used in
docs/results.md and as the visual answer to "did the AI actually
learn?".

Usage:
    uv run python -m scripts.plot_training_curves \
        --out docs/images/training_curves.png

Agents whose event files are missing are skipped with a warning;
the remaining panels still render. Re-run after a fresh training
run to refresh the canonical figure.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import (
    EventAccumulator,
)

REPO_ROOT = Path(__file__).resolve().parents[1]

# Map "panel" → (run dir, possible reward tags). The reward tag varies by
# trainer: SB3 logs to rollout/ep_rew_mean, our DQN trainer logs to
# episode/return_mean100. We try each in order.
PANELS = [
    {
        "title": "DQN — from-scratch (mixed-density, 200k env steps)",
        "run_dir": "runs/dqn_mixed_200k",
        "tags": ["episode/return_mean100", "rollout/ep_rew_mean"],
        "color": "#3b82f6",
    },
    {
        "title": "PPO v1 — default reward (500k steps)",
        "run_dir": "runs/ppo_mixed_500k/MaskablePPO_1",
        "tags": ["rollout/ep_rew_mean", "episode/return_mean100"],
        "color": "#f97316",
    },
    {
        "title": "PPO v2 — safety_heavy reward (150k steps, canonical)",
        "run_dir": "runs/sweep_safety_heavy/MaskablePPO_1",
        "tags": ["rollout/ep_rew_mean", "episode/return_mean100"],
        "color": "#10b981",
    },
    {
        # Vec-env wrapping under supersuit drops SB3's per-episode reward
        # callback, so rollout/ep_rew_mean isn't logged for MAPPO. Fall back
        # to train/explained_variance which still tells the "value function
        # is learning" story.
        "title": "MAPPO — value-function explained variance (300k steps)",
        "run_dir": "runs/mappo_v1_300k/MAPPO_shared_1",
        "tags": ["train/explained_variance", "rollout/ep_rew_mean"],
        "color": "#facc15",
        "ylabel_override": "explained variance\n(0 = no signal, 1 = perfect prediction)",
    },
]


def load_scalar(run_dir: Path, tag_candidates: list[str]) -> tuple[list[int], list[float], str | None]:
    """Return (steps, values, tag_used) for the first matching tag."""
    if not run_dir.is_dir():
        return [], [], None
    acc = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    try:
        acc.Reload()
    except Exception as exc:
        print(f"  ⚠ failed to read {run_dir}: {exc}", file=sys.stderr)
        return [], [], None
    scalar_tags = acc.Tags().get("scalars", [])
    for tag in tag_candidates:
        if tag in scalar_tags:
            evs = acc.Scalars(tag)
            return [e.step for e in evs], [e.value for e in evs], tag
    return [], [], None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=REPO_ROOT / "docs/images/training_curves.png",
        help="output PNG path",
    )
    parser.add_argument(
        "--dpi", type=int, default=140, help="figure DPI",
    )
    args = parser.parse_args(argv)

    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5))
    fig.patch.set_facecolor("#0f172a")
    flat_axes = axes.flatten()

    rendered = 0
    for ax, panel in zip(flat_axes, PANELS, strict=True):
        run_dir = REPO_ROOT / panel["run_dir"]
        steps, values, tag = load_scalar(run_dir, panel["tags"])

        ax.set_facecolor("#0f172a")
        for spine in ax.spines.values():
            spine.set_edgecolor("#475569")
        ax.tick_params(colors="#94a3b8", labelsize=8)
        ax.set_title(panel["title"], color="#e2e8f0", fontsize=10, loc="left")
        ax.grid(True, color="#1e293b", linewidth=0.6)

        if not steps:
            ax.text(
                0.5, 0.5, "no events found",
                ha="center", va="center", color="#64748b",
                transform=ax.transAxes,
            )
            print(f"  ⚠ skipped: {panel['run_dir']} (no matching tag)")
            continue

        ax.plot(steps, values, color=panel["color"], linewidth=1.6)
        ax.set_xlabel("training step", color="#94a3b8", fontsize=8)
        ylabel = panel.get("ylabel_override") or f"mean episode return\n({tag})"
        ax.set_ylabel(ylabel, color="#94a3b8", fontsize=8)
        rendered += 1
        print(f"  ✓ {panel['run_dir']}: {len(steps)} points from {tag!r}")

    fig.suptitle(
        "LaneIQ — training curves (canonical agents)",
        color="#e2e8f0", fontsize=13, y=0.995,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor=fig.get_facecolor())
    print(f"\n✓ saved → {args.out}  ({rendered}/{len(PANELS)} panels rendered)")

    # Also drop a copy in frontend/public/ so Vite serves it for the
    # in-dashboard "How it learned" panel. Skipped silently if the
    # frontend directory doesn't exist (e.g., backend-only deployments).
    frontend_public = REPO_ROOT / "frontend" / "public"
    if frontend_public.is_dir() or (REPO_ROOT / "frontend").is_dir():
        frontend_public.mkdir(parents=True, exist_ok=True)
        mirror = frontend_public / args.out.name
        fig.savefig(mirror, dpi=args.dpi, facecolor=fig.get_facecolor())
        print(f"✓ mirrored → {mirror}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
