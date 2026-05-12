"""Reward weight sweep — train PPO with multiple reward configs.

Trains a PPO for each (named) RewardConfig in `_SWEEP_CONFIGS`, then
evaluates all of them across the 3 densities x 5 seeds. Writes results
to results/week4_reward_sweep.csv. Used by scripts/plot_pareto.py to
generate the Week-4 hero Pareto plot.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/sweep_reward_ppo.py \\
        --total-steps 150000 \\
        --seed 42

Estimated wall-clock at production settings (PPO ~200 fps):
    4 configs x 150k env-steps = 600k env-steps ≈ 50 min training
    4 configs x 3 densities x 5 seeds = 60 eval episodes ≈ 5 min
    Total ≈ 55-60 min.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

# Set MPS fallback before torch import is touched anywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from laneiq.env.rewards import DEFAULT_REWARD_CONFIG, RewardConfig

# ---------------------------------------------------------------------------
# Sweep definitions
# ---------------------------------------------------------------------------

# Each entry: (name, RewardConfig). Names appear in checkpoints, plot.
_SWEEP_CONFIGS: list[tuple[str, RewardConfig]] = [
    # Baseline (v1 weights — the PPO mixed-500k config in the canonical eval).
    ("default", DEFAULT_REWARD_CONFIG),

    # Safety-heavy: collision penalty doubled, unsafe-gap penalty doubled.
    # Should push the policy toward fewer lane changes, lower top speed.
    (
        "safety_heavy",
        replace(DEFAULT_REWARD_CONFIG, w_collision=-100.0, w_unsafe_gap=-1.0),
    ),

    # Speed-heavy: speed reward doubled, lane-change cost halved.
    # Should push the policy toward higher mean speed, more lane changes,
    # accepting more collisions.
    (
        "speed_heavy",
        replace(DEFAULT_REWARD_CONFIG, w_speed=2.0, w_lane_cost=-0.02),
    ),

    # Jerk-heavy: jerk penalty 2.5x. Should produce smoother driving
    # (lower jerk_rms) at a modest speed cost.
    (
        "jerk_heavy",
        replace(DEFAULT_REWARD_CONFIG, w_jerk=-0.5),
    ),
]


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep PPO over reward weights.")
    p.add_argument("--total-steps", type=int, default=150_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints/sweep"),
        help="where to save trained PPO checkpoints",
    )
    p.add_argument(
        "--eval-seeds", type=str, default="0,1,2,3,4",
        help="comma-separated seeds for eval (per density)",
    )
    p.add_argument(
        "--out", type=Path, default=Path("results/week4_reward_sweep.csv"),
        help="aggregated per-cell CSV (one row per (config, density))",
    )
    p.add_argument(
        "--per-episode-out", type=Path,
        default=Path("results/week4_reward_sweep_episodes.csv"),
        help="raw per-episode CSV",
    )
    p.add_argument(
        "--skip-train", action="store_true",
        help="skip training (reuse existing checkpoints); only re-eval",
    )
    return p


def train_one(
    *,
    name: str,
    reward_config: RewardConfig,
    total_steps: int,
    max_steps: int,
    warmup_steps: int,
    seed: int,
    checkpoint_path: Path,
) -> None:
    """Train one PPO for the given reward config."""
    from laneiq.agents.ppo_sb3 import PPOAgent
    from laneiq.env.mixed_density_env import MixedDensityEnv

    print(f"\n{'='*70}\n[{name}] training PPO for {total_steps:,} steps\n{'='*70}", flush=True)
    print(
        f"  w_speed={reward_config.w_speed:>6.2f}  "
        f"w_collision={reward_config.w_collision:>6.1f}  "
        f"w_unsafe_gap={reward_config.w_unsafe_gap:>5.2f}  "
        f"w_jerk={reward_config.w_jerk:>5.2f}  "
        f"w_lane_cost={reward_config.w_lane_cost:>5.2f}",
        flush=True,
    )

    env = MixedDensityEnv(
        density_pool=("low", "medium", "high"),
        density_seed=seed,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        seed=seed,
        reward_config=reward_config,
    )
    agent = PPOAgent(env=env, device="cpu", seed=seed,
                     tensorboard_log=f"runs/sweep_{name}")
    t0 = time.monotonic()
    try:
        agent.learn(total_timesteps=total_steps, tb_log_name="MaskablePPO")
    finally:
        env.close()
    wall = time.monotonic() - t0
    agent.save(checkpoint_path)
    print(f"  [{name}] trained in {wall/60:.1f} min  →  {checkpoint_path}", flush=True)


def eval_one(
    *,
    name: str,
    checkpoint_path: Path,
    densities: list[str],
    seeds: list[int],
    max_steps: int,
    warmup_steps: int,
) -> tuple[list, list]:
    """Eval one PPO checkpoint across (densities x seeds). Returns
    (aggregated_per_cell, raw_per_episode)."""
    from laneiq.agents.ppo_sb3 import PPOAgent
    from laneiq.eval.metrics import aggregate
    from laneiq.eval.runner import run_episodes

    print(f"\n[{name}] evaluating across {len(densities)} densities x "
          f"{len(seeds)} seeds", flush=True)
    agent = PPOAgent.from_checkpoint(checkpoint_path, device="cpu")
    agent.name = name  # so the CSV row labels by config name, not "ppo"

    aggregated = []
    per_episode = []
    for density in densities:
        t0 = time.monotonic()
        eps = run_episodes(
            agent, density=density, seeds=seeds,
            max_steps=max_steps, warmup_steps=warmup_steps,
        )
        elapsed = time.monotonic() - t0
        agg = aggregate(eps)
        aggregated.append(agg)
        per_episode.extend(eps)
        print(
            f"  [{name}] x {density:<6}  "
            f"return={agg.return_total_mean:7.2f}  "
            f"collisions={agg.collision_rate*100:5.1f}%  "
            f"speed={agg.mean_speed_mps_mean:5.2f} m/s  "
            f"lc/min={agg.lane_changes_per_min_mean:6.2f}  "
            f"({len(seeds)} eps in {elapsed:5.1f}s)",
            flush=True,
        )
    return aggregated, per_episode


def write_aggregated_csv(rows, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def write_per_episode_csv(rows, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(asdict(r))


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    seeds = [int(s) for s in args.eval_seeds.split(",") if s.strip()]
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    overall_t0 = time.monotonic()
    print(f"\nReward weight sweep — {len(_SWEEP_CONFIGS)} configs, "
          f"{args.total_steps:,} steps each", flush=True)

    # 1. Train (sequential).
    for name, cfg in _SWEEP_CONFIGS:
        ckpt = args.checkpoint_dir / f"ppo_{name}.zip"
        if args.skip_train and ckpt.exists():
            print(f"\n[{name}] skipping training (--skip-train, checkpoint exists)",
                  flush=True)
            continue
        train_one(
            name=name, reward_config=cfg, total_steps=args.total_steps,
            max_steps=args.max_steps, warmup_steps=args.warmup_steps,
            seed=args.seed, checkpoint_path=ckpt,
        )

    # 2. Eval (sequential).
    all_aggregated = []
    all_per_episode = []
    for name, _cfg in _SWEEP_CONFIGS:
        ckpt = args.checkpoint_dir / f"ppo_{name}.zip"
        if not ckpt.exists():
            print(f"\n[{name}] WARNING: checkpoint missing, skipping eval", flush=True)
            continue
        agg, eps = eval_one(
            name=name, checkpoint_path=ckpt,
            densities=["low", "medium", "high"], seeds=seeds,
            max_steps=args.max_steps, warmup_steps=args.warmup_steps,
        )
        all_aggregated.extend(agg)
        all_per_episode.extend(eps)

    # 3. Write CSVs.
    write_aggregated_csv(all_aggregated, args.out)
    write_per_episode_csv(all_per_episode, args.per_episode_out)
    print(f"\n✓ wrote aggregated → {args.out}", flush=True)
    print(f"✓ wrote per-episode → {args.per_episode_out}", flush=True)

    wall = time.monotonic() - overall_t0
    print(f"\nTotal sweep time: {wall/60:.1f} min", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
