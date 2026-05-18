"""PPO (MaskablePPO) training CLI for LaneIQEnv.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_ppo.py \\
        --density mixed \\
        --total-steps 500000 \\
        --seed 42 \\
        --log-dir runs/ppo_mixed_500k \\
        --checkpoint-out checkpoints/ppo_mixed_500k.zip

Defaults to CPU (per Week-3 throughput profile: our small MLP is faster
on CPU than MPS).
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

# Set MPS fallback before torch import is touched anywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import gymnasium as gym


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MaskablePPO on LaneIQEnv.")
    p.add_argument(
        "--density", default="mixed",
        choices=("low", "medium", "high", "mixed", "gauntlet", "rush"),
        help="'mixed' samples low/medium/high uniformly per episode; "
             "'gauntlet' is the overtaking-forcing scenario (ego starts "
             "in lane 0 behind heavy slow traffic); 'rush' raises the "
             "speed limit so the ego outruns all background traffic.",
    )
    p.add_argument("--total-steps", type=int, default=500_000)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cpu", choices=("cpu", "mps", "cuda"),
        help="default cpu — fastest on Apple Silicon for our net size",
    )
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--checkpoint-out", type=Path, default=None)

    # PPO hyperparameter overrides.
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    p.add_argument("--clip-range", type=float, default=None)
    p.add_argument("--ent-coef", type=float, default=None)

    # Reward-shape overrides (omit any to inherit DEFAULT_REWARD_CONFIG).
    p.add_argument("--w-speed", type=float, default=None)
    p.add_argument("--w-collision", type=float, default=None)
    p.add_argument("--w-unsafe-gap", type=float, default=None)
    p.add_argument("--w-lane-cost", type=float, default=None)
    p.add_argument("--w-jerk", type=float, default=None)
    p.add_argument("--w-alive", type=float, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Lazy import — SB3/torch are heavyweight.
    from laneiq.agents.ppo_sb3 import DEFAULT_PPO_CONFIG, PPOAgent
    from laneiq.env.highway_env import LaneIQEnv
    from laneiq.env.mixed_density_env import MixedDensityEnv
    from laneiq.env.rewards import DEFAULT_REWARD_CONFIG

    # Build the PPO hyperparam config from defaults + overrides.
    cfg_overrides: dict = {}
    for key in ("lr", "n_steps", "batch_size", "n_epochs", "clip_range", "ent_coef"):
        val = getattr(args, key)
        if val is not None:
            cfg_overrides[key] = val
    cfg = replace(DEFAULT_PPO_CONFIG, **cfg_overrides) if cfg_overrides else DEFAULT_PPO_CONFIG

    # Reward-shape overrides — produce a custom RewardConfig only if the
    # user touched a flag, otherwise inherit the env's default.
    reward_overrides: dict = {}
    for key in ("w_speed", "w_collision", "w_unsafe_gap", "w_lane_cost",
                "w_jerk", "w_alive"):
        val = getattr(args, key)
        if val is not None:
            reward_overrides[key] = val
    reward_cfg = (
        replace(DEFAULT_REWARD_CONFIG, **reward_overrides)
        if reward_overrides else DEFAULT_REWARD_CONFIG
    )

    print("=" * 70)
    print(f"PPO (MaskablePPO) training on LaneIQEnv ({args.density} density)")
    print("=" * 70)
    print(f"  total_steps      = {args.total_steps:,}")
    print(f"  device           = {args.device}")
    print(f"  seed             = {args.seed}")
    print(f"  log_dir          = {args.log_dir}")
    print(f"  checkpoint_out   = {args.checkpoint_out}")
    print(f"  lr               = {cfg.lr}")
    print(f"  n_steps (rollout)= {cfg.n_steps}")
    print(f"  batch_size       = {cfg.batch_size}")
    print(f"  n_epochs         = {cfg.n_epochs}")
    print(f"  clip_range       = {cfg.clip_range}")
    print(f"  ent_coef         = {cfg.ent_coef}")
    print(f"  hidden_dims      = {cfg.hidden_dims}")
    if reward_overrides:
        print(f"  reward overrides = {reward_overrides}")
    print("-" * 70)

    if args.density == "mixed":
        env: gym.Env = MixedDensityEnv(
            density_pool=("low", "medium", "high"),
            density_seed=args.seed,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            reward_config=reward_cfg,
        )
    else:
        env = LaneIQEnv(
            density=args.density,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
            reward_config=reward_cfg,
        )

    agent = PPOAgent(
        env=env,
        config=cfg,
        device=args.device,
        seed=args.seed,
        tensorboard_log=args.log_dir,
    )

    try:
        agent.learn(
            total_timesteps=args.total_steps,
            tb_log_name="MaskablePPO",
            progress_bar=False,
        )
    finally:
        env.close()

    if args.checkpoint_out is not None:
        agent.save(args.checkpoint_out)
        print(f"\n✓ checkpoint saved → {args.checkpoint_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
