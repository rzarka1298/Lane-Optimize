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

import gymnasium as gym  # noqa: E402


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train MaskablePPO on LaneIQEnv.")
    p.add_argument(
        "--density", default="mixed",
        choices=("low", "medium", "high", "mixed"),
        help="'mixed' samples low/medium/high uniformly per episode.",
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Lazy import — SB3/torch are heavyweight.
    from laneiq.agents.ppo_sb3 import DEFAULT_PPO_CONFIG, PPOAgent
    from laneiq.env.highway_env import LaneIQEnv
    from laneiq.env.mixed_density_env import MixedDensityEnv

    # Build the config from defaults + overrides.
    cfg_overrides: dict = {}
    for key in ("lr", "n_steps", "batch_size", "n_epochs", "clip_range", "ent_coef"):
        cli_key = key.replace("_", "-")
        val = getattr(args, key)
        if val is not None:
            cfg_overrides[key] = val
    cfg = replace(DEFAULT_PPO_CONFIG, **cfg_overrides) if cfg_overrides else DEFAULT_PPO_CONFIG

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
    print("-" * 70)

    if args.density == "mixed":
        env: gym.Env = MixedDensityEnv(
            density_pool=("low", "medium", "high"),
            density_seed=args.seed,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
        )
    else:
        env = LaneIQEnv(
            density=args.density,
            max_steps=args.max_steps,
            warmup_steps=args.warmup_steps,
            seed=args.seed,
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
