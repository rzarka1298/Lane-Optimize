"""DQN training CLI for LaneIQEnv.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_dqn.py \\
        --density medium \\
        --total-steps 100000 \\
        --seed 42 \\
        --log-dir runs/dqn_medium_100k \\
        --checkpoint-out checkpoints/dqn_medium_100k.pt

Defaults to CPU (per Week-3 throughput profile: 8x faster than MPS for
our 17k-param Q-net).

Use `tensorboard --logdir runs/` to watch live training curves.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import replace
from pathlib import Path

# Set MPS fallback before torch import is touched anywhere.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train DQN on LaneIQEnv.")
    p.add_argument("--density", default="medium", choices=("low", "medium", "high"))
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--max-steps", type=int, default=1000, help="env max_steps per episode")
    p.add_argument("--warmup-steps", type=int, default=50, help="env warmup before ego enters")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cpu", choices=("cpu", "mps", "cuda"),
        help="default cpu — fastest on Apple Silicon for our Q-net size",
    )
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--checkpoint-out", type=Path, default=None)

    # DQN hyperparameter overrides (optional).
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--buffer-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--eps-decay-steps", type=int, default=None)
    p.add_argument("--target-update-freq", type=int, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Lazy import — torch is heavyweight.
    from laneiq.agents.dqn.dqn_agent import DEFAULT_DQN_CONFIG, DQNAgent
    from laneiq.agents.dqn.train import train
    from laneiq.env.highway_env import LaneIQEnv
    from laneiq.env.observations import OBS_DIM

    # Build the config: defaults + any CLI overrides.
    cfg_overrides: dict = {}
    if args.lr is not None:
        cfg_overrides["lr"] = args.lr
    if args.buffer_size is not None:
        cfg_overrides["buffer_size"] = args.buffer_size
    if args.batch_size is not None:
        cfg_overrides["batch_size"] = args.batch_size
    if args.eps_decay_steps is not None:
        cfg_overrides["eps_decay_steps"] = args.eps_decay_steps
    if args.target_update_freq is not None:
        cfg_overrides["target_update_freq"] = args.target_update_freq
    cfg = replace(DEFAULT_DQN_CONFIG, **cfg_overrides) if cfg_overrides else DEFAULT_DQN_CONFIG

    print("=" * 70)
    print(f"DQN training on LaneIQEnv ({args.density} density)")
    print("=" * 70)
    print(f"  total_steps      = {args.total_steps:,}")
    print(f"  device           = {args.device}")
    print(f"  seed             = {args.seed}")
    print(f"  max_steps/ep     = {args.max_steps}")
    print(f"  warmup_steps     = {args.warmup_steps}")
    print(f"  log_dir          = {args.log_dir}")
    print(f"  checkpoint_out   = {args.checkpoint_out}")
    print(f"  use_double_dqn   = {cfg.use_double_dqn}")
    print(f"  lr               = {cfg.lr}")
    print(f"  buffer_size      = {cfg.buffer_size:,}")
    print(f"  batch_size       = {cfg.batch_size}")
    print(f"  eps_decay_steps  = {cfg.eps_decay_steps:,}")
    print(f"  target_update    = every {cfg.target_update_freq:,} steps")
    print("-" * 70)

    env = LaneIQEnv(
        density=args.density,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        seed=args.seed,
    )
    agent = DQNAgent(
        obs_dim=OBS_DIM,
        n_actions=3,
        config=cfg,
        device=args.device,
        seed=args.seed,
    )

    # Per-episode print so the user can watch convergence in real time
    # without needing TensorBoard open.
    last_print_episode = [0]

    def _on_episode_end(ep_idx: int, ep_return: float, ep_length: int) -> None:
        if ep_idx - last_print_episode[0] >= 25 or ep_idx == 1:
            print(
                f"  ep {ep_idx:5d}  return={ep_return:8.2f}  "
                f"len={ep_length:4d}  ε={agent.epsilon(ep_idx):.3f}",
                flush=True,
            )
            last_print_episode[0] = ep_idx

    try:
        result = train(
            env,
            agent,
            total_steps=args.total_steps,
            seed=args.seed,
            log_dir=args.log_dir,
            on_episode_end=_on_episode_end,
            use_action_mask=True,    # LaneIQEnv exposes action_masks()
        )
    finally:
        env.close()

    print("-" * 70)
    print(
        f"\nTraining complete:\n"
        f"  total_steps           = {result.total_steps:,}\n"
        f"  n_episodes            = {result.n_episodes:,}\n"
        f"  mean_return_last_100  = {result.mean_return_last_100:8.2f}\n"
        f"  best_mean_last_100    = {result.best_mean_return_last_100:8.2f}\n"
        f"  final_loss            = {result.final_loss:.4f}\n"
        f"  wall_time_s           = {result.wall_time_s:6.1f}\n"
        f"                        = {result.wall_time_s / 60:.1f} min\n",
    )

    if args.checkpoint_out is not None:
        agent.save(args.checkpoint_out)
        print(f"✓ checkpoint saved → {args.checkpoint_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
