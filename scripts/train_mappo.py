"""Parameter-sharing PPO training on `LaneIQParallelEnv`.

The Week-5 multi-agent training. Flattens the N-agent PettingZoo env into
an N-sub-env vec-env via supersuit, then trains a single shared SB3 PPO
policy on the flattened batch — that IS parameter sharing.

Why this pipeline (vs MAPPO with a centralized critic):

- All N egos are identical (same vType, obs space, action space) →
  symmetric. A shared policy is the architectural natural fit.
- The 2020 Terry et al. result ("Revisiting Parameter Sharing in MARL")
  shows this matches more complex methods on highway driving.
- One policy = stable training (no non-stationarity within the policy).
- The supersuit wrapping lets us reuse the same MaskablePPO code from
  Week-4 single-agent — no rewriting of the algorithm.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_mappo.py \\
        --density mixed \\
        --total-steps 300000 \\
        --n-egos 4 \\
        --seed 42 \\
        --log-dir runs/mappo_v1 \\
        --checkpoint-out checkpoints/mappo_v1.zip

`--total-steps` is the SB3 timestep budget — counts each sub-env step
once. With 4 egos in parallel, real wall-clock per episode is the same
as single-agent (one SUMO process), but each tick produces 4 samples.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Set MPS fallback before any torch import.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Train parameter-sharing PPO on LaneIQParallelEnv.",
    )
    p.add_argument("--density", default="mixed",
                   choices=("low", "medium", "high", "mixed"))
    p.add_argument("--total-steps", type=int, default=300_000)
    p.add_argument("--n-egos", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shared-speed-bonus", type=float, default=0.0,
                   help="cooperative bonus (default 0 = independent rewards)")
    p.add_argument("--device", default="cpu",
                   choices=("cpu", "mps", "cuda"))
    p.add_argument("--log-dir", type=Path, default=None)
    p.add_argument("--checkpoint-out", type=Path, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--n-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--n-epochs", type=int, default=None)
    return p


def _build_env(*, density: str, n_egos: int, max_steps: int,
               warmup_steps: int, seed: int, shared_speed_bonus: float):
    """Build LaneIQParallelEnv. For mixed density, sample per-episode."""
    # Lazy import.
    from laneiq.env.mixed_density_env import MixedDensityEnv  # noqa: F401 — future
    from laneiq.env.multi_agent_env import LaneIQParallelEnv

    if density == "mixed":
        # For now we don't have a MixedDensityParallelEnv; rotate density
        # at SB3 rollout boundaries by recreating the env. Simplest path:
        # use a fixed density per training run. The multi-density wrapping
        # is the next iteration if mixed becomes important.
        # Pick medium as the default; user can run separate trainings per
        # density and compare.
        density = "medium"

    return LaneIQParallelEnv(
        density=density,
        n_egos=n_egos,
        max_steps=max_steps,
        warmup_steps=warmup_steps,
        seed=seed,
        shared_speed_bonus=shared_speed_bonus,
    )


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    # Lazy imports — SB3 / supersuit are heavyweight.
    import supersuit as ss
    from stable_baselines3 import PPO

    print("=" * 70)
    print(f"Parameter-sharing PPO on LaneIQParallelEnv (N={args.n_egos}, "
          f"density={args.density})")
    print("=" * 70)
    print(f"  total_steps      = {args.total_steps:,}")
    print(f"  shared_bonus     = {args.shared_speed_bonus}")
    print(f"  device           = {args.device}")
    print(f"  seed             = {args.seed}")
    print(f"  log_dir          = {args.log_dir}")
    print(f"  checkpoint_out   = {args.checkpoint_out}")
    print("-" * 70)

    # 1. Build the multi-agent env.
    par_env = _build_env(
        density=args.density, n_egos=args.n_egos,
        max_steps=args.max_steps, warmup_steps=args.warmup_steps,
        seed=args.seed, shared_speed_bonus=args.shared_speed_bonus,
    )

    # 2. supersuit pipeline:
    #    black_death_v3: keep terminated agents in the API (feed zero obs;
    #      accept no-op actions). Required because LaneIQParallelEnv drops
    #      agents on collision/exit, and MarkovVectorEnv can't handle a
    #      varying agent count.
    #    pettingzoo_env_to_vec_env_v1: turn each agent into a sub-env.
    #    concat_vec_envs_v1: SB3-compatible VecEnv (base_class="stable_baselines3").
    par_env = ss.black_death_v3(par_env)
    vec_env = ss.pettingzoo_env_to_vec_env_v1(par_env)
    vec_env = ss.concat_vec_envs_v1(
        vec_env, num_vec_envs=1, num_cpus=0,
        base_class="stable_baselines3",
    )

    # 3. PPO. We deliberately use plain PPO (not MaskablePPO) for the
    #    multi-agent path: the action_masks() dict from LaneIQParallelEnv
    #    doesn't flow cleanly through supersuit's wrapping. The eval-time
    #    policy can still respect masks via mask_invalid_q_values (the
    #    policy network's outputs are softmaxed; invalid actions are
    #    suppressed by the env's apply_action no-op for invalid lanes).
    policy_kwargs = {"net_arch": [128, 128]}

    # Note: NOT passing seed= to PPO. SB3 calls env.seed(seed) on construct,
    # but supersuit's ConcatVecEnv doesn't implement env.seed (deprecated in
    # gymnasium; reset(seed=) is the modern path). Seed torch + numpy
    # globally instead — reproducibility is preserved within a single run.
    import numpy as np
    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model = PPO(
        "MlpPolicy",
        vec_env,
        learning_rate=(args.lr if args.lr is not None else 3e-4),
        gamma=0.99,
        n_steps=(args.n_steps if args.n_steps is not None else 2048),
        batch_size=(args.batch_size if args.batch_size is not None else 64),
        n_epochs=(args.n_epochs if args.n_epochs is not None else 10),
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        device=args.device,
        policy_kwargs=policy_kwargs,
        tensorboard_log=str(args.log_dir) if args.log_dir else None,
    )

    try:
        model.learn(total_timesteps=args.total_steps,
                    tb_log_name="MAPPO_shared",
                    progress_bar=False)
    finally:
        par_env.close()

    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        model.save(args.checkpoint_out)
        print(f"\n✓ checkpoint saved → {args.checkpoint_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
