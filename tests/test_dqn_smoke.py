"""Week-2 hard gate: from-scratch DQN must solve CartPole-v1.

If this fails, the DQN code is broken in the small. No amount of LaneIQ
hyperparameter tuning will fix it. Pass → proceed to Week 3 LaneIQ
training. Fail → debug DQN, do not pass go.

Marked `@pytest.mark.slow` so it isn't part of the default suite. Runs
in 1-3 minutes on Apple Silicon CPU. Run via:

    uv run pytest tests/test_dqn_smoke.py -v -m slow

The threshold (≥195 mean over the last 100 episodes) is the canonical
"basically learned to balance the pole" mark; CartPole-v1 caps at 500
per episode but 195 is enough to demonstrate that the DQN pipeline is
correct end-to-end.
"""

from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import pytest

from laneiq.agents.dqn.dqn_agent import DEFAULT_DQN_CONFIG, DQNAgent
from laneiq.agents.dqn.train import train


@pytest.mark.slow
def test_dqn_solves_cartpole_v1() -> None:
    """End-to-end gate: train DQN on CartPole-v1, assert mean return ≥ 195."""
    env = gym.make("CartPole-v1")

    # CartPole-tuned config — smaller than the LaneIQ defaults because
    # the task is much simpler. ~50k env steps converges reliably on CPU.
    cfg = replace(
        DEFAULT_DQN_CONFIG,
        hidden_dims=(128, 128),
        lr=1e-3,
        gamma=0.99,
        buffer_size=10_000,
        batch_size=64,
        learning_starts=1_000,
        train_freq=4,
        target_update_freq=500,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay_steps=10_000,
        use_double_dqn=True,
        # CartPole has no reward outliers (just +1/step); MSE converges
        # faster than Huber here. The LaneIQ defaults use Huber because
        # of the -50 collision term. See DQNConfig.huber_loss + Task #17.
        huber_loss=False,
    )

    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n  # type: ignore[attr-defined]
    agent = DQNAgent(obs_dim=obs_dim, n_actions=n_actions, config=cfg, device="cpu", seed=42)

    result = train(
        env,
        agent,
        total_steps=50_000,
        seed=42,
        log_dir=None,
        use_action_mask=False,
    )

    env.close()

    # Diagnostics for failure debugging — printed on assertion fail.
    print(
        f"\nDQN CartPole gate result:\n"
        f"  total_steps           = {result.total_steps}\n"
        f"  n_episodes            = {result.n_episodes}\n"
        f"  mean_return_last_100  = {result.mean_return_last_100:.2f}\n"
        f"  best_mean_last_100    = {result.best_mean_return_last_100:.2f}\n"
        f"  final_loss            = {result.final_loss:.4f}\n"
        f"  wall_time_s           = {result.wall_time_s:.1f}\n"
    )

    # The gate. Use the *best* rolling mean rather than the final one so
    # we don't fail on a late-training oscillation; either way CartPole
    # is "solved" once we've sustained ≥195.
    assert result.best_mean_return_last_100 >= 195.0, (
        f"DQN failed to solve CartPole-v1: best mean100 = "
        f"{result.best_mean_return_last_100:.2f} (threshold 195). "
        f"See concept-03-dqn-math.md 'Common failure modes' table."
    )
