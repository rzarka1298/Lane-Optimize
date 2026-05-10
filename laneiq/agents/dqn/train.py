"""Training loop for `DQNAgent`.

Generic over the environment — works on CartPole-v1 (the Week-2 gate)
and on `LaneIQEnv` (Week 3+) without changes. The env is passed in; the
trainer doesn't import gym specifics beyond the standard 5-tuple step
contract.

What it does:

1. Roll out the env, ε-greedy action from the agent.
2. Push transitions to the agent's replay buffer.
3. Every `train_freq` env steps, take one gradient step.
4. Periodically sync the target network.
5. Log per-episode return, per-step loss/Q/ε to TensorBoard.
6. Track the running best mean-100-episode return for the gate.

The function returns a `TrainResult` so callers (tests, scripts) can
assert on the final mean return without parsing TensorBoard files.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np

from laneiq.agents.dqn.dqn_agent import DQNAgent

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainResult:
    """Summary of one training run. Returned by `train()`."""

    total_steps: int
    n_episodes: int
    mean_return_last_100: float
    best_mean_return_last_100: float
    final_loss: float
    wall_time_s: float


def train(
    env: gym.Env,
    agent: DQNAgent,
    *,
    total_steps: int,
    seed: int | None = None,
    log_dir: str | Path | None = None,
    log_interval_steps: int = 1_000,
    eval_interval_steps: int | None = None,
    on_episode_end: Callable[[int, float, int], None] | None = None,
    use_action_mask: bool = False,
) -> TrainResult:
    """Train `agent` on `env` for `total_steps` env steps.

    Args:
        env: a Gymnasium env. Either CartPole-v1 or `LaneIQEnv`.
        agent: an initialized `DQNAgent`.
        total_steps: number of env steps (= action selections).
        seed: per-run seed (env reset seed AND agent re-seed).
        log_dir: if provided, write TensorBoard event files here.
        log_interval_steps: per-step diagnostics (loss, Q, ε) cadence.
        eval_interval_steps: optional unused hook for future eval callbacks.
        on_episode_end: optional callback `(episode_idx, ep_return, ep_length)`.
        use_action_mask: if True, the env must expose `action_masks()` and
            both ε-greedy paths will respect it. Set True for `LaneIQEnv`,
            False for CartPole.

    Returns:
        `TrainResult` with end-of-training summary.
    """
    writer = None
    if log_dir is not None:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(log_dir=str(log_dir))

    obs, _info = env.reset(seed=seed)
    agent.reset(seed=seed)

    episode_returns: deque[float] = deque(maxlen=100)
    episode_lengths: deque[int] = deque(maxlen=100)
    current_return = 0.0
    current_length = 0
    n_episodes = 0
    last_loss = float("nan")
    best_mean_100 = float("-inf")

    t_start = time.monotonic()

    for step in range(1, total_steps + 1):
        eps = agent.epsilon(step)
        mask = env.action_masks() if use_action_mask else None  # type: ignore[attr-defined]
        action = agent.act_epsilon_greedy(obs, eps, action_mask=mask)

        next_obs, reward, terminated, truncated, _ = env.step(action)
        # Store transition. Treat truncated as non-terminal for bootstrapping
        # (the world continues; only `terminated` says "no future value").
        agent.push_transition(obs, action, float(reward), next_obs, terminated)

        current_return += float(reward)
        current_length += 1

        if terminated or truncated:
            episode_returns.append(current_return)
            episode_lengths.append(current_length)
            n_episodes += 1
            if on_episode_end is not None:
                on_episode_end(n_episodes, current_return, current_length)
            if writer is not None:
                writer.add_scalar("episode/return", current_return, step)
                writer.add_scalar("episode/length", current_length, step)
                if len(episode_returns) > 0:
                    writer.add_scalar(
                        "episode/return_mean100", float(np.mean(episode_returns)), step,
                    )
            current_return = 0.0
            current_length = 0
            obs, _info = env.reset()
        else:
            obs = next_obs

        # Gradient step on cadence.
        if step % agent._cfg.train_freq == 0:
            metrics = agent.train_step()
            if metrics is not None:
                last_loss = metrics["loss"]
                if writer is not None and step % log_interval_steps == 0:
                    writer.add_scalar("train/loss", metrics["loss"], step)
                    writer.add_scalar("train/mean_q", metrics["mean_q"], step)
                    writer.add_scalar("train/mean_target_q", metrics["mean_target_q"], step)
                    writer.add_scalar("train/grad_norm", metrics["grad_norm"], step)
                    writer.add_scalar("train/epsilon", eps, step)

        # Target sync.
        agent.maybe_sync_target(step)

        # Track best mean-100 for the gate.
        if len(episode_returns) >= 100:
            mean100 = float(np.mean(episode_returns))
            if mean100 > best_mean_100:
                best_mean_100 = mean100

    if writer is not None:
        writer.flush()
        writer.close()

    final_mean100 = float(np.mean(episode_returns)) if episode_returns else float("nan")
    return TrainResult(
        total_steps=total_steps,
        n_episodes=n_episodes,
        mean_return_last_100=final_mean100,
        best_mean_return_last_100=best_mean_100 if best_mean_100 > float("-inf") else final_mean100,
        final_loss=last_loss,
        wall_time_s=time.monotonic() - t_start,
    )
