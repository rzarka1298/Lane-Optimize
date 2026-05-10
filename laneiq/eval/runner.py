"""Evaluation runner — drives `LaneIQEnv` + a `Policy` through episodes
and produces `EpisodeMetrics`.

Eval-time semantics differ from training in two ways:

- **No exploration.** DQN's `act()` is greedy; ε-greedy is reserved for
  the trainer. Baselines have no exploration to begin with.
- **Deterministic seeds.** Each episode is seeded; same seed → same
  background traffic → reproducible numbers.

The runner reads the same RawTraCI snapshot the env's reward path uses,
so the metrics are computed against ground truth (not against the
normalized observation, which would lose units).
"""

from __future__ import annotations

import math
from typing import Any

import traci

from laneiq.agents.base import Policy
from laneiq.env.highway_env import LaneIQEnv
from laneiq.eval.metrics import EpisodeMetrics, MetricsAccumulator

_EGO_ID: str = "ego_0"  # mirrors highway_env's constant


def run_episode(
    env: LaneIQEnv,
    agent: Policy,
    *,
    seed: int,
    use_action_mask: bool = True,
) -> EpisodeMetrics:
    """Drive `agent` through one episode of `env`. Caller owns env lifecycle.

    Args:
        env: an already-constructed `LaneIQEnv`. The runner calls `reset(seed=seed)`.
        agent: any `Policy`. `agent.act()` is greedy (no exploration).
        seed: per-episode seed; flows to env.reset and agent.reset.
        use_action_mask: pass the env's action mask through to `agent.act`.

    Returns:
        `EpisodeMetrics` snapshot covering the entire episode.
    """
    obs, _info = env.reset(seed=seed)
    agent.reset(seed=seed)

    acc = MetricsAccumulator(
        agent=agent.name,
        density=env._density,
        seed=seed,
    )
    initial_pos = float(traci.vehicle.getLanePosition(_EGO_ID))
    acc.start(initial_pos=initial_pos)

    terminated = truncated = False
    last_reason: str | None = None

    while not (terminated or truncated):
        mask = env.action_masks() if use_action_mask else None
        action = agent.act(obs, action_mask=mask)
        obs, reward, terminated, truncated, info = env.step(action)

        # Pull raw-units snapshot from TraCI for metrics computation.
        # On the terminal step the ego may have been removed; fall back to
        # the previous values (we already accumulated them on the prior step).
        if _EGO_ID in traci.vehicle.getIDList():
            speed = float(traci.vehicle.getSpeed(_EGO_ID))
            accel = float(traci.vehicle.getAcceleration(_EGO_ID))
            pos = float(traci.vehicle.getLanePosition(_EGO_ID))
            leader = traci.vehicle.getLeader(_EGO_ID, 100.0)
            if leader is not None and leader[0] != "":
                gap = float(leader[1])
                front_thw = gap / max(speed, 0.5)
            else:
                front_thw = math.inf
        else:
            # Ego removed (collision or exited); skip metrics update for
            # this step, the previous step already captured the last live state.
            speed = float("nan")  # unused — observe() not called below
            accel = 0.0
            pos = acc._final_pos
            front_thw = math.inf

        if not math.isnan(speed):
            acc.observe(
                action=action,
                reward=float(reward),
                speed_mps=speed,
                accel_mps2=accel,
                pos_m=pos,
                front_thw_s=front_thw,
            )

        last_reason = info.get("ego_terminated_reason", last_reason)

    acc.finish(
        collided=(last_reason == "collision"),
        reached_end=(last_reason == "exited"),
        truncated=truncated,
    )
    return acc.snapshot()


def run_episodes(
    agent: Policy,
    *,
    density: str,
    seeds: list[int],
    max_steps: int = 1000,
    warmup_steps: int = 50,
    env_kwargs: dict[str, Any] | None = None,
) -> list[EpisodeMetrics]:
    """Run one episode per seed. Constructs and tears down the env per run
    (one SUMO process per episode — clean isolation).

    For very large eval matrices, a future refactor could reuse a single env
    across seeds (faster, but tests determinism less rigorously). For the
    Week-3 eval matrix this isn't worth optimizing.
    """
    env_kwargs = dict(env_kwargs or {})
    out: list[EpisodeMetrics] = []
    for seed in seeds:
        env = LaneIQEnv(
            density=density,
            max_steps=max_steps,
            warmup_steps=warmup_steps,
            seed=seed,
            **env_kwargs,
        )
        try:
            out.append(run_episode(env, agent, seed=seed))
        finally:
            env.close()
    return out
