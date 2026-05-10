"""Per-episode evaluation metrics.

Implements the PRD-required metric set: performance (mean speed, episode
length), safety (collision flag, near-miss count), and behavior quality
(lane changes/min, jerk RMS, mean reward).

`MetricsAccumulator` is fed once per step inside `run_episode()`; it owns
all the running statistics. `EpisodeMetrics` is the frozen output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from laneiq.env.actions import Action


@dataclass(frozen=True)
class EpisodeMetrics:
    """All numbers a single eval episode produces.

    Aggregated downstream (mean ± std across seeds) by `compare.py`.
    """

    # Identification
    agent: str                  # name of the policy
    density: str                # "low" / "medium" / "high"
    seed: int                   # the per-episode seed

    # Performance
    mean_speed_mps: float       # average ego speed (m/s) over the episode
    episode_length_steps: int   # how many env.step() calls
    episode_length_s: float     # = steps * 0.1 (sim seconds)
    distance_m: float           # how far the ego traveled (m)

    # Outcome
    collided: bool              # True iff the ego was in collision list at termination
    reached_end: bool           # True iff terminated by exiting the road (not crash, not truncate)
    truncated: bool             # True iff hit max_steps without terminating

    # Behavior quality
    lane_changes: int           # count of non-STAY actions executed
    lane_changes_per_min: float # = lane_changes / (episode_length_s / 60)
    jerk_rms: float             # sqrt(mean(Δaccel²)); m/s³-ish (per-step)
    near_miss_steps: int        # count of steps where front_thw < 1.5s
    near_miss_rate: float       # = near_miss_steps / episode_length_steps

    # RL signal
    return_total: float         # sum of per-step rewards


class MetricsAccumulator:
    """Stateful collector fed once per env.step() inside the eval runner.

    Use:
        acc = MetricsAccumulator(agent="dqn", density="medium", seed=42)
        ... env.reset() ...
        acc.start(initial_pos=ego_x_at_reset)
        for step in episode:
            obs, reward, terminated, truncated, info = env.step(action)
            acc.observe(action, reward, info, current_speed, current_accel,
                        current_pos, front_thw)
        acc.finish(terminated_reason="collision" | "exited", truncated=...)
        metrics = acc.snapshot()
    """

    NEAR_MISS_THW_S: float = 1.5

    def __init__(self, *, agent: str, density: str, seed: int) -> None:
        self._agent = agent
        self._density = density
        self._seed = seed

        self._step_count = 0
        self._lane_changes = 0
        self._near_miss_steps = 0
        self._return = 0.0

        self._initial_pos: float | None = None
        self._final_pos: float = 0.0
        self._sum_speed: float = 0.0

        # For jerk RMS: we accumulate (accel_t - accel_{t-1})² across steps.
        self._sum_accel_diff_sq: float = 0.0
        self._prev_accel: float = 0.0
        self._has_prev_accel: bool = False

        self._collided: bool = False
        self._reached_end: bool = False
        self._truncated: bool = False

    def start(self, *, initial_pos: float) -> None:
        """Record the ego's starting position for distance computation."""
        self._initial_pos = initial_pos
        self._final_pos = initial_pos

    def observe(
        self,
        action: int,
        reward: float,
        speed_mps: float,
        accel_mps2: float,
        pos_m: float,
        front_thw_s: float,
    ) -> None:
        """One per env.step() — all post-step values."""
        self._step_count += 1
        self._return += float(reward)
        self._sum_speed += speed_mps
        self._final_pos = pos_m

        if int(action) != int(Action.STAY):
            self._lane_changes += 1

        if math.isfinite(front_thw_s) and front_thw_s < self.NEAR_MISS_THW_S:
            self._near_miss_steps += 1

        if self._has_prev_accel:
            d = accel_mps2 - self._prev_accel
            self._sum_accel_diff_sq += d * d
        self._prev_accel = accel_mps2
        self._has_prev_accel = True

    def finish(
        self,
        *,
        collided: bool,
        reached_end: bool,
        truncated: bool,
    ) -> None:
        self._collided = collided
        self._reached_end = reached_end
        self._truncated = truncated

    def snapshot(self) -> EpisodeMetrics:
        if self._initial_pos is None:
            raise RuntimeError("snapshot() called before start()")
        n = max(self._step_count, 1)
        episode_length_s = self._step_count * 0.1
        mean_speed = self._sum_speed / n
        jerk_rms = math.sqrt(self._sum_accel_diff_sq / n)
        lc_per_min = (self._lane_changes / episode_length_s) * 60.0 if episode_length_s > 0 else 0.0
        near_miss_rate = self._near_miss_steps / n
        distance = max(0.0, self._final_pos - self._initial_pos)

        return EpisodeMetrics(
            agent=self._agent,
            density=self._density,
            seed=self._seed,
            mean_speed_mps=mean_speed,
            episode_length_steps=self._step_count,
            episode_length_s=episode_length_s,
            distance_m=distance,
            collided=self._collided,
            reached_end=self._reached_end,
            truncated=self._truncated,
            lane_changes=self._lane_changes,
            lane_changes_per_min=lc_per_min,
            jerk_rms=jerk_rms,
            near_miss_steps=self._near_miss_steps,
            near_miss_rate=near_miss_rate,
            return_total=self._return,
        )


# ---------------------------------------------------------------------------
# Aggregation across episodes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AggregatedMetrics:
    """Mean ± std for one (agent, density) cell across N seeds."""

    agent: str
    density: str
    n_seeds: int
    mean_speed_mps_mean: float
    mean_speed_mps_std: float
    return_total_mean: float
    return_total_std: float
    collision_rate: float           # fraction of episodes with collision
    reached_end_rate: float         # fraction of episodes that reached end
    lane_changes_per_min_mean: float
    lane_changes_per_min_std: float
    jerk_rms_mean: float
    jerk_rms_std: float
    near_miss_rate_mean: float
    near_miss_rate_std: float
    episode_length_s_mean: float


def aggregate(episodes: list[EpisodeMetrics]) -> AggregatedMetrics:
    """Reduce per-episode metrics to (mean, std) across seeds.

    Requires all input episodes to share (agent, density). Raises if not.
    """
    if not episodes:
        raise ValueError("cannot aggregate empty episode list")
    agents = {e.agent for e in episodes}
    densities = {e.density for e in episodes}
    if len(agents) != 1:
        raise ValueError(f"mixed agents in aggregation: {agents}")
    if len(densities) != 1:
        raise ValueError(f"mixed densities in aggregation: {densities}")

    speeds = np.array([e.mean_speed_mps for e in episodes])
    returns = np.array([e.return_total for e in episodes])
    lc_pm = np.array([e.lane_changes_per_min for e in episodes])
    jerk = np.array([e.jerk_rms for e in episodes])
    near_miss = np.array([e.near_miss_rate for e in episodes])
    lengths = np.array([e.episode_length_s for e in episodes])

    return AggregatedMetrics(
        agent=episodes[0].agent,
        density=episodes[0].density,
        n_seeds=len(episodes),
        mean_speed_mps_mean=float(speeds.mean()),
        mean_speed_mps_std=float(speeds.std(ddof=0)),
        return_total_mean=float(returns.mean()),
        return_total_std=float(returns.std(ddof=0)),
        collision_rate=float(np.mean([e.collided for e in episodes])),
        reached_end_rate=float(np.mean([e.reached_end for e in episodes])),
        lane_changes_per_min_mean=float(lc_pm.mean()),
        lane_changes_per_min_std=float(lc_pm.std(ddof=0)),
        jerk_rms_mean=float(jerk.mean()),
        jerk_rms_std=float(jerk.std(ddof=0)),
        near_miss_rate_mean=float(near_miss.mean()),
        near_miss_rate_std=float(near_miss.std(ddof=0)),
        episode_length_s_mean=float(lengths.mean()),
    )
