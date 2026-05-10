"""Tests for `laneiq.eval.metrics` — the per-episode collector and the
across-seeds aggregator. Pure-function tests; no SUMO needed."""

from __future__ import annotations

import math

import pytest

from laneiq.eval.metrics import (
    AggregatedMetrics,
    EpisodeMetrics,
    MetricsAccumulator,
    aggregate,
)


def _drive(
    *,
    n_steps: int,
    speeds: list[float],
    accels: list[float],
    actions: list[int],
    front_thws: list[float],
    rewards: list[float],
    initial_pos: float = 0.0,
    pos_step: float = 1.0,
) -> EpisodeMetrics:
    """Run the accumulator over a hand-crafted trajectory and snapshot."""
    acc = MetricsAccumulator(agent="test", density="medium", seed=42)
    acc.start(initial_pos=initial_pos)
    pos = initial_pos
    for i in range(n_steps):
        pos += pos_step
        acc.observe(
            action=actions[i],
            reward=rewards[i],
            speed_mps=speeds[i],
            accel_mps2=accels[i],
            pos_m=pos,
            front_thw_s=front_thws[i],
        )
    acc.finish(collided=False, reached_end=True, truncated=False)
    return acc.snapshot()


def test_snapshot_before_start_raises() -> None:
    acc = MetricsAccumulator(agent="x", density="low", seed=0)
    with pytest.raises(RuntimeError, match=r"snapshot\(\) called before start"):
        acc.snapshot()


def test_basic_summary_for_constant_trajectory() -> None:
    """Constant speed, no actions, no near-misses → predictable output."""
    n = 100
    em = _drive(
        n_steps=n,
        speeds=[20.0] * n,
        accels=[0.0] * n,
        actions=[0] * n,                 # all STAY
        front_thws=[5.0] * n,            # always safe
        rewards=[0.1] * n,
    )
    assert em.episode_length_steps == n
    assert em.episode_length_s == pytest.approx(10.0)
    assert em.mean_speed_mps == pytest.approx(20.0)
    assert em.lane_changes == 0
    assert em.lane_changes_per_min == 0.0
    assert em.jerk_rms == pytest.approx(0.0, abs=1e-9)
    assert em.near_miss_steps == 0
    assert em.return_total == pytest.approx(n * 0.1)


def test_lane_change_count_excludes_stay() -> None:
    em = _drive(
        n_steps=10,
        speeds=[20.0] * 10,
        accels=[0.0] * 10,
        actions=[0, 1, 0, 2, 0, 1, 0, 0, 2, 0],   # 4 non-STAY
        front_thws=[5.0] * 10,
        rewards=[0.0] * 10,
    )
    assert em.lane_changes == 4
    # 4 changes / (1 sim sec) = 240 / min
    assert em.lane_changes_per_min == pytest.approx(240.0)


def test_jerk_rms_matches_hand_calc() -> None:
    """accel sequence 0, 1, 3, 0 → diffs 1, 2, -3 → squares 1+4+9=14 → RMS=sqrt(14/4)."""
    em = _drive(
        n_steps=4,
        speeds=[10.0] * 4,
        accels=[0.0, 1.0, 3.0, 0.0],
        actions=[0] * 4,
        front_thws=[5.0] * 4,
        rewards=[0.0] * 4,
    )
    # Step 1: prev=0, curr=0 → no diff (first step has no prev)
    # Step 2: prev=0, curr=1 → diff=1
    # Step 3: prev=1, curr=3 → diff=2
    # Step 4: prev=3, curr=0 → diff=-3
    # Sum sq = 1+4+9 = 14; n=4; rms = sqrt(14/4)
    assert em.jerk_rms == pytest.approx(math.sqrt(14 / 4))


def test_near_miss_count_below_threshold() -> None:
    em = _drive(
        n_steps=10,
        speeds=[20.0] * 10,
        accels=[0.0] * 10,
        actions=[0] * 10,
        front_thws=[2.0, 1.4, 1.0, 1.6, 0.5, 5.0, 0.99, 1.49, 3.0, 1.5],
        # Below 1.5 strictly: 1.4, 1.0, 0.5, 0.99, 1.49 → 5 near-miss steps
        # (1.5 itself is NOT below 1.5)
        rewards=[0.0] * 10,
    )
    assert em.near_miss_steps == 5
    assert em.near_miss_rate == pytest.approx(0.5)


def test_near_miss_treats_inf_as_safe() -> None:
    em = _drive(
        n_steps=5,
        speeds=[20.0] * 5,
        accels=[0.0] * 5,
        actions=[0] * 5,
        front_thws=[math.inf] * 5,
        rewards=[0.0] * 5,
    )
    assert em.near_miss_steps == 0


def test_distance_uses_pos_delta() -> None:
    em = _drive(
        n_steps=10,
        speeds=[20.0] * 10,
        accels=[0.0] * 10,
        actions=[0] * 10,
        front_thws=[5.0] * 10,
        rewards=[0.0] * 10,
        initial_pos=100.0,
        pos_step=2.0,
    )
    # Final pos = 100 + 10*2 = 120; distance = 20
    assert em.distance_m == pytest.approx(20.0)


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------


def _ep(*, agent: str = "x", density: str = "medium", seed: int, **overrides) -> EpisodeMetrics:
    defaults = {
        "agent": agent,
        "density": density,
        "seed": seed,
        "mean_speed_mps": 20.0,
        "episode_length_steps": 100,
        "episode_length_s": 10.0,
        "distance_m": 200.0,
        "collided": False,
        "reached_end": True,
        "truncated": False,
        "lane_changes": 5,
        "lane_changes_per_min": 30.0,
        "jerk_rms": 0.5,
        "near_miss_steps": 2,
        "near_miss_rate": 0.02,
        "return_total": 1.0,
    }
    defaults.update(overrides)
    return EpisodeMetrics(**defaults)


def test_aggregate_means_and_collision_rate() -> None:
    eps = [
        _ep(seed=0, mean_speed_mps=18.0, return_total=2.0, collided=False),
        _ep(seed=1, mean_speed_mps=20.0, return_total=3.0, collided=True),
        _ep(seed=2, mean_speed_mps=22.0, return_total=1.0, collided=False),
    ]
    agg = aggregate(eps)
    assert isinstance(agg, AggregatedMetrics)
    assert agg.n_seeds == 3
    assert agg.mean_speed_mps_mean == pytest.approx(20.0)
    assert agg.return_total_mean == pytest.approx(2.0)
    assert agg.collision_rate == pytest.approx(1 / 3)


def test_aggregate_rejects_mixed_agents() -> None:
    eps = [_ep(agent="a", seed=0), _ep(agent="b", seed=1)]
    with pytest.raises(ValueError, match="mixed agents"):
        aggregate(eps)


def test_aggregate_rejects_mixed_densities() -> None:
    eps = [_ep(density="low", seed=0), _ep(density="high", seed=1)]
    with pytest.raises(ValueError, match="mixed densities"):
        aggregate(eps)


def test_aggregate_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        aggregate([])
