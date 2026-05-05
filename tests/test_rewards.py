"""Tests for the reward composer.

Pure-function tests — no SUMO needed. Each component gets at least one
direct test plus a few invariants.
"""

from __future__ import annotations

import math
from dataclasses import replace

import pytest

from laneiq.env.actions import Action
from laneiq.env.rewards import (
    COMPONENT_KEYS,
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    RewardState,
    compute_reward,
    reward_state_at_rest,
)

# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _state(
    *,
    speed: float = 18.05,
    accel: float = 0.0,
    front_gap: float = 50.0,
    front_thw: float = 2.5,
    collided: bool = False,
    lane_idx: int = 1,
) -> RewardState:
    """Compact RewardState builder. All defaults are 'normal cruising'."""
    return RewardState(
        speed=speed,
        accel=accel,
        front_gap=front_gap,
        front_thw=front_thw,
        collided=collided,
        lane_idx=lane_idx,
    )


# ---------------------------------------------------------------------------
# Config + module-level invariants
# ---------------------------------------------------------------------------


def test_reward_config_defaults_match_plan() -> None:
    """The v1 weights from the master plan must match what's in code.

    If you intentionally tune them, update both the plan and this test.
    """
    cfg = RewardConfig()
    assert cfg.w_speed == 1.0
    assert cfg.w_lane_cost == -0.05
    assert cfg.w_jerk == -0.2
    assert cfg.w_unsafe_gap == -0.5
    assert cfg.w_collision == -50.0
    assert cfg.w_alive == 0.01
    assert cfg.unsafe_thw_s == 1.5
    assert cfg.a_max_norm == 4.5
    assert cfg.v_max == 36.11


def test_default_config_is_singleton() -> None:
    """DEFAULT_REWARD_CONFIG should be equal to a fresh RewardConfig()."""
    assert RewardConfig() == DEFAULT_REWARD_CONFIG


def test_component_keys_match_returned_dict() -> None:
    """COMPONENT_KEYS must enumerate every key in the returned dict."""
    _, components = compute_reward(_state(), Action.STAY, _state())
    assert tuple(components.keys()) == COMPONENT_KEYS


def test_components_sum_to_total() -> None:
    """Total returned must equal sum(components.values()) exactly."""
    total, components = compute_reward(_state(speed=10), Action.CHANGE_LEFT, _state(speed=18))
    assert total == sum(components.values())


# ---------------------------------------------------------------------------
# Speed component
# ---------------------------------------------------------------------------


def test_speed_component_zero_at_zero_speed() -> None:
    _, c = compute_reward(_state(speed=0), Action.STAY, _state(speed=0))
    assert c["speed"] == 0.0


def test_speed_component_at_v_max() -> None:
    """Cruising at the lane speed limit → w_speed * 1.0."""
    _, c = compute_reward(
        _state(speed=36.11),
        Action.STAY,
        _state(speed=36.11),
    )
    assert c["speed"] == pytest.approx(1.0, abs=1e-6)


def test_speed_component_clipped_above_v_max() -> None:
    """Aggressive vType could exceed v_max momentarily; speed term clips at 1."""
    _, c = compute_reward(_state(speed=40), Action.STAY, _state(speed=40))
    assert c["speed"] == pytest.approx(1.0, abs=1e-6)


def test_speed_component_uses_nxt_not_prev() -> None:
    """Speed term reads `nxt.speed`, not `prev.speed`."""
    _, c = compute_reward(_state(speed=0), Action.STAY, _state(speed=18.055))
    assert c["speed"] == pytest.approx(0.5, abs=1e-3)


# ---------------------------------------------------------------------------
# Lane-cost component
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "action,expected_cost",
    [
        (Action.STAY, 0.0),
        (Action.CHANGE_LEFT, -0.05),
        (Action.CHANGE_RIGHT, -0.05),
        (0, 0.0),    # raw int STAY
        (1, -0.05),  # raw int CHANGE_LEFT
        (2, -0.05),  # raw int CHANGE_RIGHT
    ],
)
def test_lane_cost_component(action: Action | int, expected_cost: float) -> None:
    _, c = compute_reward(_state(), action, _state())
    assert c["lane_cost"] == pytest.approx(expected_cost)


# ---------------------------------------------------------------------------
# Jerk component
# ---------------------------------------------------------------------------


def test_jerk_zero_when_accel_unchanged() -> None:
    _, c = compute_reward(_state(accel=2.0), Action.STAY, _state(accel=2.0))
    assert c["jerk"] == 0.0


def test_jerk_proportional_to_delta_accel() -> None:
    """|Δa|=2.25 → jerk = w_jerk * (2.25 / 4.5) = -0.1."""
    _, c = compute_reward(_state(accel=0.0), Action.STAY, _state(accel=2.25))
    assert c["jerk"] == pytest.approx(-0.1, abs=1e-6)


def test_jerk_uses_absolute_value() -> None:
    """Sign of Δaccel doesn't matter — both braking and accelerating count."""
    _, c_accel = compute_reward(_state(accel=0.0), Action.STAY, _state(accel=2.0))
    _, c_brake = compute_reward(_state(accel=0.0), Action.STAY, _state(accel=-2.0))
    assert c_accel["jerk"] == c_brake["jerk"]


def test_first_step_zero_jerk_via_at_rest_helper() -> None:
    """Using reward_state_at_rest as `prev` on step 0 yields zero jerk."""
    nxt = _state(accel=2.0)
    _, c = compute_reward(reward_state_at_rest(), Action.STAY, nxt)
    # jerk = w_jerk * (|2-0| / 4.5) = -0.0888...
    # NOT zero — this just documents that the helper is "neutral", not that
    # callers should special-case the first step. If you want zero jerk on
    # step 0, pass `prev = nxt`.
    assert c["jerk"] != 0.0
    # And the prev=nxt trick:
    _, c0 = compute_reward(nxt, Action.STAY, nxt)
    assert c0["jerk"] == 0.0


# ---------------------------------------------------------------------------
# Unsafe-gap component
# ---------------------------------------------------------------------------


def test_unsafe_gap_fires_below_threshold() -> None:
    _, c = compute_reward(_state(), Action.STAY, _state(front_thw=1.0))
    assert c["unsafe_gap"] == pytest.approx(-0.5)


def test_unsafe_gap_safe_above_threshold() -> None:
    _, c = compute_reward(_state(), Action.STAY, _state(front_thw=2.0))
    assert c["unsafe_gap"] == 0.0


def test_unsafe_gap_at_exact_threshold_is_safe() -> None:
    """The comparison is strict-less-than; equality is not unsafe."""
    _, c = compute_reward(
        _state(),
        Action.STAY,
        _state(front_thw=DEFAULT_REWARD_CONFIG.unsafe_thw_s),
    )
    assert c["unsafe_gap"] == 0.0


def test_unsafe_gap_no_leader_is_safe() -> None:
    """No leader → THW=inf → no unsafe penalty."""
    _, c = compute_reward(
        _state(),
        Action.STAY,
        _state(front_gap=math.inf, front_thw=math.inf),
    )
    assert c["unsafe_gap"] == 0.0


# ---------------------------------------------------------------------------
# Collision component
# ---------------------------------------------------------------------------


def test_collision_component_fires_on_collision() -> None:
    _, c = compute_reward(_state(), Action.STAY, _state(collided=True))
    assert c["collision"] == pytest.approx(-50.0)


def test_collision_component_zero_when_safe() -> None:
    _, c = compute_reward(_state(), Action.STAY, _state(collided=False))
    assert c["collision"] == 0.0


def test_total_dominated_by_collision() -> None:
    """Even at max speed cruising, collision pulls the total massively negative."""
    total, _ = compute_reward(
        _state(speed=36.11),
        Action.STAY,
        _state(speed=36.11, collided=True),
    )
    assert total < -45  # speed=+1 + alive=+0.01 + collision=-50 → ~-49


# ---------------------------------------------------------------------------
# Alive component
# ---------------------------------------------------------------------------


def test_alive_component_always_positive() -> None:
    """Per-step alive bonus is constant (= w_alive) regardless of state."""
    for state in [_state(), _state(collided=True), _state(speed=0)]:
        _, c = compute_reward(_state(), Action.STAY, state)
        assert c["alive"] == pytest.approx(0.01)


# ---------------------------------------------------------------------------
# Custom config
# ---------------------------------------------------------------------------


def test_custom_config_overrides_weights() -> None:
    """A safety-heavy config should produce a more-negative total when
    the only thing changing is the unsafe-gap multiplier."""
    safety_cfg = replace(DEFAULT_REWARD_CONFIG, w_unsafe_gap=-2.0)
    _, c_default = compute_reward(_state(), Action.STAY, _state(front_thw=1.0))
    _, c_safety = compute_reward(_state(), Action.STAY, _state(front_thw=1.0), config=safety_cfg)
    assert c_safety["unsafe_gap"] == pytest.approx(-2.0)
    assert c_default["unsafe_gap"] == pytest.approx(-0.5)


def test_zero_v_max_does_not_crash() -> None:
    """Defensive: v_max=0 gives speed_term=0 instead of div-by-zero."""
    cfg = replace(DEFAULT_REWARD_CONFIG, v_max=0.0)
    _, c = compute_reward(_state(speed=20), Action.STAY, _state(speed=20), config=cfg)
    assert c["speed"] == 0.0


def test_zero_a_max_norm_does_not_crash() -> None:
    """Defensive: a_max_norm=0 gives jerk_term=0 instead of div-by-zero."""
    cfg = replace(DEFAULT_REWARD_CONFIG, a_max_norm=0.0)
    _, c = compute_reward(_state(accel=0), Action.STAY, _state(accel=2), config=cfg)
    assert c["jerk"] == 0.0


# ---------------------------------------------------------------------------
# Realistic compound scenarios
# ---------------------------------------------------------------------------


def test_smooth_cruising_yields_small_positive() -> None:
    """Cruising at moderate speed, no leader, no action — should be small +."""
    state = _state(speed=25.0, accel=0.0, front_gap=math.inf, front_thw=math.inf)
    total, _ = compute_reward(state, Action.STAY, state)
    # speed: 25/36.11 ≈ 0.692
    # alive: 0.01
    # everything else: 0
    assert total == pytest.approx(0.692 + 0.01, abs=1e-2)


def test_lane_change_into_safe_gap_modest_cost() -> None:
    """Cost of a lane change should be small relative to the speed reward."""
    prev = _state(speed=20, accel=0)
    nxt = _state(speed=22, accel=0.5, front_thw=3.0)
    total, c = compute_reward(prev, Action.CHANGE_LEFT, nxt)
    assert c["lane_cost"] == -0.05  # not zero
    assert total > 0   # net positive — the speed gain wins
