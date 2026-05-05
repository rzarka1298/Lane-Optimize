"""Tests for the action enum, applier, and validity mask.

Mirrors the layered pattern used elsewhere in the suite: many fast unit
tests that drive against a `Mock` TraCI, plus one slow integration test
that actually moves a vehicle in a running SUMO.
"""

from __future__ import annotations

from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from laneiq.env.actions import (
    ACTION_NAMES,
    ACTION_SPACE,
    N_ACTIONS,
    Action,
    apply_action,
    disable_sumo_lane_change,
    mask_invalid_q_values,
    valid_actions_mask,
)
from laneiq.env.observations import OBS_DIM

# ---------------------------------------------------------------------------
# Enum / space invariants
# ---------------------------------------------------------------------------


def test_action_enum_values() -> None:
    assert int(Action.STAY) == 0
    assert int(Action.CHANGE_LEFT) == 1
    assert int(Action.CHANGE_RIGHT) == 2
    assert N_ACTIONS == 3
    assert ACTION_NAMES == ("stay", "change_left", "change_right")


def test_action_space_is_discrete_3() -> None:
    assert isinstance(ACTION_SPACE, gym.spaces.Discrete)
    assert ACTION_SPACE.n == 3


# ---------------------------------------------------------------------------
# disable_sumo_lane_change
# ---------------------------------------------------------------------------


def test_disable_sumo_lane_change_calls_set_mode_zero() -> None:
    """Must call setLaneChangeMode(ego, 0) — no other args, no other calls."""
    mock = Mock()
    disable_sumo_lane_change(mock, "ego")
    mock.vehicle.setLaneChangeMode.assert_called_once_with("ego", 0)


# ---------------------------------------------------------------------------
# apply_action — STAY is a no-op
# ---------------------------------------------------------------------------


def test_apply_action_stay_is_noop() -> None:
    mock = Mock()
    apply_action(mock, "ego", Action.STAY)
    mock.vehicle.changeLane.assert_not_called()
    mock.vehicle.getLaneIndex.assert_not_called()


# ---------------------------------------------------------------------------
# apply_action — CHANGE_LEFT / CHANGE_RIGHT issue the right TraCI call
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current_lane,action,expected_target",
    [
        (0, Action.CHANGE_LEFT, 1),
        (1, Action.CHANGE_LEFT, 2),
        (1, Action.CHANGE_RIGHT, 0),
        (2, Action.CHANGE_RIGHT, 1),
    ],
)
def test_apply_action_valid_transitions(
    current_lane: int, action: Action, expected_target: int,
) -> None:
    mock = Mock()
    mock.vehicle.getLaneIndex.return_value = current_lane
    apply_action(mock, "ego", action, duration_s=1.0)
    mock.vehicle.changeLane.assert_called_once_with("ego", expected_target, 1.0)


def test_apply_action_accepts_raw_int() -> None:
    """Convenience: env may pass through the policy's raw int output."""
    mock = Mock()
    mock.vehicle.getLaneIndex.return_value = 1
    apply_action(mock, "ego", 1)  # 1 == CHANGE_LEFT
    mock.vehicle.changeLane.assert_called_once_with("ego", 2, 1.0)


def test_apply_action_custom_duration() -> None:
    mock = Mock()
    mock.vehicle.getLaneIndex.return_value = 1
    apply_action(mock, "ego", Action.CHANGE_LEFT, duration_s=2.5)
    mock.vehicle.changeLane.assert_called_once_with("ego", 2, 2.5)


# ---------------------------------------------------------------------------
# apply_action — invalid target → silent no-op (defense in depth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "current_lane,action",
    [
        (0, Action.CHANGE_RIGHT),  # nothing to the right of lane 0
        (2, Action.CHANGE_LEFT),   # nothing to the left of lane 2
    ],
)
def test_apply_action_invalid_target_is_silent_noop(
    current_lane: int, action: Action,
) -> None:
    mock = Mock()
    mock.vehicle.getLaneIndex.return_value = current_lane
    apply_action(mock, "ego", action)
    mock.vehicle.changeLane.assert_not_called()


# ---------------------------------------------------------------------------
# valid_actions_mask
# ---------------------------------------------------------------------------


def _obs_with_validity(packed: float) -> np.ndarray:
    """Build a minimal valid observation with the given lane_validity_packed."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[23] = packed
    return obs


@pytest.mark.parametrize(
    "packed,expected",
    [
        # lane 0: left ok, right not → packed = 2/3
        (2 / 3, [True, True, False]),
        # lane 1: both → packed = 1.0
        (1.0, [True, True, True]),
        # lane 2: right ok, left not → packed = 1/3
        (1 / 3, [True, False, True]),
        # extreme: nothing valid (shouldn't happen but mask must handle)
        (0.0, [True, False, False]),
    ],
)
def test_valid_actions_mask_values(packed: float, expected: list[bool]) -> None:
    obs = _obs_with_validity(packed)
    mask = valid_actions_mask(obs)
    assert mask.shape == (3,)
    assert mask.dtype == bool
    assert mask.tolist() == expected


def test_valid_actions_mask_rejects_wrong_shape() -> None:
    with pytest.raises(ValueError, match=r"expected shape \(25,\)"):
        valid_actions_mask(np.zeros(10, dtype=np.float32))


def test_valid_actions_mask_robust_to_float_drift() -> None:
    """np.clip and float roundoff might leave packed slightly off the
    canonical value; round-then-clamp must still recover the right bits.
    """
    # 2/3 ± tiny noise should still decode to (left=1, right=0)
    for delta in (-1e-5, +1e-5):
        obs = _obs_with_validity(2 / 3 + delta)
        mask = valid_actions_mask(obs)
        assert mask.tolist() == [True, True, False]


# ---------------------------------------------------------------------------
# mask_invalid_q_values
# ---------------------------------------------------------------------------


def test_mask_invalid_q_values_basic() -> None:
    q = np.array([1.5, 2.5, 0.5])
    mask = np.array([True, True, False])
    out = mask_invalid_q_values(q, mask)
    assert out[0] == 1.5
    assert out[1] == 2.5
    assert out[2] == -np.inf


def test_mask_invalid_q_values_does_not_mutate_input() -> None:
    q = np.array([1.0, 2.0, 3.0])
    mask = np.array([True, False, True])
    out = mask_invalid_q_values(q, mask)
    # Input untouched
    assert q.tolist() == [1.0, 2.0, 3.0]
    # Output has -inf at the masked position
    assert out[1] == -np.inf


def test_mask_invalid_q_values_argmax_safety() -> None:
    """Even if the network's highest Q is on an invalid action, argmax of
    the masked Qs picks a valid action."""
    q = np.array([0.1, 99.9, 0.5])    # network "wants" CHANGE_LEFT
    mask = np.array([True, False, True])
    out = mask_invalid_q_values(q, mask)
    assert int(np.argmax(out)) == 2   # CHANGE_RIGHT — only remaining valid


def test_mask_invalid_q_values_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        mask_invalid_q_values(np.zeros(3), np.zeros(2, dtype=bool))


# ---------------------------------------------------------------------------
# Slow integration test — actually moves a vehicle in SUMO
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.sumo
def test_apply_action_changes_lane_in_running_sumo() -> None:
    """End-to-end: open SUMO, add ego in lane 1, request CHANGE_LEFT,
    step a few sim-seconds, verify the ego is now in lane 2.
    """
    import traci

    from laneiq.env.sumo_runtime import sumo_session

    with sumo_session(seed=123, end_time=120.0):
        # Warm up so background traffic populates.
        for _ in range(50):
            traci.simulationStep()

        traci.vehicle.add(
            vehID="ego_lc",
            routeID="r_through",
            typeID="ego",
            depart="now",
            departLane="1",
            departSpeed="20.0",
        )
        disable_sumo_lane_change(traci, "ego_lc")
        traci.simulationStep()  # ego now in vehicle list
        assert traci.vehicle.getLaneIndex("ego_lc") == 1

        # Request a left lane change with a short duration so we see the
        # transition complete within ~20 sim-steps (2 sim-seconds).
        apply_action(traci, "ego_lc", Action.CHANGE_LEFT, duration_s=1.0)

        # Step until either the lane change completes or we time out.
        for _ in range(40):  # up to 4 sim-seconds
            traci.simulationStep()
            if traci.vehicle.getLaneIndex("ego_lc") == 2:
                break
        else:
            pytest.fail("ego did not reach lane 2 within 4 sim-seconds")
