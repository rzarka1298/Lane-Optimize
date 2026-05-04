"""Tests for the observation builder.

Two layers, mirroring `tests/test_sumo_runtime.py`:

- 9 fast unit tests that drive `build_observation` against a synthetic mock
  TraCI (Mock-based; no SUMO process). One test per design property.
- 1 slow `@sumo`-marked integration test that builds an observation against
  a real running SUMO via `sumo_session()`.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest

from laneiq.env.observations import (
    OBS_DIM,
    OBS_NAMES,
    OBSERVATION_SPEC,
    ObservationBuilder,
    ObservationConfig,
    build_observation,
)

# ---------------------------------------------------------------------------
# Mock factory — builds a Mock traci_conn that responds to every call
# `build_observation` makes. Test cases override only what they need.
# ---------------------------------------------------------------------------

_DEFAULT_V_MAX = 36.11


def _make_mock_traci(
    *,
    ego_id: str = "ego",
    ego_speed: float = 18.05,
    ego_lane: int = 1,
    ego_accel: float = 2.25,
    ego_lat: float = 0.0,
    ego_pos: float = 500.0,
    leader: tuple[str, float, float] | None = None,        # (type, gap, speed)
    follower: tuple[str, float, float] | None = None,      # (type, gap, speed)
    side_neighbors: dict[tuple[int, str], tuple[str, float, float]] | None = None,
    extra_density_count: int = 0,
    v_max_lane: float = _DEFAULT_V_MAX,
    ego_in_list: bool = True,
) -> Mock:
    """Build a Mock traci_conn for a single observation call.

    Args:
        side_neighbors: keys are (lane_idx, "front"|"rear"), values are
            (vType, gap, speed). Only entries for valid (existing) lanes are
            honored; out-of-range lanes are silently ignored (mirrors SUMO).
        extra_density_count: how many additional in-window vehicles to add
            for the density feature, beyond any leader/follower/side neighbors.
    """
    side_neighbors = side_neighbors or {}

    # Track every "vehicle" we add so getIDList / getLanePosition are coherent.
    veh_data: dict[str, dict[str, Any]] = {}
    veh_ids: list[str] = []

    if ego_in_list:
        veh_ids.append(ego_id)

    # Same-lane leader / follower
    if leader is not None:
        l_type, l_gap, l_speed = leader
        l_id = "_leader"
        veh_ids.append(l_id)
        veh_data[l_id] = {"speed": l_speed, "type": l_type, "pos": ego_pos + l_gap, "lane": ego_lane}

    if follower is not None:
        f_type, f_gap, f_speed = follower
        f_id = "_follower"
        veh_ids.append(f_id)
        veh_data[f_id] = {"speed": f_speed, "type": f_type, "pos": ego_pos - f_gap, "lane": ego_lane}

    # Side neighbors keyed by (target_lane, slot). We pre-compute the four
    # SUMO modes the builder will call so getNeighbors returns the right
    # tuple per mode.
    mode_to_target = {
        0b011: (ego_lane + 1, "front"),  # left-leader
        0b001: (ego_lane + 1, "rear"),   # left-follower
        0b010: (ego_lane - 1, "front"),  # right-leader
        0b000: (ego_lane - 1, "rear"),   # right-follower
    }
    mode_results: dict[int, tuple[tuple[str, float], ...]] = {m: () for m in mode_to_target}
    for mode, (tlane, tslot) in mode_to_target.items():
        if (tlane, tslot) in side_neighbors:
            n_type, n_gap, n_speed = side_neighbors[(tlane, tslot)]
            n_id = f"_side_{tlane}_{tslot}"
            veh_ids.append(n_id)
            sign = +1 if tslot == "front" else -1
            veh_data[n_id] = {"speed": n_speed, "type": n_type, "pos": ego_pos + sign * n_gap, "lane": tlane}
            mode_results[mode] = ((n_id, n_gap),)

    # Extra in-window vehicles for density. Place them at distinct positions
    # all within max_gap_m of the ego.
    for i in range(extra_density_count):
        eid = f"_extra_{i}"
        veh_ids.append(eid)
        offset = (i + 1) * 5.0  # 5m, 10m, 15m, ...
        offset = ((-1) ** i) * offset  # alternate sides
        veh_data[eid] = {"speed": 0.0, "type": "normal", "pos": ego_pos + offset, "lane": ego_lane}

    mock = Mock()
    mock.vehicle.getIDList.return_value = tuple(veh_ids)
    mock.vehicle.getSpeed.side_effect = lambda v: ego_speed if v == ego_id else veh_data[v]["speed"]
    mock.vehicle.getLaneIndex.side_effect = lambda v: ego_lane if v == ego_id else veh_data[v]["lane"]
    mock.vehicle.getAcceleration.return_value = ego_accel
    mock.vehicle.getLateralLanePosition.return_value = ego_lat
    mock.vehicle.getLanePosition.side_effect = lambda v: ego_pos if v == ego_id else veh_data[v]["pos"]
    mock.vehicle.getTypeID.side_effect = lambda v: "ego" if v == ego_id else veh_data[v]["type"]

    mock.vehicle.getLeader.return_value = ("_leader", leader[1]) if leader else None
    mock.vehicle.getFollower.return_value = ("_follower", follower[1]) if follower else ("", -1.0)
    mock.vehicle.getNeighbors.side_effect = lambda _id, mode: mode_results.get(mode, ())

    mock.lane.getMaxSpeed.return_value = v_max_lane
    return mock


# ---------------------------------------------------------------------------
# Fast unit tests
# ---------------------------------------------------------------------------


def test_observation_spec_shape_and_bounds() -> None:
    """OBSERVATION_SPEC must declare a Box(-1, 1, (25,), float32)."""
    space = OBSERVATION_SPEC.space
    assert isinstance(space, gym.spaces.Box)
    assert space.shape == (OBS_DIM,)
    assert space.dtype == np.float32
    assert np.all(space.low == -1.0)
    assert np.all(space.high == 1.0)
    assert len(OBS_NAMES) == OBS_DIM


def test_build_observation_returns_correct_dtype_shape() -> None:
    """Default mock should produce a finite, in-shape observation."""
    obs = build_observation(_make_mock_traci(), "ego")
    assert obs.shape == (OBS_DIM,)
    assert obs.dtype == np.float32
    assert np.all(np.isfinite(obs))


def test_ego_state_normalization() -> None:
    """Ego dims [0..3] must reflect the normalized values exactly."""
    obs = build_observation(
        _make_mock_traci(ego_speed=18.055, ego_lane=1, ego_accel=2.25, ego_lat=0.0),
        "ego",
    )
    assert obs[0] == pytest.approx(0.5, abs=1e-3)   # 18.055 / 36.11
    assert obs[1] == pytest.approx(0.5, abs=1e-6)   # lane 1 / 2
    assert obs[2] == pytest.approx(0.5, abs=1e-6)   # 2.25 / 4.5
    assert obs[3] == pytest.approx(0.0, abs=1e-6)


def test_neighbor_missing_sentinel() -> None:
    """With no neighbors at all, every (gap, dv, vc) triple uses the sentinel."""
    obs = build_observation(_make_mock_traci(), "ego")
    for lane in (0, 1, 2):
        for slot_idx in range(2):
            offset = 4 + lane * 6 + slot_idx * 3
            assert obs[offset]     == pytest.approx(1.0)
            assert obs[offset + 1] == pytest.approx(0.0)
            assert obs[offset + 2] == pytest.approx(-1.0)


def test_neighbor_filled_slot_same_lane_front() -> None:
    """A leader at gap=50, dv=-5, type=aggressive populates dims [10..12]
    (lane 1 front slot since ego is in lane 1).
    """
    obs = build_observation(
        _make_mock_traci(
            ego_speed=20.0, ego_lane=1,
            leader=("aggressive", 50.0, 15.0),  # rel_v = 15 - 20 = -5
        ),
        "ego",
    )
    # Lane 1 front = base 4 + 1*6 + 0*3 = idx 10
    assert obs[10] == pytest.approx(0.5, abs=1e-6)              # 50 / 100
    assert obs[11] == pytest.approx(-5.0 / 36.11, abs=1e-3)     # dv / v_max
    assert obs[12] == pytest.approx(-0.5, abs=1e-6)             # aggressive bucket


def test_neighbor_filled_slot_left_leader() -> None:
    """A left-leader (lane 2 front when ego in lane 1) fills dims [16..18]."""
    obs = build_observation(
        _make_mock_traci(
            ego_speed=20.0, ego_lane=1,
            side_neighbors={(2, "front"): ("normal", 30.0, 25.0)},  # dv = +5
        ),
        "ego",
    )
    # Lane 2 front = base 4 + 2*6 + 0*3 = idx 16
    assert obs[16] == pytest.approx(0.3, abs=1e-6)
    assert obs[17] == pytest.approx(5.0 / 36.11, abs=1e-3)
    assert obs[18] == pytest.approx(0.0, abs=1e-6)              # normal bucket → (1-1)/2 = 0


@pytest.mark.parametrize(
    "ego_lane,expected",
    [
        (0, pytest.approx(2 / 3, abs=1e-6)),   # left ok, right not
        (1, pytest.approx(1.0)),                # both
        (2, pytest.approx(1 / 3, abs=1e-6)),   # right ok, left not
    ],
)
def test_lane_validity_packed_value(ego_lane: int, expected: Any) -> None:
    obs = build_observation(_make_mock_traci(ego_lane=ego_lane), "ego")
    assert obs[23] == expected


def test_local_density_clipped() -> None:
    """30 nearby vehicles → density 1.0 (clipped, not 1.5)."""
    obs = build_observation(
        _make_mock_traci(extra_density_count=30),
        "ego",
    )
    assert obs[22] == pytest.approx(1.0)


@pytest.mark.parametrize(
    "leader,ego_speed,expected_thw",
    [
        # gap=10, v=2 → THW=5s → clipped to 5 → / 5 = 1.0
        (("normal", 10.0, 5.0), 2.0, 1.0),
        # gap=10, v=10 → THW=1s → 1/5 = 0.2
        (("normal", 10.0, 12.0), 10.0, 0.2),
        # No leader → 1.0
        (None, 20.0, 1.0),
    ],
)
def test_thw_normalization_and_clip(
    leader: tuple[str, float, float] | None,
    ego_speed: float,
    expected_thw: float,
) -> None:
    obs = build_observation(
        _make_mock_traci(ego_speed=ego_speed, leader=leader),
        "ego",
    )
    assert obs[24] == pytest.approx(expected_thw, abs=1e-6)


def test_clip_range_invariant() -> None:
    """Adversarial inputs (huge speeds, tiny gaps) must still yield [-1, 1] dims."""
    obs = build_observation(
        _make_mock_traci(
            ego_speed=200.0,                # >> v_max
            ego_lane=2,
            ego_accel=-50.0,                # huge negative
            ego_lat=10.0,                   # way outside lane width
            leader=("aggressive", 0.1, 0.0),  # negative dv, tiny gap
            side_neighbors={
                (1, "front"): ("aggressive", 100.0, -50.0),
                (1, "rear"):  ("slow",       100.0, +50.0),
            },
            extra_density_count=100,
        ),
        "ego",
    )
    assert np.all(obs >= -1.0), f"underflow: {obs.min()}"
    assert np.all(obs <= 1.0), f"overflow: {obs.max()}"
    assert np.all(np.isfinite(obs))


def test_missing_ego_raises_keyerror() -> None:
    """Calling build_observation when ego is not in the vehicle list errors loudly."""
    with pytest.raises(KeyError, match="not in vehicle list"):
        build_observation(_make_mock_traci(ego_in_list=False), "ego")


def test_observation_builder_caches_v_max_lane() -> None:
    """ObservationBuilder.on_episode_start should pre-cache v_max_lane so
    subsequent build() calls don't re-query lane.getMaxSpeed().
    """
    mock = _make_mock_traci()
    builder = ObservationBuilder(mock, config=ObservationConfig())
    builder.on_episode_start("ego")
    assert mock.lane.getMaxSpeed.call_count == 1
    builder.build("ego")
    builder.build("ego")
    # Still 1 — subsequent builds use the cached value.
    assert mock.lane.getMaxSpeed.call_count == 1
    builder.on_episode_end()


# ---------------------------------------------------------------------------
# Slow integration test — actually launches SUMO
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.sumo
def test_observation_against_running_sumo() -> None:
    """End-to-end: open SUMO, warm up traffic, add ego, build observation.

    Asserts shape, dtype, finite-ness, in-range, and that the ego_lane_idx
    dimension matches what TraCI reports for the just-added ego.
    """
    import traci

    from laneiq.env.sumo_runtime import sumo_session

    with sumo_session(seed=42, end_time=120.0):
        # Warm up so background traffic populates.
        for _ in range(50):
            traci.simulationStep()

        traci.vehicle.add(
            vehID="ego_test",
            routeID="r_through",
            typeID="ego",
            depart="now",
            departLane="1",
            departSpeed="20.0",
        )
        traci.vehicle.setLaneChangeMode("ego_test", 0)
        traci.simulationStep()  # ego now appears in getIDList

        obs = build_observation(traci, "ego_test")

        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32
        assert np.all(np.isfinite(obs)), f"non-finite in obs: {obs}"
        assert OBSERVATION_SPEC.space.contains(obs), f"out of [-1,1]: {obs}"

        # ego_lane_idx (dim 1) should reflect the lane TraCI reports.
        actual_lane = traci.vehicle.getLaneIndex("ego_test")
        expected_dim1 = max(0, min(2, actual_lane)) / 2.0
        assert obs[1] == pytest.approx(expected_dim1, abs=1e-6)
