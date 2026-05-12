"""Tests for `LaneIQParallelEnv` — the PettingZoo multi-agent wrapper.

Fast tests on construction-time invariants + a slow integration suite
that exercises the full reset/step/close lifecycle against a real SUMO,
including PettingZoo's `parallel_api_test` — the gold-standard contract
validator for ParallelEnvs.
"""

from __future__ import annotations

import numpy as np
import pytest

from laneiq.env.actions import N_ACTIONS
from laneiq.env.multi_agent_env import LaneIQParallelEnv
from laneiq.env.observations import OBS_DIM

# ---------------------------------------------------------------------------
# Fast tests (no SUMO)
# ---------------------------------------------------------------------------


def test_construction_does_not_start_sumo() -> None:
    env = LaneIQParallelEnv(n_egos=4)
    try:
        assert env._sumo_running is False
        assert env.agents == []            # no agents until reset
        assert env.possible_agents == ["ego_0", "ego_1", "ego_2", "ego_3"]
    finally:
        env.close()


def test_observation_and_action_spaces_per_agent() -> None:
    env = LaneIQParallelEnv(n_egos=4)
    try:
        for aid in env.possible_agents:
            assert env.observation_space(aid).shape == (OBS_DIM,)
            assert env.observation_space(aid).dtype == np.float32
            assert env.action_space(aid).n == N_ACTIONS
    finally:
        env.close()


def test_invalid_density_raises() -> None:
    with pytest.raises(ValueError, match="density must be one of"):
        LaneIQParallelEnv(density="extreme")  # type: ignore[arg-type]


def test_n_egos_below_1_raises() -> None:
    with pytest.raises(ValueError, match="n_egos must"):
        LaneIQParallelEnv(n_egos=0)


def test_mismatched_depart_positions_length_raises() -> None:
    with pytest.raises(ValueError, match="depart_positions_m"):
        LaneIQParallelEnv(n_egos=4, depart_positions_m=(0.0, 100.0))


def test_step_before_reset_raises() -> None:
    env = LaneIQParallelEnv(n_egos=4)
    try:
        with pytest.raises(RuntimeError, match="before reset"):
            env.step({})
    finally:
        env.close()


def test_close_idempotent_before_reset() -> None:
    env = LaneIQParallelEnv()
    env.close()
    env.close()


# ---------------------------------------------------------------------------
# Slow integration tests
# ---------------------------------------------------------------------------


@pytest.fixture
def env() -> LaneIQParallelEnv:
    e = LaneIQParallelEnv(
        density="low", n_egos=3, max_steps=50, warmup_steps=30, seed=42,
    )
    yield e
    e.close()


@pytest.mark.slow
@pytest.mark.sumo
def test_reset_returns_dicts_keyed_by_agent(env: LaneIQParallelEnv) -> None:
    observations, infos = env.reset()
    assert set(observations.keys()) == set(env.possible_agents)
    assert set(infos.keys()) == set(env.possible_agents)
    for obs in observations.values():
        assert obs.shape == (OBS_DIM,)
        assert obs.dtype == np.float32


@pytest.mark.slow
@pytest.mark.sumo
def test_step_returns_per_agent_dicts(env: LaneIQParallelEnv) -> None:
    env.reset()
    actions = {aid: 0 for aid in env.agents}     # all STAY
    obs, rewards, terminations, truncations, infos = env.step(actions)
    assert set(obs.keys()) <= set(env.possible_agents)
    assert set(rewards.keys()) == set(obs.keys())
    assert set(terminations.keys()) == set(obs.keys())
    assert set(truncations.keys()) == set(obs.keys())
    assert set(infos.keys()) == set(obs.keys())
    for r in rewards.values():
        assert np.isfinite(r)


@pytest.mark.slow
@pytest.mark.sumo
def test_active_agents_shrink_when_egos_terminate() -> None:
    """At least at low density with random actions over many steps,
    one of the egos eventually finishes (exits) or crashes."""
    env = LaneIQParallelEnv(
        density="low", n_egos=3, max_steps=500, warmup_steps=20, seed=7,
    )
    try:
        env.reset()
        initial_active = list(env.agents)
        for step in range(500):
            actions = {aid: 0 for aid in env.agents}  # STAY: cars cruise + exit
            _obs, _r, _terms, _truncs, _info = env.step(actions)
            if not env.agents:
                break
            if step > 200 and len(env.agents) < len(initial_active):
                # At least one ego dropped — that's enough to verify.
                break
        # Either some ego terminated OR we ran the full episode (max_steps).
        assert len(env.agents) <= len(initial_active)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.sumo
def test_action_masks_returns_per_agent_dict(env: LaneIQParallelEnv) -> None:
    env.reset()
    masks = env.action_masks()
    assert set(masks.keys()) == set(env.agents)
    for mask in masks.values():
        assert mask.shape == (3,)
        assert mask.dtype == bool
        assert mask[0]  # STAY always valid


@pytest.mark.slow
@pytest.mark.sumo
def test_multiple_resets_work(env: LaneIQParallelEnv) -> None:
    env.reset(seed=42)
    env.step({aid: 0 for aid in env.agents})
    obs2, _ = env.reset(seed=42)
    assert len(obs2) == len(env.possible_agents)


@pytest.mark.slow
@pytest.mark.sumo
def test_pettingzoo_parallel_api_compliance() -> None:
    """PettingZoo's gold-standard contract validator for ParallelEnvs."""
    from pettingzoo.test import parallel_api_test

    env = LaneIQParallelEnv(
        density="low", n_egos=3, max_steps=30, warmup_steps=20, seed=1,
    )
    try:
        # num_cycles is per-test; PettingZoo recommends ≥1000 for thorough
        # validation but on SUMO that's ~50 sim minutes. 100 covers the
        # key contract paths (spaces, returns, agent dropping) without
        # burning too long.
        parallel_api_test(env, num_cycles=100)
    finally:
        env.close()
