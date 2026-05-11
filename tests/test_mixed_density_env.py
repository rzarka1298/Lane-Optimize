"""Tests for `MixedDensityEnv`. Fast unit tests on construction + sampling
invariants; the slow integration test confirms reset/step actually flow
through to a real SUMO."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest

from laneiq.env.mixed_density_env import MixedDensityEnv
from laneiq.env.observations import OBS_DIM


def test_construction_does_not_start_sumo() -> None:
    """__init__ must be cheap; inner env constructed lazily on reset()."""
    env = MixedDensityEnv()
    try:
        assert env._inner is None
        assert env.current_density is None
    finally:
        env.close()


def test_default_pool_is_all_three() -> None:
    env = MixedDensityEnv()
    try:
        assert env.density_pool == ("low", "medium", "high")
    finally:
        env.close()


def test_invalid_pool_rejected() -> None:
    with pytest.raises(ValueError, match="invalid density"):
        MixedDensityEnv(density_pool=("low", "extreme"))  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="non-empty"):
        MixedDensityEnv(density_pool=())


def test_observation_and_action_spaces_density_invariant() -> None:
    env = MixedDensityEnv()
    try:
        assert env.observation_space.shape == (OBS_DIM,)
        assert env.action_space.n == 3
        assert isinstance(env.observation_space, gym.spaces.Box)
    finally:
        env.close()


def test_step_before_reset_raises() -> None:
    env = MixedDensityEnv()
    try:
        with pytest.raises(AssertionError):
            env.step(0)
    finally:
        env.close()


def test_action_masks_before_reset_raises() -> None:
    env = MixedDensityEnv()
    try:
        with pytest.raises(AssertionError):
            env.action_masks()
    finally:
        env.close()


def test_close_idempotent_before_reset() -> None:
    env = MixedDensityEnv()
    env.close()
    env.close()  # no exception


@pytest.mark.slow
@pytest.mark.sumo
def test_reset_step_round_trip() -> None:
    """End-to-end: reset → step → obs in space → info has 'density' tag."""
    env = MixedDensityEnv(
        density_pool=("low", "medium"),
        density_seed=0,
        max_steps=50,
        warmup_steps=20,
    )
    try:
        obs, _info = env.reset(seed=42)
        assert env.observation_space.contains(obs)
        assert env.current_density in ("low", "medium")

        obs2, reward, _terminated, _truncated, info = env.step(0)
        assert env.observation_space.contains(obs2)
        assert info["density"] in ("low", "medium")
        assert info["density"] == env.current_density
        assert np.isfinite(reward)
    finally:
        env.close()


@pytest.mark.slow
@pytest.mark.sumo
def test_multiple_resets_sample_different_densities() -> None:
    """Across many resets, all densities in the pool should be sampled."""
    env = MixedDensityEnv(
        density_pool=("low", "medium", "high"),
        density_seed=7,
        max_steps=10,
        warmup_steps=10,
    )
    try:
        seen = set()
        for _ in range(15):
            env.reset()
            seen.add(env.current_density)
        # 15 samples x 3 densities with uniform prob — extremely likely to
        # see all three (~3 x 0.999... probability).
        assert seen == {"low", "medium", "high"}, f"only saw {seen}"
    finally:
        env.close()
