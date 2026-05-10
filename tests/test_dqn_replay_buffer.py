"""Tests for the replay buffer."""

from __future__ import annotations

import numpy as np
import pytest

from laneiq.agents.dqn.replay_buffer import ReplayBatch, ReplayBuffer


def _push_n(buf: ReplayBuffer, n: int, *, obs_dim: int = 4) -> None:
    for i in range(n):
        obs = np.full(obs_dim, i, dtype=np.float32)
        nxt = np.full(obs_dim, i + 0.5, dtype=np.float32)
        buf.push(obs, action=i % 2, reward=float(i), next_obs=nxt, done=(i % 5 == 0))


def test_initial_state() -> None:
    buf = ReplayBuffer(capacity=10, obs_dim=4)
    assert len(buf) == 0
    assert not buf.is_full
    assert buf.capacity == 10


def test_push_grows_size_until_capacity() -> None:
    buf = ReplayBuffer(capacity=5, obs_dim=4)
    _push_n(buf, 3)
    assert len(buf) == 3
    _push_n(buf, 2)
    assert len(buf) == 5
    assert buf.is_full
    # Pushing past capacity overwrites; size stays at capacity.
    _push_n(buf, 10)
    assert len(buf) == 5


def test_push_validates_obs_shape() -> None:
    buf = ReplayBuffer(capacity=3, obs_dim=4)
    with pytest.raises(ValueError, match="obs shape"):
        buf.push(np.zeros(5, dtype=np.float32), 0, 1.0, np.zeros(4, dtype=np.float32), False)
    with pytest.raises(ValueError, match="next_obs shape"):
        buf.push(np.zeros(4, dtype=np.float32), 0, 1.0, np.zeros(5, dtype=np.float32), False)


def test_sample_shapes_and_dtypes() -> None:
    buf = ReplayBuffer(capacity=100, obs_dim=4, seed=0)
    _push_n(buf, 50)
    batch = buf.sample(16)
    assert isinstance(batch, ReplayBatch)
    assert batch.obs.shape == (16, 4)
    assert batch.next_obs.shape == (16, 4)
    assert batch.actions.shape == (16,)
    assert batch.rewards.shape == (16,)
    assert batch.dones.shape == (16,)
    assert batch.obs.dtype == np.float32
    assert batch.next_obs.dtype == np.float32
    assert batch.actions.dtype == np.int64
    assert batch.rewards.dtype == np.float32
    assert batch.dones.dtype == bool


def test_sample_from_empty_raises() -> None:
    buf = ReplayBuffer(capacity=10, obs_dim=4)
    with pytest.raises(RuntimeError, match="empty"):
        buf.sample(4)


def test_sample_validates_batch_size() -> None:
    buf = ReplayBuffer(capacity=10, obs_dim=4, seed=0)
    _push_n(buf, 5)
    with pytest.raises(ValueError, match="batch_size"):
        buf.sample(0)


def test_seeding_reproducible_sampling() -> None:
    buf1 = ReplayBuffer(capacity=100, obs_dim=4, seed=42)
    buf2 = ReplayBuffer(capacity=100, obs_dim=4, seed=42)
    _push_n(buf1, 100)
    _push_n(buf2, 100)
    s1 = buf1.sample(16)
    s2 = buf2.sample(16)
    np.testing.assert_array_equal(s1.actions, s2.actions)
    np.testing.assert_array_equal(s1.obs, s2.obs)


def test_overwrite_keeps_only_last_capacity_transitions() -> None:
    """After pushing 2*capacity items, only the latest `capacity` should remain."""
    buf = ReplayBuffer(capacity=5, obs_dim=4, seed=0)
    _push_n(buf, 12)
    # Items pushed are i=0..11. With capacity 5, the buffer holds the
    # transitions whose write-index is the most recent 5 — not necessarily
    # i=7..11 because of circular indexing, but the *set* of stored rewards
    # should equal {7, 8, 9, 10, 11}.
    rewards_present = set()
    for _ in range(50):
        b = buf.sample(1)
        rewards_present.add(int(b.rewards[0]))
    assert rewards_present == {7, 8, 9, 10, 11}, f"got {rewards_present}"


def test_invalid_capacity_rejected() -> None:
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=0, obs_dim=4)
    with pytest.raises(ValueError, match="capacity"):
        ReplayBuffer(capacity=-1, obs_dim=4)
