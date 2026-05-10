"""Uniform experience-replay buffer for DQN.

Stores the last `capacity` transitions in pre-allocated numpy arrays.
Sampling is uniform random with replacement (the canonical DQN choice;
prioritized replay is a known extension we deliberately defer).

Why pre-allocated arrays vs. a `deque` of dicts:

- `deque` of dicts forces a Python dict allocation per push and a numpy
  conversion per sample → thrashes the allocator and pegs the GIL.
- Pre-allocated arrays + circular-write index = O(1) push, O(batch) sample,
  zero per-step allocation.

The buffer is **observation-shape agnostic**: it stores whatever shape
you push. Used unchanged for both CartPole (obs_dim=4) and LaneIQEnv
(obs_dim=25).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ReplayBatch:
    """A sampled minibatch. All arrays have the same first dim = batch size."""

    obs: np.ndarray            # shape (batch, obs_dim) float32
    actions: np.ndarray         # shape (batch,) int64
    rewards: np.ndarray         # shape (batch,) float32
    next_obs: np.ndarray        # shape (batch, obs_dim) float32
    dones: np.ndarray           # shape (batch,) bool


class ReplayBuffer:
    """FIFO replay buffer with uniform sampling.

    Args:
        capacity: maximum number of transitions retained. New pushes
            overwrite the oldest when full.
        obs_dim: dimensionality of the observation vector.
        seed: optional seed for the sampling RNG.
    """

    def __init__(
        self,
        capacity: int,
        obs_dim: int,
        *,
        seed: int | None = None,
    ) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        self._capacity = capacity
        self._obs_dim = obs_dim
        self._rng = np.random.default_rng(seed)

        # Pre-allocated storage. dtypes chosen to match what the trainer
        # converts to torch tensors (no implicit casts at sample time).
        self._obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self._actions = np.zeros(capacity, dtype=np.int64)
        self._rewards = np.zeros(capacity, dtype=np.float32)
        self._dones = np.zeros(capacity, dtype=bool)

        self._idx = 0           # next write position
        self._size = 0          # number of valid entries (≤ capacity)

    # ------------------------------------------------------------------
    # Read-only properties
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return self._size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def is_full(self) -> bool:
        return self._size >= self._capacity

    # ------------------------------------------------------------------
    # Mutators
    # ------------------------------------------------------------------

    def push(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Insert one transition. Overwrites the oldest if full."""
        if obs.shape != (self._obs_dim,):
            raise ValueError(
                f"obs shape {obs.shape} != expected ({self._obs_dim},)"
            )
        if next_obs.shape != (self._obs_dim,):
            raise ValueError(
                f"next_obs shape {next_obs.shape} != expected ({self._obs_dim},)"
            )

        self._obs[self._idx] = obs
        self._next_obs[self._idx] = next_obs
        self._actions[self._idx] = action
        self._rewards[self._idx] = reward
        self._dones[self._idx] = done

        self._idx = (self._idx + 1) % self._capacity
        if self._size < self._capacity:
            self._size += 1

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, batch_size: int) -> ReplayBatch:
        """Sample `batch_size` transitions uniformly (with replacement)."""
        if self._size == 0:
            raise RuntimeError("cannot sample from an empty buffer")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be > 0, got {batch_size}")

        idx = self._rng.integers(0, self._size, size=batch_size)
        return ReplayBatch(
            obs=self._obs[idx],
            actions=self._actions[idx],
            rewards=self._rewards[idx],
            next_obs=self._next_obs[idx],
            dones=self._dones[idx],
        )
