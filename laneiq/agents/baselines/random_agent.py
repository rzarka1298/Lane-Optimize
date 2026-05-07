"""Uniform-random policy.

The PRD's lowest-bar baseline. If your trained DQN doesn't beat this on
mean-speed AND collision rate, something is broken end-to-end. Used as the
floor in the Week-3+ eval matrix.
"""

from __future__ import annotations

import numpy as np

from laneiq.env.actions import N_ACTIONS, valid_actions_mask


class RandomAgent:
    """Picks uniformly at random over the valid actions.

    Valid actions are recovered from `action_mask` if provided; otherwise
    derived from the observation via `valid_actions_mask`.
    """

    name: str = "random"

    def __init__(self, *, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)

    def reset(self, *, seed: int | None = None) -> None:
        """Optionally re-seed the internal RNG."""
        if seed is not None:
            self._rng = np.random.default_rng(seed)

    def act(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | None = None,
    ) -> int:
        if action_mask is None:
            action_mask = valid_actions_mask(observation)

        valid_indices = np.flatnonzero(action_mask)
        if valid_indices.size == 0:
            # Pathological: no valid actions. Fall back to STAY.
            # Should never happen since the obs builder guarantees STAY is valid.
            return 0
        if valid_indices.size == N_ACTIONS:
            return int(self._rng.integers(0, N_ACTIONS))
        return int(self._rng.choice(valid_indices))
