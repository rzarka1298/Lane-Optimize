"""Density-sampling wrapper around `LaneIQEnv` for robust training.

A DQN trained on a single density (say medium) often handles other
densities poorly — observed in Week 3's Task #17 followup: 0% collisions
at low, 0% at medium, but **80% collisions at high**. The policy never
saw the kind of dense traffic where the safe move is different.

`MixedDensityEnv` solves this without touching `LaneIQEnv`: it samples a
density uniformly from a configurable pool on every `reset()`. The
sampled density requires a fresh SUMO process (different route file),
which costs ~150ms of overhead per episode — negligible vs. the 27s of
sim time inside each episode.

The wrapper preserves the Gymnasium contract exactly:

- `observation_space` and `action_space` are density-invariant.
- `action_masks()` delegates to the active underlying env.
- `reset(seed=...)` flows seed to BOTH the density RNG and the inner env.
- `step()` delegates straight through.

The training loop is unchanged; only the env construction differs.
"""

from __future__ import annotations

from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np

from laneiq.env.highway_env import DensityT, LaneIQEnv
from laneiq.env.observations import OBSERVATION_SPEC

DensityPool = tuple[Literal["low", "medium", "high"], ...]
_DEFAULT_POOL: DensityPool = ("low", "medium", "high")


class MixedDensityEnv(gym.Env):
    """`LaneIQEnv` that samples a density per episode.

    Args mirror `LaneIQEnv` except `density` is replaced by `density_pool`
    (a sequence to sample uniformly from) and `density_seed` (the RNG
    seed for the sampler; can be re-seeded on `reset(seed=...)`).
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": [], "render_fps": 10}

    def __init__(
        self,
        *,
        density_pool: DensityPool = _DEFAULT_POOL,
        density_seed: int | None = None,
        **lane_iq_kwargs: Any,
    ) -> None:
        super().__init__()
        if not density_pool:
            raise ValueError("density_pool must be non-empty")
        for d in density_pool:
            if d not in ("low", "medium", "high"):
                raise ValueError(f"invalid density in pool: {d!r}")

        self._density_pool: tuple[str, ...] = tuple(density_pool)
        self._density_rng = np.random.default_rng(density_seed)
        self._inner_kwargs = lane_iq_kwargs
        # The active inner env. Built on first reset(); rebuilt when the
        # sampled density changes.
        self._inner: LaneIQEnv | None = None
        self._current_density: DensityT | None = None

        # These ARE density-invariant by construction, so we can expose them
        # without an inner env.
        self.observation_space: gym.spaces.Box = OBSERVATION_SPEC.space
        from laneiq.env.actions import ACTION_SPACE

        self.action_space: gym.spaces.Discrete = ACTION_SPACE

    # ------------------------------------------------------------------
    # Gymnasium contract
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        # Optionally re-seed the density sampler when the caller seeds.
        if seed is not None:
            self._density_rng = np.random.default_rng(seed)

        # Sample a density. Rebuild the inner env iff it changed (or first call).
        sampled: DensityT = self._density_rng.choice(self._density_pool)  # type: ignore[assignment]
        if self._inner is None or self._current_density != sampled:
            if self._inner is not None:
                self._inner.close()
            self._inner = LaneIQEnv(
                density=sampled,
                **self._inner_kwargs,
            )
            self._current_density = sampled

        return self._inner.reset(seed=seed, options=options)

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self._inner is not None, "step() before reset()"
        obs, reward, terminated, truncated, info = self._inner.step(action)
        info = {**info, "density": self._current_density}  # tag for downstream logging
        return obs, reward, terminated, truncated, info

    def close(self) -> None:
        if self._inner is not None:
            self._inner.close()
            self._inner = None
            self._current_density = None

    def render(self) -> None:
        return None

    # ------------------------------------------------------------------
    # MaskablePPO support — forwarded
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        assert self._inner is not None, "action_masks() before reset()"
        return self._inner.action_masks()

    # ------------------------------------------------------------------
    # Introspection (useful for logging / debugging)
    # ------------------------------------------------------------------

    @property
    def current_density(self) -> DensityT | None:
        return self._current_density

    @property
    def density_pool(self) -> tuple[str, ...]:
        return self._density_pool
