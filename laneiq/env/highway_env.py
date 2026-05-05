"""Single-agent Gymnasium environment for LaneIQ.

`LaneIQEnv` stitches together the four env building blocks
(`sumo_runtime`, `observations`, `actions`, `rewards`) into a
`gymnasium.Env` the RL trainer can drive directly.

Lifecycle:

- `__init__` validates config but does NOT start SUMO.
- `reset()` closes any prior TraCI connection, starts a fresh SUMO process
  with the seeded argv, warms up background traffic, adds the ego, and
  returns the first observation.
- `step()` applies the action, advances the simulation one tick, computes
  the reward, and returns the 5-tuple Gymnasium expects.
- `close()` shuts down SUMO. Idempotent.

Single-instance assumption: this class uses TraCI's module-level connection
state, so you cannot run two `LaneIQEnv` instances in the same Python
process. For parallel rollouts, use `SubprocVecEnv` from SB3 (each subprocess
gets its own TraCI). See `Project-Documentation/laneiq/env/highway_env.md`
for the full design contract.
"""

from __future__ import annotations

import logging
import math
from contextlib import suppress
from dataclasses import replace
from typing import Any, ClassVar, Literal

import gymnasium as gym
import numpy as np
import traci

from laneiq.env.actions import (
    ACTION_SPACE,
    apply_action,
    disable_sumo_lane_change,
    valid_actions_mask,
)
from laneiq.env.observations import (
    DEFAULT_OBSERVATION_CONFIG,
    OBSERVATION_SPEC,
    ObservationBuilder,
    ObservationConfig,
)
from laneiq.env.rewards import (
    DEFAULT_REWARD_CONFIG,
    RewardConfig,
    RewardState,
    compute_reward,
)
from laneiq.env.sumo_runtime import (
    DEFAULT_SCENARIO_DIR,
    DEFAULT_SUMOCFG,
    build_argv,
)

_log = logging.getLogger(__name__)

# Map density label → route file relative to DEFAULT_SCENARIO_DIR.
_DENSITY_TO_ROUTES: dict[str, str] = {
    "low": "routes_low.rou.xml",
    "medium": "routes_medium.rou.xml",
    "high": "routes_high.rou.xml",
}

# Hardcoded names for the 3-lane scenario. If we ever ship multi-edge
# networks, lift these into the constructor.
_EGO_ID: str = "ego_0"
_ROUTE_ID: str = "r_through"
_EDGE_ID: str = "e1"
_EDGE_LENGTH_M: float = 1000.0

# TraCI connection label — distinct from `sumo_session`'s "laneiq" so the
# two can't collide in a process that uses both (e.g., a script running
# tests + an env back-to-back).
_TRACI_LABEL: str = "laneiq_env"

DensityT = Literal["low", "medium", "high"]


class LaneIQEnv(gym.Env):
    """Single-agent Gymnasium env wrapping SUMO via TraCI.

    Observation space: `gym.spaces.Box(-1, 1, (25,), float32)` — see
        `laneiq.env.observations.OBSERVATION_SPEC`.
    Action space: `gym.spaces.Discrete(3)` — `Action.STAY / CHANGE_LEFT
        / CHANGE_RIGHT`.

    Episode termination:
        - `terminated=True` if the ego collides or reaches the end of the road.
        - `truncated=True` if `step_count >= max_steps`.
        - `info["reward_components"]` carries the per-term reward breakdown
          for TensorBoard logging.

    Determinism:
        Pass a `seed` to `__init__` (default seed) or to `reset(seed=...)`
        (per-episode override). The seed flows to SUMO's `--seed` flag,
        making background traffic and Krauss `sigma` reproducible.
    """

    metadata: ClassVar[dict[str, Any]] = {"render_modes": [], "render_fps": 10}

    def __init__(
        self,
        scenario_path: str | bytes = DEFAULT_SUMOCFG,
        density: DensityT = "medium",
        *,
        max_steps: int = 1000,
        warmup_steps: int = 50,
        gui: bool = False,
        ego_depart_lane: int = 1,
        ego_depart_speed: float = 20.0,
        seed: int | None = None,
        obs_config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
    ) -> None:
        """Construct the env. Does NOT start SUMO; that happens in `reset()`."""
        super().__init__()

        if density not in _DENSITY_TO_ROUTES:
            raise ValueError(
                f"density must be one of {sorted(_DENSITY_TO_ROUTES)}, got {density!r}"
            )

        self._scenario_path = scenario_path
        self._density: DensityT = density
        self._max_steps = max_steps
        self._warmup_steps = warmup_steps
        self._gui = gui
        self._ego_depart_lane = ego_depart_lane
        self._ego_depart_speed = ego_depart_speed
        self._init_seed = seed
        self._obs_config = obs_config
        self._reward_config = reward_config

        self.observation_space: gym.spaces.Box = OBSERVATION_SPEC.space
        self.action_space: gym.spaces.Discrete = ACTION_SPACE

        # Per-episode state (None until first reset())
        self._builder: ObservationBuilder | None = None
        self._step_count: int = 0
        self._last_obs: np.ndarray | None = None
        self._prev_reward_state: RewardState | None = None
        self._sumo_running: bool = False

    # ------------------------------------------------------------------
    # Gymnasium contract
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Restart SUMO, warm up traffic, spawn the ego, return first obs."""
        super().reset(seed=seed)

        # Per-episode seed override > constructor seed. None → SUMO's default.
        resolved_seed = seed if seed is not None else self._init_seed

        # Tear down any prior connection. Idempotent.
        self._close_sumo_safely()

        # Compose argv for this density. The route_file CLI flag overrides
        # whatever the .sumocfg names by default.
        route_file = DEFAULT_SCENARIO_DIR / _DENSITY_TO_ROUTES[self._density]
        argv = build_argv(
            self._scenario_path,
            gui=self._gui,
            seed=resolved_seed,
            route_file=route_file,
        )

        traci.start(argv, label=_TRACI_LABEL)
        self._sumo_running = True

        # Warmup: let background traffic populate before the ego enters,
        # so the first observation reflects realistic neighbors.
        for _ in range(self._warmup_steps):
            traci.simulationStep()

        # Spawn the ego. `depart="now"` schedules immediate insertion;
        # the next simulationStep() makes it appear in getIDList.
        traci.vehicle.add(
            vehID=_EGO_ID,
            routeID=_ROUTE_ID,
            typeID="ego",
            depart="now",
            departLane=str(self._ego_depart_lane),
            departSpeed=str(self._ego_depart_speed),
            departPos="0.0",
        )
        traci.simulationStep()

        # The ego is now in the vehicle list; disable its internal LC
        # override so the policy is the sole decider.
        disable_sumo_lane_change(traci, _EGO_ID)

        # Initialize the observation builder (caches lane speed limit).
        self._builder = ObservationBuilder(traci, config=self._obs_config)
        self._builder.on_episode_start(_EGO_ID)

        # Build the initial observation; cache the reward-state slice for
        # the first step's `prev` (so jerk is computed from a real number,
        # not a placeholder).
        obs = self._builder.build(_EGO_ID)
        self._last_obs = obs
        self._prev_reward_state = self._extract_reward_state()
        self._step_count = 0

        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Apply action → step SUMO → compute reward → return 5-tuple."""
        if not self._sumo_running:
            raise RuntimeError("step() called before reset() — start a new episode first")

        assert self._prev_reward_state is not None
        assert self._builder is not None
        prev = self._prev_reward_state

        # Apply the action (silent no-op if invalid; mask should prevent).
        try:
            apply_action(traci, _EGO_ID, action)
        except traci.exceptions.TraCIException as exc:
            # Race window: ego could have been removed between getIDList
            # check and changeLane call. Log and proceed; the post-step
            # check below will catch the ego-disappeared case.
            _log.warning("apply_action raised; continuing: %s", exc)

        # Advance the simulation by one tick.
        traci.simulationStep()
        self._step_count += 1

        # Detect terminal conditions.
        try:
            collision_ids = traci.simulation.getCollidingVehiclesIDList()
            ego_collided = _EGO_ID in collision_ids
            ego_in_list = _EGO_ID in traci.vehicle.getIDList()
        except traci.exceptions.TraCIException:
            # SUMO blew up on this step; treat as terminal-no-collision.
            ego_collided = False
            ego_in_list = False

        if ego_collided or not ego_in_list:
            # Episode is ending. Build a terminal RewardState; if the ego
            # is still in the list use real values, otherwise reuse prev
            # with the appropriate flags flipped.
            if ego_in_list:
                nxt = self._extract_reward_state(collided=ego_collided)
            else:
                nxt = replace(prev, collided=ego_collided)

            reward, components = compute_reward(
                prev, action, nxt, config=self._reward_config
            )

            assert self._last_obs is not None  # set by reset()
            info = {
                "reward_components": components,
                "ego_terminated_reason": "collision" if ego_collided else "exited",
                "step_count": self._step_count,
            }
            return self._last_obs, float(reward), True, False, info

        # Normal continuation. Build next obs + RewardState; compute reward.
        nxt = self._extract_reward_state()
        reward, components = compute_reward(
            prev, action, nxt, config=self._reward_config
        )

        obs = self._builder.build(_EGO_ID)
        self._last_obs = obs
        self._prev_reward_state = nxt

        truncated = self._step_count >= self._max_steps
        info: dict[str, Any] = {
            "reward_components": components,
            "step_count": self._step_count,
        }
        return obs, float(reward), False, truncated, info

    def close(self) -> None:
        """Shut down SUMO. Idempotent — safe to call multiple times or before
        any reset()."""
        self._close_sumo_safely()

    def render(self) -> None:
        """No-op. Real visualization is the FastAPI/React dashboard
        (Week 5+); the GUI mode for debugging goes through `make sumo-gui-*`."""
        return None

    # ------------------------------------------------------------------
    # MaskablePPO support — exposed so `sb3-contrib` can fetch the mask.
    # ------------------------------------------------------------------

    def action_masks(self) -> np.ndarray:
        """Return the valid-action mask for the current state.

        Returns:
            shape-(3,) bool array — True where the action is valid.

        Raises:
            RuntimeError: if called before any `reset()`.
        """
        if self._last_obs is None:
            raise RuntimeError("action_masks() called before any reset()")
        return valid_actions_mask(self._last_obs)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _close_sumo_safely(self) -> None:
        """Tear down the TraCI connection; never raise."""
        if self._builder is not None:
            with suppress(Exception):
                self._builder.on_episode_end()
            self._builder = None
        if self._sumo_running:
            with suppress(Exception):
                traci.close(False)
            self._sumo_running = False
        self._last_obs = None
        self._prev_reward_state = None
        self._step_count = 0

    def _extract_reward_state(self, collided: bool = False) -> RewardState:
        """Pull the raw-units RewardState slice from the current TraCI snapshot.

        Caller must guarantee the ego is in `getIDList`; otherwise TraCI
        will raise.
        """
        veh = traci.vehicle
        speed = float(veh.getSpeed(_EGO_ID))
        accel = float(veh.getAcceleration(_EGO_ID))
        lane_idx = int(veh.getLaneIndex(_EGO_ID))

        leader = veh.getLeader(_EGO_ID, self._obs_config.max_gap_m)
        if leader is not None and leader[0] != "":
            front_gap = float(leader[1])
            front_thw = front_gap / max(speed, self._obs_config.speed_eps)
        else:
            front_gap = math.inf
            front_thw = math.inf

        return RewardState(
            speed=speed,
            accel=accel,
            front_gap=front_gap,
            front_thw=front_thw,
            collided=collided,
            lane_idx=lane_idx,
        )
