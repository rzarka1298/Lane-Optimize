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
    COMPONENT_KEYS,
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
# "gauntlet" and "rush" are not densities per se — they're special
# scenarios. gauntlet forces overtaking (heavy slow traffic in lane 0,
# ego spawned in lane 0). rush is a high-speed scenario: the road
# speed limit is raised so the ego (vType maxSpeed 50) can genuinely
# outrun every background vehicle. See __init__ and reset().
_DENSITY_TO_ROUTES: dict[str, str] = {
    "low": "routes_low.rou.xml",
    "medium": "routes_medium.rou.xml",
    "high": "routes_high.rou.xml",
    "gauntlet": "routes_gauntlet.rou.xml",
    "rush": "routes_rush.rou.xml",
}

# The rush scenario raises the lane speed limit to this value (m/s) at
# reset() via TraCI. The ego vType maxSpeed (50) then becomes reachable
# while background vTypes stay capped at their own maxSpeeds
# (aggressive 40, normal 33.33, slow 25) — so the ego is uniquely the
# fastest vehicle on the road.
_RUSH_SPEED_LIMIT: float = 50.0

# Rush scenario uses dynamic, ego-relative traffic spawning instead of
# fixed-position Poisson flows. Vehicles appear just past the visible
# camera window ahead of the ego and despawn once they've fallen far
# enough behind. Tuned so a viewer never sees a car materialize or
# vanish on-screen.
_RUSH_SPAWN_AHEAD_M: float = 300.0       # spawn distance forward of ego
_RUSH_DESPAWN_BEHIND_M: float = 250.0    # despawn threshold behind ego
# Sparse spawning so the ego has clear road most of the time and only
# slows when a leader is genuinely close. (15-30 ticks was too dense —
# ego averaged 35 m/s, often caught in traffic. 40-80 keeps 4-7 visible.)
_RUSH_SPAWN_TICK_RANGE: tuple[int, int] = (40, 80)  # 4-8s between spawns
_RUSH_INIT_TRAFFIC: int = 5              # vehicles to pre-seed at reset
# 70/30 normal/slow — fewer slow vehicles means fewer hard speed walls.
_RUSH_SLOW_FRACTION: float = 0.30

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

DensityT = Literal["low", "medium", "high", "gauntlet", "rush"]


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
        loop: bool = False,
    ) -> None:
        """Construct the env. Does NOT start SUMO; that happens in `reset()`."""
        super().__init__()

        if density not in _DENSITY_TO_ROUTES:
            raise ValueError(
                f"density must be one of {sorted(_DENSITY_TO_ROUTES)}, got {density!r}"
            )

        self._scenario_path = scenario_path
        self._density: DensityT = density

        # The gauntlet scenario only works if the ego starts in lane 0
        # (behind the slow-vehicle wall). Force it regardless of the
        # caller's ego_depart_lane — a gauntlet episode that spawns the
        # ego in lane 2 defeats the entire purpose.
        if density == "gauntlet":
            ego_depart_lane = 0

        # The rush scenario raises the lane speed limit (see reset()).
        # The reward's v_max — the speed at which the speed term
        # saturates — must match, or the policy gets no incentive to
        # exceed the old 36.11 limit. v_max is a scenario constant, not
        # a tuning knob, so override it regardless of what was passed.
        if density == "rush":
            reward_config = replace(reward_config, v_max=_RUSH_SPEED_LIMIT)
        self._max_steps = max_steps
        self._warmup_steps = warmup_steps
        self._gui = gui
        self._ego_depart_lane = ego_depart_lane
        self._ego_depart_speed = ego_depart_speed
        self._init_seed = seed
        self._obs_config = obs_config
        self._reward_config = reward_config
        self._loop = loop
        # Lap count is incremented every time the ego completes the 1km
        # road and is re-inserted at x=0 (loop mode only). Episode runs
        # until collision or max_steps.
        self._lap_count: int = 0
        # Rush-mode dynamic traffic state (only used when density=="rush").
        self._spawn_cooldown: int = 0
        self._spawn_count: int = 0

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
        self._lap_count = 0

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

        # Rush scenario: raise the lane speed limit so the ego (vType
        # maxSpeed 50) can outrun all background traffic. The observation
        # builder reads the lane limit live (_lookup_v_max_lane), so the
        # speed normalization adapts automatically — no obs_config change.
        if self._density == "rush":
            traci.edge.setMaxSpeed(_EDGE_ID, _RUSH_SPEED_LIMIT)

        # Warmup: let background traffic populate before the ego enters,
        # so the first observation reflects realistic neighbors.
        for _ in range(self._warmup_steps):
            traci.simulationStep()

        # Spawn the ego. `depart="now"` schedules immediate insertion,
        # but at high density the requested (lane, speed) slot may be
        # occupied — SUMO defers insertion silently in that case. Loop a
        # few sim steps until the ego actually appears in getIDList.
        traci.vehicle.add(
            vehID=_EGO_ID,
            routeID=_ROUTE_ID,
            typeID="ego",
            depart="now",
            departLane=str(self._ego_depart_lane),
            departSpeed=str(self._ego_depart_speed),
            departPos="0.0",
        )
        max_insertion_wait = 50  # 5 sim seconds at 0.1s step
        inserted = False
        for _ in range(max_insertion_wait):
            traci.simulationStep()
            if _EGO_ID in traci.vehicle.getIDList():
                inserted = True
                break
        if not inserted:
            raise RuntimeError(
                f"ego {_EGO_ID!r} failed to insert within "
                f"{max_insertion_wait} sim steps at density={self._density!r}",
            )

        # The ego is now in the vehicle list; disable its internal LC
        # override so the policy is the sole decider.
        disable_sumo_lane_change(traci, _EGO_ID)

        # Rush-mode ego tuning + dynamic traffic setup (see module
        # constants for the rationale). All scoped to this episode's
        # SUMO process — no global vTypes changes.
        if self._density == "rush":
            # Krauss params: tight following + snappy accel so the ego
            # holds high speed except when a leader is genuinely close.
            traci.vehicle.setTau(_EGO_ID, 0.4)
            traci.vehicle.setMinGap(_EGO_ID, 1.0)
            traci.vehicle.setAccel(_EGO_ID, 4.0)
            # Re-enable SUMO's speed-gain lane-change motivation only
            # (mode = TraCI override "do if safe" | speed-gain "do if
            # safe"). The RL policy can still issue changeLane(); SUMO
            # will additionally pick faster lanes autonomously. The
            # disable_sumo_lane_change() call above set mode=0; we
            # override here.
            traci.vehicle.setLaneChangeMode(_EGO_ID, 1040)
            self._spawn_cooldown = 0
            self._spawn_count = 0
            self._seed_rush_traffic()

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

        # Loop mode: if the ego exited cleanly (off the end of the road,
        # no collision), re-insert it at x=0 and continue instead of
        # terminating. The episode keeps running until collision or
        # max_steps. The transition step returns zero reward (it's a
        # teleport, not a real physical step).
        if (
            not ego_collided and not ego_in_list and self._loop
            and self._reinsert_ego_for_lap()
        ):
            nxt = self._extract_reward_state()
            obs = self._builder.build(_EGO_ID)
            self._last_obs = obs
            self._prev_reward_state = nxt
            truncated = self._step_count >= self._max_steps
            info = {
                "reward_components": {k: 0.0 for k in COMPONENT_KEYS},
                "step_count": self._step_count,
                "lap_count": self._lap_count,
                "lap_transition": True,
            }
            return obs, 0.0, False, truncated, info
        # If loop was active but re-insertion failed (depart slot
        # occupied at high traffic), fall through to the normal terminal
        # path below.

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

        # Rush mode: dynamic traffic. Runs only in the normal continuation
        # path (lap-transition steps already advanced SUMO internally).
        if self._density == "rush":
            self._maybe_spawn_ahead()
            self._despawn_behind()

        truncated = self._step_count >= self._max_steps
        info: dict[str, Any] = {
            "reward_components": components,
            "step_count": self._step_count,
            "lap_count": self._lap_count,
        }
        return obs, float(reward), False, truncated, info

    # ------------------------------------------------------------------
    # Rush-mode dynamic traffic (see _RUSH_* constants).
    # ------------------------------------------------------------------

    def _seed_rush_traffic(self) -> None:
        """Pre-populate vehicles ahead of the ego at reset so the road
        isn't empty when the policy starts driving."""
        try:
            ego_x = traci.vehicle.getLanePosition(_EGO_ID)
        except traci.exceptions.TraCIException:
            return
        for i in range(_RUSH_INIT_TRAFFIC):
            # Distribute the initial fleet between ~30 m and ~400 m
            # ahead, evenly-ish so they're not all stacked.
            offset = 30 + (i + self.np_random.random()) * (370 / _RUSH_INIT_TRAFFIC)
            self._spawn_rush_vehicle(ego_x + offset)

    def _spawn_rush_vehicle(self, spawn_x: float) -> None:
        """Add a single rush-mode background vehicle at `spawn_x` (wrap
        to road length). Silent on insertion failure (slot occupied)."""
        # Wrap to the road and clip safely inside (depart fails at the
        # very edge).
        spawn_x = spawn_x % _EDGE_LENGTH_M
        spawn_x = max(5.0, min(_EDGE_LENGTH_M - 5.0, spawn_x))
        # Pick vType per _RUSH_SLOW_FRACTION (slow vehicles are the
        # visible "wall to overtake," normal are the in-between traffic).
        vtype = "slow" if self.np_random.random() < _RUSH_SLOW_FRACTION else "normal"
        lane = int(self.np_random.integers(0, 3))
        vid = f"rush_{self._spawn_count}"
        self._spawn_count += 1
        with suppress(traci.exceptions.TraCIException):
            traci.vehicle.add(
                vehID=vid, routeID=_ROUTE_ID, typeID=vtype,
                depart="now", departLane=str(lane),
                departSpeed="desired", departPos=f"{spawn_x:.2f}",
            )

    def _maybe_spawn_ahead(self) -> None:
        """Per-step: cooldown-throttled spawn ahead of the ego."""
        if self._spawn_cooldown > 0:
            self._spawn_cooldown -= 1
            return
        self._spawn_cooldown = int(
            self.np_random.integers(*_RUSH_SPAWN_TICK_RANGE)
        )
        try:
            ego_x = traci.vehicle.getLanePosition(_EGO_ID)
        except traci.exceptions.TraCIException:
            return
        self._spawn_rush_vehicle(ego_x + _RUSH_SPAWN_AHEAD_M)

    def _despawn_behind(self) -> None:
        """Per-step: remove background vehicles that have fallen far
        enough behind the ego (wrap-aware). The ego itself is never
        despawned this way."""
        try:
            ego_x = traci.vehicle.getLanePosition(_EGO_ID)
            ids = traci.vehicle.getIDList()
        except traci.exceptions.TraCIException:
            return
        for vid in ids:
            if vid == _EGO_ID:
                continue
            try:
                vx = traci.vehicle.getLanePosition(vid)
            except traci.exceptions.TraCIException:
                continue
            # Signed wrap-aware offset; negative = behind the ego.
            rel = vx - ego_x
            if rel > _EDGE_LENGTH_M / 2:
                rel -= _EDGE_LENGTH_M
            elif rel < -_EDGE_LENGTH_M / 2:
                rel += _EDGE_LENGTH_M
            if rel < -_RUSH_DESPAWN_BEHIND_M:
                with suppress(traci.exceptions.TraCIException):
                    traci.vehicle.remove(vid)

    def _reinsert_ego_for_lap(self) -> bool:
        """Re-add the ego at x=0 to start a new lap. Returns True iff the
        ego is visible in `getIDList` afterwards. Caller checks the bool
        and falls back to terminal handling on False (the only failure
        mode in practice is the depart slot being occupied at high traffic).

        Sim steps consumed by the insertion wait are NOT counted in
        `self._step_count` — they're internal env machinery, not policy
        decisions.
        """
        try:
            traci.vehicle.add(
                vehID=_EGO_ID, routeID=_ROUTE_ID, typeID="ego",
                depart="now",
                departLane=str(self._ego_depart_lane),
                departSpeed=str(self._ego_depart_speed),
                departPos="0.0",
            )
        except traci.exceptions.TraCIException:
            return False
        for _ in range(50):
            traci.simulationStep()
            if _EGO_ID in traci.vehicle.getIDList():
                disable_sumo_lane_change(traci, _EGO_ID)
                self._lap_count += 1
                return True
        return False

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
