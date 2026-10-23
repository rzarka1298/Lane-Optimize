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
    DEFAULT_LANE_CHANGE_COOLDOWN_STEPS,
    LaneChangeCooldown,
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

# Rush scenario runs on a dedicated 10 km straight road (no looping).
# The ego drives ~3.5 min at ~45 m/s without reaching the end, so there
# is no seam to wrap and no glitches. Background traffic is spawned
# dynamically just ahead of the ego (off the visible camera window) and
# despawned once it has fallen far enough behind — so cars appear "from
# the horizon" and recede naturally, never popping in/out on screen.
_RUSH_EDGE_LENGTH_M: float = 10000.0      # rush road length (vs 1 km default)
_RUSH_TARGET_TRAFFIC: int = 4             # cars kept in the band around the ego
_RUSH_SPAWN_AHEAD_M: float = 320.0        # spawn this far ahead (just off-screen)
_RUSH_DESPAWN_BEHIND_M: float = 320.0     # despawn once this far behind (off-screen)
# 70/30 normal/slow for random scenery cars — fewer slow vehicles means
# fewer hard speed walls.
_RUSH_SLOW_FRACTION: float = 0.30
# Fraction of refill spawns placed as a blocker in the ego's CURRENT lane
# (the rest are random scenery). This is the weave forcer: the ego keeps
# catching a car in its own lane and must dodge into a clear lane to pass.
# The blocker is a "normal" (not "slow") car so the closing speed from the
# ego's ~48 m/s is small enough that, if the ego is briefly boxed in, it can
# brake without rear-ending. Bounded by _RUSH_TARGET_TRAFFIC. 0.5 balances a
# steady supply of blockers against keeping the dodge lanes clear — higher
# values box the ego in more often and reintroduce collisions.
_RUSH_EGO_LANE_BIAS: float = 0.5

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
        lane_change_cooldown_steps: int = DEFAULT_LANE_CHANGE_COOLDOWN_STEPS,
    ) -> None:
        """Construct the env. Does NOT start SUMO; that happens in `reset()`."""
        super().__init__()

        if density not in _DENSITY_TO_ROUTES:
            raise ValueError(
                f"density must be one of {sorted(_DENSITY_TO_ROUTES)}, got {density!r}"
            )

        self._scenario_path = scenario_path
        self._density: DensityT = density
        # Road length for THIS scenario. The rush scenario uses a
        # dedicated 10 km network so the ego can drive ~3.5 min without
        # reaching the end — no looping/wrapping needed (that caused
        # seam glitches). Every other scenario uses the 1 km road.
        self._edge_length: float = _EDGE_LENGTH_M

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
        # It also swaps to the 10 km network (unless the caller passed a
        # custom scenario_path).
        if density == "rush":
            reward_config = replace(reward_config, v_max=_RUSH_SPEED_LIMIT)
            self._edge_length = _RUSH_EDGE_LENGTH_M
            if scenario_path == DEFAULT_SUMOCFG:
                self._scenario_path = str(
                    DEFAULT_SCENARIO_DIR / "highway_long.sumocfg")
        self._max_steps = max_steps
        self._warmup_steps = warmup_steps
        self._gui = gui
        self._ego_depart_lane = ego_depart_lane
        self._ego_depart_speed = ego_depart_speed
        self._init_seed = seed
        self._obs_config = obs_config
        self._reward_config = reward_config
        self._loop = loop
        # Min dwell between executed ego lane changes; also masks obs[23]
        # while active. 0 = off (pre-2026-10 dynamics). See actions.py.
        self._cooldown = LaneChangeCooldown(lane_change_cooldown_steps)
        # Lap count is incremented every time the ego completes the 1km
        # road and is re-inserted at x=0 (loop mode only). Episode runs
        # until collision or max_steps.
        self._lap_count: int = 0
        # Rush-mode circular traffic state (only used when density=="rush").
        self._spawn_count: int = 0
        self._rush_ids: set[str] = set()
        self._rush_seen: set[str] = set()

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
            self._apply_rush_ego_tuning()
            self._spawn_count = 0
            self._rush_ids = set()
            self._rush_seen = set()
            self._seed_rush_traffic()
            # vehicle.add schedules insertion for the NEXT sim step, so
            # right now the fleet is still pending (only the ego is in
            # the sim). Settle a few steps so the fleet actually inserts
            # before reset returns — otherwise the refill in
            # _manage_rush_traffic fires every tick during the ramp-up and
            # massively over-spawns.
            for _ in range(15):
                traci.simulationStep()
                bg = sum(1 for v in traci.vehicle.getIDList() if v != _EGO_ID)
                if bg >= _RUSH_TARGET_TRAFFIC:
                    break

        # Initialize the observation builder (caches lane speed limit).
        self._builder = ObservationBuilder(traci, config=self._obs_config)
        self._builder.on_episode_start(_EGO_ID)

        # Build the initial observation; cache the reward-state slice for
        # the first step's `prev` (so jerk is computed from a real number,
        # not a placeholder).
        self._cooldown.reset()
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

        # Gate through the lane-change cooldown, then apply. `executed` is
        # what SUMO actually sees; the reward still charges the *requested*
        # action so unmasked learners pay for spamming changes.
        executed = self._cooldown.gate(action, valid_actions_mask(self._last_obs))
        try:
            apply_action(traci, _EGO_ID, executed)
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
            obs = self._cooldown.mask_observation(self._builder.build(_EGO_ID))
            self._last_obs = obs
            self._prev_reward_state = nxt
            truncated = self._step_count >= self._max_steps
            info = {
                "reward_components": {k: 0.0 for k in COMPONENT_KEYS},
                "step_count": self._step_count,
                "executed_action": int(executed),
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
                "executed_action": int(executed),
            }
            return self._last_obs, float(reward), True, False, info

        # Normal continuation. Build next obs + RewardState; compute reward.
        nxt = self._extract_reward_state()
        reward, components = compute_reward(
            prev, action, nxt, config=self._reward_config
        )

        obs = self._cooldown.mask_observation(self._builder.build(_EGO_ID))
        self._last_obs = obs
        self._prev_reward_state = nxt

        # Rush mode: keep a band of background traffic around the ego
        # (despawn cars left behind, spawn fresh ones ahead). Runs only in
        # the normal continuation path.
        if self._density == "rush":
            self._manage_rush_traffic()

        truncated = self._step_count >= self._max_steps
        info: dict[str, Any] = {
            "reward_components": components,
            "step_count": self._step_count,
            "executed_action": int(executed),
            "lap_count": self._lap_count,
        }
        return obs, float(reward), False, truncated, info

    # ------------------------------------------------------------------
    # Rush-mode dynamic traffic (see _RUSH_* constants).
    # ------------------------------------------------------------------

    def _apply_rush_ego_tuning(self) -> None:
        """Tune the ego's SUMO car-following dynamics for the rush demo.
        Applied on the initial spawn. Scoped to the episode's SUMO process.

        Note: this deliberately does NOT touch the lane-change mode, so the
        `disable_sumo_lane_change(mode=0)` set in reset() stands and the
        DRIVING AGENT is the sole lane-change decider (no autonomous SUMO
        speed-gain). SUMO autonomous LC for a vehicle this much faster than
        traffic is unreliable — it camps in the leftmost lane and rarely
        overtakes. The `overtaker` baseline (the Aggressive preset's agent)
        does the weaving explicitly, so the decision ticker reflects the
        real maneuvers. See laneiq/agents/baselines/overtaker_agent.py."""
        # Krauss: fairly tight following + snappy accel so the ego holds
        # high speed except when a leader is close. tau=0.6 (not 0.4 —
        # 0.4 followed too close to brake in time and rear-ended slow
        # cars).
        traci.vehicle.setTau(_EGO_ID, 0.6)
        traci.vehicle.setMinGap(_EGO_ID, 1.5)
        traci.vehicle.setAccel(_EGO_ID, 4.0)
        # Strong braking as a collision safety net: at ~48 m/s the ego
        # can't always dodge a blocker (adjacent lane occupied), so it must
        # be able to scrub speed hard without rear-ending. Defaults
        # (decel 4.5) are too weak for this speed gap.
        traci.vehicle.setDecel(_EGO_ID, 7.5)
        # Lane-change mode 1024 = no autonomous LC, but TraCI lane-change
        # requests ARE safety-checked. Other scenarios use mode 0 (TraCI
        # changes executed unchecked) and rely on the trained policy to
        # never pick an unsafe change. The hand-coded `overtaker` has a
        # blind spot — a car sitting exactly alongside in the target lane
        # reads as neither a front nor a rear neighbour — so without the
        # safety check it would occasionally change into it. With 1024,
        # SUMO simply refuses an unsafe change (the ego holds its lane and
        # brakes instead). This is the rush demo's collision backstop.
        traci.vehicle.setLaneChangeMode(_EGO_ID, 1024)
        # Allow passing on the right. LC2013's default (lcOvertakeRight=0)
        # makes the ego slow to match any car in the lane to its LEFT, so a
        # same-speed wall in lanes 1-2 pinned it at ~33 m/s in an EMPTY
        # lane 0 (measured: overtaker mean 34 m/s → 45 m/s with this, 5
        # seeds, 0 collisions). Rush only; other scenarios keep SUMO's rule.
        traci.vehicle.setParameter(
            _EGO_ID, "laneChangeModel.lcOvertakeRight", "1",
        )

    def _seed_rush_traffic(self) -> None:
        """Pre-populate the band around the ego at reset so it meets
        traffic immediately. Spread from ~60 m to ~360 m ahead (covers the
        visible window plus the just-off-screen spawn zone) with jitter."""
        for i in range(_RUSH_TARGET_TRAFFIC):
            offset = 60 + (i + self.np_random.random()) * (300 / _RUSH_TARGET_TRAFFIC)
            self._spawn_rush_vehicle(offset)

    def _spawn_rush_vehicle(
        self, spawn_x: float, speed: str = "desired",
        lane: int | None = None, vtype: str | None = None,
    ) -> str | None:
        """Add one managed background vehicle at absolute road position
        `spawn_x` (clipped inside the edges — no wrapping; the rush road is
        a 10 km straight). Tracks the id in `_rush_ids` and returns it
        (or None on insertion failure — slot occupied, tried again next
        step)."""
        spawn_x = max(3.0, min(self._edge_length - 5.0, spawn_x))
        if vtype is None:
            vtype = "slow" if self.np_random.random() < _RUSH_SLOW_FRACTION else "normal"
        if lane is None:
            lane = int(self.np_random.integers(0, 3))
        vid = f"rush_{self._spawn_count}"
        self._spawn_count += 1
        try:
            traci.vehicle.add(
                vehID=vid, routeID=_ROUTE_ID, typeID=vtype,
                depart="now", departLane=str(lane),
                departSpeed=speed, departPos=f"{spawn_x:.2f}",
            )
        except traci.exceptions.TraCIException:
            return None
        self._rush_ids.add(vid)
        return vid

    def _manage_rush_traffic(self) -> None:
        """Keep a band of background traffic around the ego on the 10 km
        straight, biased so the ego always has something to overtake. Each
        step:

        1. **Prune** cars that genuinely left the sim (collided/exited).
        2. **Lock** newly-inserted cars to their spawn lane (no autonomous
           LC) so blockers stay in the ego's lane and scenery stays put.
        3. **Despawn** cars that have fallen more than
           `_RUSH_DESPAWN_BEHIND_M` behind the ego (it overtook them).
        4. **Refill** up to `_RUSH_TARGET_TRAFFIC`, spawning each car
           `_RUSH_SPAWN_AHEAD_M` ahead (off-screen). A `_RUSH_EGO_LANE_BIAS`
           share land in the ego's *current* lane as blockers; the rest are
           random scenery. The ego repeatedly catches a blocker in its lane,
           dodges into a clear lane, overtakes — what the `overtaker` agent
           turns into a visible weave.

        No wrapping: cars enter ahead and recede behind, exactly as on a
        real highway. This replaces the old circular model whose lap seam
        produced glitches.

        `traci.vehicle.add` only departs a car on the *next* sim step (and
        later still if its slot is briefly blocked), so a just-spawned id
        is in `_rush_ids` but not yet in `getIDList`. We must NOT treat
        such pending ids as "gone" — doing so would refill against them
        every step and over-spawn. `_rush_seen` records ids confirmed
        inserted at least once; only a *seen* id missing from the sim has
        truly left. The managed count for refill is `len(_rush_ids)`
        (pending included), which keeps the fleet at target."""
        try:
            alive = set(traci.vehicle.getIDList())
            ego_x = traci.vehicle.getLanePosition(_EGO_ID)
            ego_lane = traci.vehicle.getLaneIndex(_EGO_ID)
        except traci.exceptions.TraCIException:
            return

        # Lock each newly-inserted background car to its spawn lane
        # (setLaneChangeMode 0 = no autonomous LC). Otherwise SUMO's
        # keep-right pulls the slow blocker out of the ego's lane before
        # the ego ever reaches it, and scenery wanders unpredictably into
        # the dodge lanes. Must be done after the car departs (a pending
        # car isn't in `alive` yet), hence "newly confirmed".
        newly_seen = (self._rush_ids & alive) - self._rush_seen
        for vid in newly_seen:
            with suppress(traci.exceptions.TraCIException):
                traci.vehicle.setLaneChangeMode(vid, 0)

        # Confirm insertions, then prune only cars that were inserted and
        # have since vanished (collision/exit) — never pending ids.
        self._rush_seen |= (self._rush_ids & alive)
        gone = {v for v in self._rush_ids if v in self._rush_seen and v not in alive}
        self._rush_ids -= gone
        self._rush_seen -= gone

        # Despawn cars the ego has left behind (off the back of the window).
        for vid in list(self._rush_ids & alive):
            try:
                x = traci.vehicle.getLanePosition(vid)
            except traci.exceptions.TraCIException:
                continue
            if x < ego_x - _RUSH_DESPAWN_BEHIND_M:
                with suppress(traci.exceptions.TraCIException):
                    traci.vehicle.remove(vid)
                self._rush_ids.discard(vid)
                self._rush_seen.discard(vid)

        # Refill ahead, staggered so several added in one step don't stack
        # on the same insertion slot. A share of refills are placed as a
        # blocker in the ego's CURRENT lane (the weave forcer); the rest are
        # random scenery. Bounded by `_RUSH_TARGET_TRAFFIC` so the band
        # never piles into an inescapable wall.
        deficit = _RUSH_TARGET_TRAFFIC - len(self._rush_ids)
        for i in range(max(0, deficit)):
            spawn_x = ego_x + _RUSH_SPAWN_AHEAD_M + i * 25 \
                + self.np_random.random() * 40
            if spawn_x >= self._edge_length - 10.0:
                continue
            if self.np_random.random() < _RUSH_EGO_LANE_BIAS:
                self._spawn_rush_vehicle(spawn_x, lane=ego_lane, vtype="normal")
            else:
                self._spawn_rush_vehicle(spawn_x)

    def _reinsert_ego_for_lap(self) -> bool:
        """Re-add the ego at x=0 to start a new lap. Returns True iff the
        ego is visible in `getIDList` afterwards. Caller checks the bool
        and falls back to terminal handling on False (the only failure
        mode in practice is the depart slot being occupied at high traffic).

        Sim steps consumed by the insertion wait are NOT counted in
        `self._step_count` — they're internal env machinery, not policy
        decisions.
        """
        # Re-insertion speed = pre-wrap speed, so the ego doesn't
        # visibly slow from ~40 m/s to ego_depart_speed (20 m/s) every
        # lap. Falls back to depart speed only if there's no prior state.
        prev = self._prev_reward_state
        target_speed = prev.speed if prev is not None else self._ego_depart_speed
        # departLane="free" — SUMO picks the lane with the most room at
        # the seam. Re-inserting at a fixed lane failed (episode ended
        # with "exited") when that lane was occupied by background
        # traffic at the seam. The loop only runs for rush, where any
        # lane is fine.
        try:
            traci.vehicle.add(
                vehID=_EGO_ID, routeID=_ROUTE_ID, typeID="ego",
                depart="now",
                departLane="free",
                departSpeed=f"{target_speed:.2f}",
                departPos="0.0",
            )
        except traci.exceptions.TraCIException:
            return False
        for _ in range(50):
            traci.simulationStep()
            if _EGO_ID in traci.vehicle.getIDList():
                disable_sumo_lane_change(traci, _EGO_ID)
                # Re-apply the rush ego tuning — disable_sumo_lane_change
                # above reset the LC mode to 0, undoing it.
                if self._density == "rush":
                    self._apply_rush_ego_tuning()
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
