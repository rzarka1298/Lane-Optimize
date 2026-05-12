"""Multi-agent variant of `LaneIQEnv` — N ego vehicles, one SUMO process.

PettingZoo `ParallelEnv` API: every step takes a dict {agent_id: action}
and returns dicts {agent_id: obs}, {agent_id: reward}, etc. Each agent
has the same observation space (25-dim) and action space (3 discrete) as
the single-agent env — symmetric, which is precisely what
parameter-sharing PPO needs.

Architecture:

- One SUMO process via the same `sumo_runtime.build_argv` machinery used
  by LaneIQEnv. TraCI single-connection-per-label is fine; we have N
  egos, not N TraCI connections.
- N=4 egos by default, spawned at distinct lane-positions during reset
  so they don't immediately collide.
- Each ego has `setLaneChangeMode(0)` so the policy is the sole decider
  (same as single-agent env).
- Per-step: each agent's reward is computed independently from its own
  RewardState. An optional small shared bonus (`shared_speed_bonus`) is
  added equal across all agents — the cooperative term, off by default.
- Agent removal: when an ego collides or exits, it's removed from
  `agents` (PettingZoo's "ended" signal). Episode terminates when no
  active agents remain or max_steps reached.

Why not `LaneIQEnv` subclass? Multi-agent step semantics are too
different — joint action handling, agent dict outputs, per-agent
termination. Cleaner to write a fresh class that calls the *internal*
helpers (`build_observation`, `apply_action`, `compute_reward`) shared
with single-agent.
"""

from __future__ import annotations

import logging
from contextlib import suppress
from dataclasses import replace
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
import traci
from pettingzoo import ParallelEnv

from laneiq.env.actions import (
    ACTION_SPACE,
    apply_action,
    disable_sumo_lane_change,
)
from laneiq.env.highway_env import (
    _DENSITY_TO_ROUTES,
    _ROUTE_ID,
    DensityT,
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

# We rebuild scenario knowledge here rather than importing from highway_env's
# constants to keep multi_agent_env independently testable.
_MA_TRACI_LABEL: str = "laneiq_ma_env"
_DEFAULT_N_EGOS: int = 4
# Default starting positions per ego (m). Spread along the road so no
# immediate collisions. Each ego in a different starting lane.
_DEFAULT_DEPART_POSITIONS_M: tuple[float, ...] = (0.0, 80.0, 160.0, 240.0)
_DEFAULT_DEPART_LANES: tuple[int, ...] = (1, 0, 2, 1)


class LaneIQParallelEnv(ParallelEnv):
    """N-ego LaneIQ env. PettingZoo ParallelEnv.

    `agents` lists currently-active ego IDs (those still in the SUMO sim).
    Once an ego terminates (collision/exit), it drops from `agents` AND
    from the dicts returned by `step()`. Episode ends when `agents` is
    empty or `step_count >= max_steps`.

    Observation per ego: same 25-dim vector as single-agent env (built
    by the same `build_observation` function — each agent sees the
    other egos as ordinary neighbors).

    Action per ego: same 3-discrete as single-agent env.

    Reward per ego: per-agent `compute_reward(prev, action, nxt)`. Plus
    an optional shared bonus = `shared_speed_bonus * mean(speed_term across active agents)`,
    added equally to every active agent. Default 0.0 (independent rewards).
    """

    metadata: ClassVar[dict[str, Any]] = {
        "name": "laneiq_parallel_v0",
        "render_modes": [],
    }

    def __init__(
        self,
        scenario_path: str | bytes = DEFAULT_SUMOCFG,
        density: DensityT = "medium",
        *,
        n_egos: int = _DEFAULT_N_EGOS,
        max_steps: int = 1000,
        warmup_steps: int = 50,
        gui: bool = False,
        seed: int | None = None,
        obs_config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
        reward_config: RewardConfig = DEFAULT_REWARD_CONFIG,
        shared_speed_bonus: float = 0.0,
        depart_positions_m: tuple[float, ...] | None = None,
        depart_lanes: tuple[int, ...] | None = None,
        depart_speed_mps: float = 18.0,
    ) -> None:
        if density not in _DENSITY_TO_ROUTES:
            raise ValueError(
                f"density must be one of {sorted(_DENSITY_TO_ROUTES)}, got {density!r}",
            )
        if n_egos < 1:
            raise ValueError(f"n_egos must be ≥1, got {n_egos}")

        self._scenario_path = scenario_path
        self._density: DensityT = density
        self._n_egos = n_egos
        self._max_steps = max_steps
        self._warmup_steps = warmup_steps
        self._gui = gui
        self._init_seed = seed
        self._obs_config = obs_config
        self._reward_config = reward_config
        self._shared_speed_bonus = shared_speed_bonus
        self._depart_speed_mps = depart_speed_mps

        # Resolve depart positions / lanes per-ego, with cycling defaults.
        if depart_positions_m is None:
            depart_positions_m = tuple(
                _DEFAULT_DEPART_POSITIONS_M[i % len(_DEFAULT_DEPART_POSITIONS_M)]
                for i in range(n_egos)
            )
        if depart_lanes is None:
            depart_lanes = tuple(
                _DEFAULT_DEPART_LANES[i % len(_DEFAULT_DEPART_LANES)]
                for i in range(n_egos)
            )
        if len(depart_positions_m) != n_egos:
            raise ValueError(
                f"depart_positions_m has length {len(depart_positions_m)}, "
                f"expected {n_egos}",
            )
        if len(depart_lanes) != n_egos:
            raise ValueError(
                f"depart_lanes has length {len(depart_lanes)}, expected {n_egos}",
            )
        self._depart_positions_m = depart_positions_m
        self._depart_lanes = depart_lanes

        # Canonical ego ids in arrival order. `possible_agents` is the
        # PettingZoo contract — always lists ALL egos, even terminated ones.
        self.possible_agents: list[str] = [f"ego_{i}" for i in range(n_egos)]
        self.agents: list[str] = []          # currently-active subset

        # Per-agent action/observation spaces. PettingZoo expects dicts.
        self._observation_spaces: dict[str, gym.spaces.Box] = {
            aid: OBSERVATION_SPEC.space for aid in self.possible_agents
        }
        self._action_spaces: dict[str, gym.spaces.Discrete] = {
            aid: ACTION_SPACE for aid in self.possible_agents
        }

        # Lazy state
        self._sumo_running: bool = False
        self._step_count: int = 0
        self._builder: ObservationBuilder | None = None
        self._prev_reward_states: dict[str, RewardState] = {}
        self._last_obs: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------------
    # PettingZoo API
    # ------------------------------------------------------------------

    def observation_space(self, agent: str) -> gym.spaces.Box:
        return self._observation_spaces[agent]

    def action_space(self, agent: str) -> gym.spaces.Discrete:
        return self._action_spaces[agent]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, dict[str, Any]]]:
        """Restart SUMO, warm up, insert all N egos, return per-agent obs."""
        resolved_seed = seed if seed is not None else self._init_seed
        self._close_sumo_safely()

        route_file = DEFAULT_SCENARIO_DIR / _DENSITY_TO_ROUTES[self._density]
        argv = build_argv(
            self._scenario_path,
            gui=self._gui,
            seed=resolved_seed,
            route_file=route_file,
        )

        traci.start(argv, label=_MA_TRACI_LABEL)
        self._sumo_running = True

        # Warmup so background traffic is non-empty before egos enter.
        for _ in range(self._warmup_steps):
            traci.simulationStep()

        # Add all N egos at their assigned (lane, position).
        for i, ego_id in enumerate(self.possible_agents):
            traci.vehicle.add(
                vehID=ego_id,
                routeID=_ROUTE_ID,
                typeID="ego",
                depart="now",
                departLane=str(self._depart_lanes[i]),
                departSpeed=str(self._depart_speed_mps),
                departPos=str(self._depart_positions_m[i]),
            )

        # Wait for all egos to actually appear in the vehicle list. At
        # high density a slot may be momentarily full; loop until insertion
        # or fail with a clear error.
        max_insertion_wait = 80
        for _ in range(max_insertion_wait):
            traci.simulationStep()
            in_list = set(traci.vehicle.getIDList())
            if all(aid in in_list for aid in self.possible_agents):
                break
        else:
            missing = [
                aid for aid in self.possible_agents
                if aid not in traci.vehicle.getIDList()
            ]
            raise RuntimeError(
                f"egos {missing!r} failed to insert within "
                f"{max_insertion_wait} sim steps at density={self._density!r}",
            )

        # Disable internal LC override for every ego.
        for aid in self.possible_agents:
            disable_sumo_lane_change(traci, aid)

        # Builder is shared (TraCI module-level state); each ego just calls
        # build with its own ego_id.
        self._builder = ObservationBuilder(traci, config=self._obs_config)
        self._builder.on_episode_start(self.possible_agents[0])  # caches v_max_lane

        self.agents = list(self.possible_agents)
        self._step_count = 0

        observations: dict[str, np.ndarray] = {}
        self._prev_reward_states = {}
        for aid in self.agents:
            obs = self._builder.build(aid)
            observations[aid] = obs
            self._prev_reward_states[aid] = self._extract_reward_state(aid)
        self._last_obs = dict(observations)

        infos: dict[str, dict[str, Any]] = {aid: {} for aid in self.agents}
        return observations, infos

    def step(
        self, actions: dict[str, int],
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, float],
        dict[str, bool],
        dict[str, bool],
        dict[str, dict[str, Any]],
    ]:
        """Apply each agent's action, advance one tick, return per-agent dicts.

        For agents that terminate (collision or off-road), this is their
        final step — their entries appear in the returned dicts, then
        they're dropped from `self.agents`.
        """
        if not self._sumo_running:
            raise RuntimeError("step() called before reset()")
        assert self._builder is not None

        # 1. Apply each agent's action (use STAY for any missing).
        for aid in self.agents:
            a = actions.get(aid, 0)  # default to STAY
            try:
                apply_action(traci, aid, a)
            except traci.exceptions.TraCIException as exc:
                _log.warning("apply_action failed for %s: %s", aid, exc)

        # 2. Advance SUMO one tick.
        traci.simulationStep()
        self._step_count += 1

        # 3. Determine which agents terminated this step.
        try:
            collision_ids = set(traci.simulation.getCollidingVehiclesIDList())
            in_list = set(traci.vehicle.getIDList())
        except traci.exceptions.TraCIException:
            collision_ids = set()
            in_list = set()

        # 4. Compute per-agent reward + observation. Build the response dicts
        # so terminated agents still appear with their last reward/obs.
        observations: dict[str, np.ndarray] = {}
        rewards: dict[str, float] = {}
        terminations: dict[str, bool] = {}
        truncations: dict[str, bool] = {}
        infos: dict[str, dict[str, Any]] = {}

        truncated_this_step = self._step_count >= self._max_steps

        # First pass: per-agent terms. Track speeds for shared bonus.
        agent_speed_terms: dict[str, float] = {}
        agent_done_info: dict[str, dict[str, Any]] = {}

        for aid in self.agents:
            prev = self._prev_reward_states[aid]
            ego_collided = aid in collision_ids
            ego_in_list = aid in in_list

            if ego_collided or not ego_in_list:
                # Terminal step for this agent.
                if ego_in_list:
                    nxt = self._extract_reward_state(aid, collided=ego_collided)
                else:
                    nxt = replace(prev, collided=ego_collided)
                reward, components = compute_reward(
                    prev, actions.get(aid, 0), nxt, config=self._reward_config,
                )
                rewards[aid] = float(reward)
                terminations[aid] = True
                truncations[aid] = False
                observations[aid] = self._last_obs[aid]  # final obs = previous
                infos[aid] = {
                    "reward_components": components,
                    "ego_terminated_reason": "collision" if ego_collided else "exited",
                    "step_count": self._step_count,
                }
                agent_speed_terms[aid] = 0.0
                continue

            # Normal continuation.
            nxt = self._extract_reward_state(aid)
            reward, components = compute_reward(
                prev, actions.get(aid, 0), nxt, config=self._reward_config,
            )
            rewards[aid] = float(reward)
            terminations[aid] = False
            truncations[aid] = truncated_this_step

            obs = self._builder.build(aid)
            observations[aid] = obs
            self._last_obs[aid] = obs
            self._prev_reward_states[aid] = nxt

            agent_speed_terms[aid] = components["speed"]
            agent_done_info[aid] = {
                "reward_components": components,
                "step_count": self._step_count,
            }
            infos[aid] = agent_done_info[aid]

        # 5. Optional shared-speed bonus (cooperative reward shaping).
        if self._shared_speed_bonus > 0.0 and agent_speed_terms:
            active_speeds = [
                v for aid, v in agent_speed_terms.items()
                if not terminations.get(aid, True)
            ]
            if active_speeds:
                bonus = self._shared_speed_bonus * float(np.mean(active_speeds))
                for aid in agent_done_info:
                    if not terminations[aid]:
                        rewards[aid] += bonus
                        infos[aid].setdefault("reward_components", {})
                        infos[aid]["reward_components"]["shared_bonus"] = bonus

        # 6. Drop terminated agents from `self.agents` for the next step.
        # PettingZoo's contract: an agent in `terminations`/`truncations`
        # with True is being removed. Next step won't include it.
        still_active = [
            aid for aid in self.agents
            if not (terminations.get(aid, False) or truncations.get(aid, False))
        ]
        self.agents = still_active

        return observations, rewards, terminations, truncations, infos

    def close(self) -> None:
        self._close_sumo_safely()

    def render(self) -> None:
        return None

    # ------------------------------------------------------------------
    # MaskablePPO support via supersuit (every agent must expose mask)
    # ------------------------------------------------------------------

    def action_masks(self) -> dict[str, np.ndarray]:
        """Per-agent valid-action masks. Used by MaskablePPO after
        supersuit flattening (each flattened env exposes its agent's mask).
        """
        from laneiq.env.actions import valid_actions_mask

        out: dict[str, np.ndarray] = {}
        for aid in self.agents:
            if aid in self._last_obs:
                out[aid] = valid_actions_mask(self._last_obs[aid])
        return out

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _close_sumo_safely(self) -> None:
        if self._builder is not None:
            with suppress(Exception):
                self._builder.on_episode_end()
            self._builder = None
        if self._sumo_running:
            with suppress(Exception):
                traci.close(False)
            self._sumo_running = False
        self._last_obs = {}
        self._prev_reward_states = {}
        self.agents = []
        self._step_count = 0

    def _extract_reward_state(
        self, ego_id: str, collided: bool = False,
    ) -> RewardState:
        """Pull a RewardState slice for one ego. Caller guarantees presence."""
        import math

        veh = traci.vehicle
        speed = float(veh.getSpeed(ego_id))
        accel = float(veh.getAcceleration(ego_id))
        lane_idx = int(veh.getLaneIndex(ego_id))

        leader = veh.getLeader(ego_id, self._obs_config.max_gap_m)
        if leader is not None and leader[0] != "":
            front_gap = float(leader[1])
            front_thw = front_gap / max(speed, self._obs_config.speed_eps)
        else:
            front_gap = math.inf
            front_thw = math.inf

        return RewardState(
            speed=speed, accel=accel, front_gap=front_gap, front_thw=front_thw,
            collided=collided, lane_idx=lane_idx,
        )
