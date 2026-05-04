"""Observation builder for LaneIQEnv.

Translates the live TraCI state of the ego vehicle into a fixed-length 25-dim
float32 vector clipped to [-1, 1] — the Markov state the RL policy sees.

Design decisions are documented in:

- `Project-Documentation/laneiq/env/observations.md` (per-file API reference)
- `Project-Documentation/concepts/01-sumo-traffic-model.md` (why these
  features mirror SUMO's two-layer behavior model)

Two surfaces are exposed:

- **`build_observation(traci_conn, ego_id)`** — pure function. Used in tests
  (with a mock `traci_conn`) and in any one-off script that doesn't want to
  carry state.
- **`ObservationBuilder`** — stateful wrapper that caches the lane speed
  limit and (eventually, post-Week-3 profiling) owns TraCI subscriptions.
  Used by `LaneIQEnv` so per-step TraCI traffic is amortized across the
  episode. Same wire format as the pure function.

The module DOES NOT manage SUMO lifecycle — callers must guarantee a TraCI
connection is open and `ego_id` is in the current vehicle list.
"""

from __future__ import annotations

from contextlib import suppress
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

# ---------------------------------------------------------------------------
# Module-level constants — keep in 1:1 sync with observations.md and the
# layout table in the plan file.
# ---------------------------------------------------------------------------

OBS_DIM: int = 25

# Order matches the layout table; lane index ascends 0 (right) → 2 (left).
OBS_NAMES: tuple[str, ...] = (
    "ego_speed",
    "ego_lane_idx",
    "ego_accel",
    "ego_lat_pos",
    "lane0_front_gap", "lane0_front_dv", "lane0_front_vc",
    "lane0_rear_gap",  "lane0_rear_dv",  "lane0_rear_vc",
    "lane1_front_gap", "lane1_front_dv", "lane1_front_vc",
    "lane1_rear_gap",  "lane1_rear_dv",  "lane1_rear_vc",
    "lane2_front_gap", "lane2_front_dv", "lane2_front_vc",
    "lane2_rear_gap",  "lane2_rear_dv",  "lane2_rear_vc",
    "local_density",
    "lane_validity_packed",
    "ego_thw_to_leader",
)
assert len(OBS_NAMES) == OBS_DIM, "OBS_NAMES length must equal OBS_DIM"

# Scalar bucket per vType id. Order chosen so (id-1)/2 ∈ {-0.5, 0, 0.5, 1.0}.
# Anything not in this map (or no neighbor) → -1.0 sentinel.
_VCLASS_BUCKET: dict[str, int] = {
    "aggressive": 0,
    "normal": 1,
    "slow": 2,
    "ego": 3,
}

# Defensive default if the lane speed limit can't be read from TraCI.
# Matches the speed authored in `sumo_scenarios/highway_3lane/highway.edg.xml`.
_DEFAULT_V_MAX_LANE: float = 36.11

# SUMO `getNeighbors` mode bits:
#   bit0 = left(1) / right(0)
#   bit1 = leader(1) / follower(0)
# Map each side-mode to the (lane offset, slot name) it covers.
_SIDE_MODES: tuple[tuple[int, int, str], ...] = (
    (0b011, +1, "front"),  # left-leader  (lane = ego_lane + 1)
    (0b001, +1, "rear"),   # left-follower
    (0b010, -1, "front"),  # right-leader (lane = ego_lane - 1)
    (0b000, -1, "rear"),   # right-follower
)


# ---------------------------------------------------------------------------
# Public dataclasses — frozen so they hash/compare cleanly inside RewardConfig
# and any future YAML→dataclass loader.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationConfig:
    """Tunable constants for the observation builder.

    Defaults match the v1 design in the plan file. Override per-experiment
    by passing a constructed instance to `build_observation` or
    `ObservationBuilder`.
    """

    max_gap_m: float = 100.0          # also the search radius for getLeader/getNeighbors
    max_density_count: int = 20       # vehicles within ±max_gap_m → density=1.0
    accel_norm: float = 4.5           # ego.decel; sets dim[2] scale
    thw_clip_s: float = 5.0           # cap on time-headway; >5s reads as "safe"
    speed_eps: float = 0.5            # avoid div-by-zero in THW when ego stopped
    edge_id: str = "e1"               # for lane.getMaxSpeed lookup
    lane_width_m: float = 3.2         # SUMO's default lane width


@dataclass(frozen=True)
class ObservationSpec:
    """Module-level metadata: dim count, names, and Gymnasium space."""

    dim: int
    names: tuple[str, ...]
    space: gym.spaces.Box


OBSERVATION_SPEC: ObservationSpec = ObservationSpec(
    dim=OBS_DIM,
    names=OBS_NAMES,
    space=gym.spaces.Box(low=-1.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32),
)

# Shared default config — frozen, so safe to use as a sentinel default for
# function arguments without B008 warnings.
DEFAULT_OBSERVATION_CONFIG: ObservationConfig = ObservationConfig()


# ---------------------------------------------------------------------------
# Internal helpers (private; tested via the public API).
# ---------------------------------------------------------------------------


def _vclass_id(type_id: str) -> int:
    """Map a SUMO vType string to its scalar bucket (or -1 if unknown)."""
    return _VCLASS_BUCKET.get(type_id, -1)


def _normalize_vclass(vc: int) -> float:
    """Map bucket id to [-1, 1]: missing → -1, aggressive → -0.5, ..., ego → 1."""
    if vc < 0:
        return -1.0
    return (vc - 1) / 2.0


def _lookup_v_max_lane(traci_conn, edge_id: str) -> float:
    """Read the leftmost lane's speed limit. Falls back to the documented
    default if TraCI raises (race conditions, edge id typo, etc.).
    """
    try:
        return float(traci_conn.lane.getMaxSpeed(f"{edge_id}_2"))
    except Exception:
        return _DEFAULT_V_MAX_LANE


def _safe_neighbor_attrs(traci_conn, neighbor_id: str) -> tuple[float, str] | None:
    """Look up a neighbor's (speed, type). Returns None if the neighbor
    has exited between the `getNeighbors` call and this lookup (rare race).
    """
    try:
        return float(traci_conn.vehicle.getSpeed(neighbor_id)), traci_conn.vehicle.getTypeID(neighbor_id)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def ego_present(traci_conn, ego_id: str) -> bool:
    """True iff `ego_id` is in the current vehicle list. Cheap; safe to call
    before `build_observation` to avoid the KeyError path during reset.
    """
    return ego_id in traci_conn.vehicle.getIDList()


def build_observation(
    traci_conn,
    ego_id: str,
    *,
    config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
    v_max_lane: float | None = None,
) -> np.ndarray:
    """Read TraCI state for `ego_id`; return a shape-(25,) float32 array.

    Args:
        traci_conn: anything with the TraCI module's call surface — the real
            `traci` module in production, a `Mock` in unit tests.
        ego_id: the vehicle ID of the RL-controlled ego.
        config: tunable normalization constants. Keep frozen across an episode.
        v_max_lane: optional pre-cached lane speed limit (m/s). If `None`,
            looked up via `_lookup_v_max_lane()` once per call.

    Raises:
        KeyError: if `ego_id` is not in the current vehicle list. Callers
            (`LaneIQEnv.reset`/`step`) must guarantee presence.

    Notes:
        Result is clipped to [-1, 1] dim-wise as the final operation. No NaN
        or Inf can leak through (anything division-related is guarded).
    """
    if not ego_present(traci_conn, ego_id):
        raise KeyError(f"ego {ego_id!r} not in vehicle list")

    if v_max_lane is None:
        v_max_lane = _lookup_v_max_lane(traci_conn, config.edge_id)
    if v_max_lane <= 0:
        v_max_lane = _DEFAULT_V_MAX_LANE

    obs = np.zeros(OBS_DIM, dtype=np.float32)
    veh = traci_conn.vehicle

    # --- Ego state (dims 0-3) -----------------------------------------------
    ego_speed = float(veh.getSpeed(ego_id))
    ego_lane_raw = int(veh.getLaneIndex(ego_id))
    ego_lane = max(0, min(2, ego_lane_raw))     # clamp; SUMO can return -1 mid-LC
    ego_accel = float(veh.getAcceleration(ego_id))
    ego_lat = float(veh.getLateralLanePosition(ego_id))
    ego_pos = float(veh.getLanePosition(ego_id))

    obs[0] = ego_speed / v_max_lane
    obs[1] = ego_lane / 2.0
    obs[2] = ego_accel / config.accel_norm
    obs[3] = ego_lat / (config.lane_width_m / 2.0)

    # --- Neighbor lookup: build a (lane, slot) → (gap, dv, vc) dict ---------
    # Same-lane front via getLeader / rear via getFollower.
    # Side lanes via getNeighbors with the four mode bitmasks.
    neighbors: dict[tuple[int, str], tuple[float, float, int]] = {}

    leader = veh.getLeader(ego_id, config.max_gap_m)
    if leader is not None and leader[0] != "":
        l_id, l_gap = leader
        attrs = _safe_neighbor_attrs(traci_conn, l_id)
        if attrs is not None:
            l_speed, l_type = attrs
            neighbors[(ego_lane, "front")] = (l_gap, l_speed - ego_speed, _vclass_id(l_type))

    follower = veh.getFollower(ego_id, config.max_gap_m)
    # SUMO returns ("", -1.0) when no follower in range.
    if follower is not None and follower[0] != "":
        f_id, f_gap = follower
        attrs = _safe_neighbor_attrs(traci_conn, f_id)
        if attrs is not None:
            f_speed, f_type = attrs
            neighbors[(ego_lane, "rear")] = (f_gap, f_speed - ego_speed, _vclass_id(f_type))

    for mode, lane_offset, slot in _SIDE_MODES:
        target_lane = ego_lane + lane_offset
        if not (0 <= target_lane <= 2):
            continue                            # lane doesn't exist (ego at edge)
        results = veh.getNeighbors(ego_id, mode)
        if not results:
            continue
        # `results` is a tuple of (vehID, gap); pick the closest by gap.
        n_id, n_gap = min(results, key=lambda pair: pair[1])
        attrs = _safe_neighbor_attrs(traci_conn, n_id)
        if attrs is None:
            continue                            # raced; treat as missing
        n_speed, n_type = attrs
        neighbors[(target_lane, slot)] = (n_gap, n_speed - ego_speed, _vclass_id(n_type))

    # --- Fill the 18 neighbor dims (idx 4-21) ------------------------------
    base = 4
    for lane in (0, 1, 2):
        for slot_idx, slot in enumerate(("front", "rear")):
            offset = base + (lane * 6) + (slot_idx * 3)
            if (lane, slot) in neighbors:
                gap, dv, vc = neighbors[(lane, slot)]
                obs[offset]     = gap / config.max_gap_m
                obs[offset + 1] = dv / v_max_lane
                obs[offset + 2] = _normalize_vclass(vc)
            else:
                obs[offset]     = 1.0
                obs[offset + 1] = 0.0
                obs[offset + 2] = -1.0

    # --- Local density (dim 22) --------------------------------------------
    nearby = 0
    for vid in veh.getIDList():
        if vid == ego_id:
            continue
        with suppress(Exception):
            other_pos = float(veh.getLanePosition(vid))
            if abs(other_pos - ego_pos) <= config.max_gap_m:
                nearby += 1
    obs[22] = min(nearby / config.max_density_count, 1.0)

    # --- Lane validity packed (dim 23) -------------------------------------
    # Pack: (left_ok * 2 + right_ok) / 3 ∈ {0, 1/3, 2/3, 1}.
    left_ok = 1 if ego_lane < 2 else 0
    right_ok = 1 if ego_lane > 0 else 0
    obs[23] = (left_ok * 2 + right_ok) / 3.0

    # --- Time-headway to ego's leader (dim 24) -----------------------------
    if (ego_lane, "front") in neighbors:
        gap_f = neighbors[(ego_lane, "front")][0]
        v_safe = max(ego_speed, config.speed_eps)
        thw = gap_f / v_safe
        obs[24] = min(thw, config.thw_clip_s) / config.thw_clip_s
    else:
        obs[24] = 1.0                           # no leader → safest possible

    # Final clip — defensive; everything above already targets [-1, 1].
    np.clip(obs, -1.0, 1.0, out=obs)
    return obs


# ---------------------------------------------------------------------------
# Stateful wrapper — what `LaneIQEnv` will hold across episodes
# ---------------------------------------------------------------------------


class ObservationBuilder:
    """Caches per-episode constants and (eventually) owns TraCI subscriptions.

    Identical wire format to `build_observation()` — this is purely a
    perf-amortization wrapper. The env calls:

        builder = ObservationBuilder(traci, config=...)
        # at reset:
        builder.on_episode_start("ego_0")
        # each step:
        obs = builder.build("ego_0")
        # at done:
        builder.on_episode_end()

    Subscriptions are not yet enabled — Week 3 profiling (Task #5) decides.
    The hooks are in place so flipping them on is a one-file change.
    """

    def __init__(
        self,
        traci_conn,
        *,
        config: ObservationConfig = DEFAULT_OBSERVATION_CONFIG,
    ) -> None:
        self._traci = traci_conn
        self._config = config
        self._v_max_lane: float | None = None

    def on_episode_start(self, ego_id: str) -> None:
        """Cache the lane speed limit for the new episode."""
        self._v_max_lane = _lookup_v_max_lane(self._traci, self._config.edge_id)
        # TODO(week3-profile): enable `traci.vehicle.subscribe(ego_id, [...])` here.

    def on_vehicle_seen(self, veh_id: str) -> None:
        """No-op for now. Hook for lazy subscription on first sighting."""
        return

    def build(self, ego_id: str) -> np.ndarray:
        """Build the observation using the cached lane speed limit."""
        return build_observation(
            self._traci,
            ego_id,
            config=self._config,
            v_max_lane=self._v_max_lane,
        )

    def on_episode_end(self) -> None:
        """Clear cached state. Hook for unsubscribing in the future."""
        self._v_max_lane = None
        # TODO(week3-profile): `traci.vehicle.unsubscribe(...)` for every subscribed id.
