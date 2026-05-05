"""Discrete lane-change actions for LaneIQEnv.

Three responsibilities:

1. Define the action enum and Gymnasium space.
2. Apply a chosen action to the ego via TraCI (`apply_action`).
3. Translate observations into a valid-action mask (`valid_actions_mask`),
   consumed both by the env's action filter and by the DQN's argmax.

Plus the load-bearing `disable_sumo_lane_change()` call: every episode,
right after the ego is added, we set its lane-change mode to 0. This
clears every internal SUMO lane-change motivation AND every safety
override — the ego moves only when the policy says so, and CAN crash
from bad policy decisions. The collision penalty in `rewards.py` is what
teaches the policy not to. See
`Project-Documentation/concepts/01-sumo-traffic-model.md` for the full
rationale.

Module is importable without a running SUMO; `traci_conn` is duck-typed
(real `traci` module in production, `Mock` in tests).
"""

from __future__ import annotations

from enum import IntEnum

import gymnasium as gym
import numpy as np

# Hardcoded for the 3-lane scenario. If we ever ship a multi-edge network
# with variable lane counts, lift this into ObservationConfig and read it
# from `traci.edge.getLaneNumber(...)` at episode start.
_MIN_LANE_IDX: int = 0
_MAX_LANE_IDX: int = 2


class Action(IntEnum):
    """Discrete RL actions. Values match the index in `ACTION_SPACE`.

    Lane indexing follows SUMO: lane 0 is rightmost, lane 2 is leftmost.
    `CHANGE_LEFT` thus *increases* the integer lane index; `CHANGE_RIGHT`
    *decreases* it. This is counterintuitive at first but matches the
    SUMO convention everywhere else in the codebase.
    """

    STAY = 0
    CHANGE_LEFT = 1
    CHANGE_RIGHT = 2


N_ACTIONS: int = 3
ACTION_SPACE: gym.spaces.Discrete = gym.spaces.Discrete(N_ACTIONS)

# Names parallel to the enum, useful for TensorBoard / decision-explanation logs.
ACTION_NAMES: tuple[str, ...] = ("stay", "change_left", "change_right")


# ---------------------------------------------------------------------------
# Per-episode lifecycle
# ---------------------------------------------------------------------------


def disable_sumo_lane_change(traci_conn, ego_id: str) -> None:
    """Disable every internal lane-change motivation and safety override
    on the ego vehicle.

    Called once per episode by `LaneIQEnv.reset()`, immediately after
    `traci.vehicle.add(ego_id, ...)` returns.

    `setLaneChangeMode(ego, 0)` clears every bit:

    - bits 0-1: strategic (route-following)
    - bits 2-3: cooperative (yield to others wanting to change)
    - bits 4-5: speed-gain
    - bits 6-7: keep-right
    - bits 8-9: sublane
    - bits 10-11: TraCI-requested override behavior

    With every bit clear, SUMO will execute exactly the lane changes
    `apply_action()` requests, including ones the LC2013 model would have
    blocked as unsafe. **This is intentional** — the policy is the sole
    decider, and it learns to avoid unsafe maneuvers via the collision
    penalty in `rewards.py`. See
    `Project-Documentation/concepts/01-sumo-traffic-model.md`.
    """
    traci_conn.vehicle.setLaneChangeMode(ego_id, 0)


# ---------------------------------------------------------------------------
# Per-step action application
# ---------------------------------------------------------------------------


def apply_action(
    traci_conn,
    ego_id: str,
    action: Action | int,
    *,
    duration_s: float = 1.0,
) -> None:
    """Apply a discrete lane-change decision via TraCI.

    Args:
        traci_conn: TraCI module (or test mock).
        ego_id: vehicle ID.
        action: STAY → no-op; CHANGE_LEFT → lane+1; CHANGE_RIGHT → lane-1.
            Accepts a raw int for convenience (the env may pass through
            whatever the policy returns without coercing).
        duration_s: how long the lateral maneuver should take. SUMO uses
            this to derive lateral acceleration; 1.0 s is a normal urgency,
            close to a real driver's lane change.

    The function is a **silent no-op** if the target lane doesn't exist
    (e.g., CHANGE_LEFT while in lane 2). The action mask in
    `valid_actions_mask()` should prevent this; the no-op is defense in
    depth, not the primary guard. Callers SHOULD mask in DQN's argmax
    and SHOULD use `MaskablePPO` for PPO so that invalid actions never
    leave the policy.
    """
    action = Action(int(action))

    if action == Action.STAY:
        return

    current = int(traci_conn.vehicle.getLaneIndex(ego_id))
    target = current + 1 if action == Action.CHANGE_LEFT else current - 1

    if not (_MIN_LANE_IDX <= target <= _MAX_LANE_IDX):
        return  # silent no-op; the mask should have prevented this

    traci_conn.vehicle.changeLane(ego_id, target, duration_s)


# ---------------------------------------------------------------------------
# Action validity mask — read from observation[23]
# ---------------------------------------------------------------------------


def valid_actions_mask(observation: np.ndarray) -> np.ndarray:
    """Return a shape-(3,) bool array marking which actions are valid.

    `STAY` is always valid. `CHANGE_LEFT` / `CHANGE_RIGHT` validity is
    recovered from `observation[23]` (the packed lane-validity scalar
    set by `build_observation` — see
    `Project-Documentation/laneiq/env/observations.md` for the encoding).

    Args:
        observation: shape-(25,) float32 array from `build_observation`.

    Returns:
        bool array of shape (3,): `[stay_valid, left_valid, right_valid]`.

    Raises:
        ValueError: if `observation` doesn't have shape (25,).
    """
    if observation.shape != (25,):
        raise ValueError(f"expected shape (25,), got {observation.shape}")

    # Packed value: (left_ok * 2 + right_ok) / 3 ∈ {0, 1/3, 2/3, 1}.
    # Round-then-clamp is robust to small float drift after np.clip.
    pack = round(float(observation[23]) * 3)
    pack = max(0, min(3, pack))

    left_ok = bool((pack >> 1) & 1)
    right_ok = bool(pack & 1)

    return np.array([True, left_ok, right_ok], dtype=bool)


def mask_invalid_q_values(q_values: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Return Q-values with invalid-action positions set to -inf.

    Used by the DQN's ε-greedy / argmax path so an invalid action is
    never chosen. Does NOT modify the input.

    Args:
        q_values: shape-(3,) array of Q-values from the network.
        mask: shape-(3,) bool from `valid_actions_mask()`.

    Returns:
        shape-(3,) array — same as `q_values` but with positions where
        `mask == False` replaced by -inf.
    """
    if q_values.shape != mask.shape:
        raise ValueError(
            f"shape mismatch: q_values {q_values.shape} vs mask {mask.shape}"
        )
    out = q_values.astype(np.float64, copy=True)
    out[~mask] = -np.inf
    return out
