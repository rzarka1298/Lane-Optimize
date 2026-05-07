"""Safe rule-based policy: change lanes only when target is clear AND faster.

Encodes a "human driving instructor" heuristic:

1. Default to STAY.
2. If left lane is reachable AND clear (front + rear gaps above thresholds)
   AND projected speed strictly faster than current → CHANGE_LEFT.
3. Else if right lane is reachable AND clear AND faster → CHANGE_RIGHT.
4. Otherwise STAY.

What this models: a cautious driver who prioritizes not-crashing over
going-fast. Expect: low collision rate, modest mean speed. Useful as the
"safety-only" baseline in the Pareto plot — it bounds the safety axis.

Tries left first because in our 3-lane setup faster traffic is to the left
(slow vehicles default to lane 0); changing left is more likely to find
faster traffic flow.
"""

from __future__ import annotations

import numpy as np

from laneiq.agents.baselines.greedy_agent import _ego_lane_from_obs, _projected_speed
from laneiq.env.actions import Action, valid_actions_mask
from laneiq.env.observations import OBS_DIM


class SafeRuleAgent:
    """Conservative rule: change only if target lane is clear AND faster."""

    name: str = "safe_rule"

    def __init__(
        self,
        *,
        clear_front_gap_norm: float = 0.30,   # ≥ 30 m to leader in target lane
        clear_rear_gap_norm: float = 0.20,    # ≥ 20 m to follower in target lane
        faster_threshold_norm: float = 0.05,  # ≥ ~1.8 m/s faster
    ) -> None:
        self._clear_front = clear_front_gap_norm
        self._clear_rear = clear_rear_gap_norm
        self._faster_threshold = faster_threshold_norm

    def reset(self, *, seed: int | None = None) -> None:
        return

    def act(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | None = None,
    ) -> int:
        if observation.shape != (OBS_DIM,):
            raise ValueError(f"expected obs shape ({OBS_DIM},), got {observation.shape}")

        if action_mask is None:
            action_mask = valid_actions_mask(observation)

        ego_lane = _ego_lane_from_obs(observation)
        ego_speed_norm = float(observation[0])
        current_speed = _projected_speed(observation, ego_lane, ego_speed_norm)

        # Try left first (faster lanes tend to be left in our setup).
        if action_mask[Action.CHANGE_LEFT] and ego_lane + 1 <= 2:
            target = ego_lane + 1
            if self._lane_is_clear(observation, target) and self._is_faster(
                observation, target, current_speed, ego_speed_norm,
            ):
                return int(Action.CHANGE_LEFT)

        # Then try right.
        if action_mask[Action.CHANGE_RIGHT] and ego_lane - 1 >= 0:
            target = ego_lane - 1
            if self._lane_is_clear(observation, target) and self._is_faster(
                observation, target, current_speed, ego_speed_norm,
            ):
                return int(Action.CHANGE_RIGHT)

        return int(Action.STAY)

    # --- internals ----------------------------------------------------------

    def _lane_is_clear(self, observation: np.ndarray, lane: int) -> bool:
        """Both front AND rear gaps in `lane` exceed the thresholds."""
        front_offset = 4 + lane * 6
        rear_offset = front_offset + 3
        front_gap = float(observation[front_offset])
        rear_gap = float(observation[rear_offset])
        return front_gap >= self._clear_front and rear_gap >= self._clear_rear

    def _is_faster(
        self,
        observation: np.ndarray,
        target_lane: int,
        current_speed: float,
        ego_speed_norm: float,
    ) -> bool:
        target_speed = _projected_speed(observation, target_lane, ego_speed_norm)
        return target_speed > current_speed + self._faster_threshold
