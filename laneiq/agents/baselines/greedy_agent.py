"""Greedy "highest-projected-speed" policy.

For each reachable lane, project what the ego's speed would be if it sat
behind that lane's leader (= ego_speed + relative_speed_of_leader). Pick
the lane with the highest projected speed. Tie-break in favor of STAY so
the policy doesn't thrash.

What this models: a myopic optimizer that only cares about going fast right
now. Ignores safety entirely (no gap check, no follower check). Will
happily change into a tight gap. Expect: high mean speed, high collision
rate. Useful as the "speed-only" baseline in the Pareto plot — it bounds
the speed axis.
"""

from __future__ import annotations

import numpy as np

from laneiq.env.actions import Action, valid_actions_mask
from laneiq.env.observations import OBS_DIM


class GreedyAgent:
    """Picks the lane with the highest projected speed."""

    name: str = "greedy"

    def __init__(self) -> None:
        return  # stateless

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

        scores = {
            Action.STAY: _projected_speed(observation, ego_lane, ego_speed_norm),
        }
        if action_mask[Action.CHANGE_LEFT] and ego_lane + 1 <= 2:
            scores[Action.CHANGE_LEFT] = _projected_speed(
                observation, ego_lane + 1, ego_speed_norm,
            )
        if action_mask[Action.CHANGE_RIGHT] and ego_lane - 1 >= 0:
            scores[Action.CHANGE_RIGHT] = _projected_speed(
                observation, ego_lane - 1, ego_speed_norm,
            )

        # Pick the highest-scoring action; tie-break to STAY.
        best_action: Action = Action.STAY
        best_score: float = scores[Action.STAY]
        for action in (Action.CHANGE_LEFT, Action.CHANGE_RIGHT):
            if action in scores and scores[action] > best_score:
                best_score = scores[action]
                best_action = action

        return int(best_action)


# --- shared helpers (also used by safe_rule_agent) ---------------------------


def _ego_lane_from_obs(observation: np.ndarray) -> int:
    """Recover the integer ego lane from `obs[1]` (= lane / 2.0)."""
    raw = round(float(observation[1]) * 2)
    return max(0, min(2, raw))


def _projected_speed(
    observation: np.ndarray, lane: int, ego_speed_norm: float,
) -> float:
    """What speed (normalized) would the ego cruise at if it were in `lane`?

    If the lane has no leader (vc sentinel = -1) → assume open lane → 1.0.
    Otherwise → ego_speed + dv (where dv = leader_speed - ego_speed,
    both normalized).
    """
    offset = 4 + lane * 6
    vc = float(observation[offset + 2])
    dv = float(observation[offset + 1])
    if vc <= -0.99:  # missing-neighbor sentinel
        return 1.0   # open lane → free to accelerate to v_max
    return ego_speed_norm + dv
