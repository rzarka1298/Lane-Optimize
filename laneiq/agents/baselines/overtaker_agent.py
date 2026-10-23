"""Aggressive overtaker policy: live in the fast lanes, dart around blockers.

The Aggressive demo wants a car that is visibly the fastest on the road and
*weaves* through slower traffic. The trained PPO policies don't do this
(they converge to a disciplined fast-lane cruiser — see RESULTS.md), and
SUMO's autonomous lane-change model is unreliable for a vehicle this much
faster than traffic (it camps in the leftmost lane and rarely overtakes).
So the Aggressive preset drives with this explicit rule instead, and
because it acts through the normal action path the decision ticker shows
its real maneuvers.

The behaviour is a "fast-lane weaver":

1. **Held up** — a close, slower leader is ahead in the current lane. Dodge
   into an adjacent lane that is *clear* (real front AND rear gaps). Prefer
   left (toward the fast side); fall back to right. If boxed in, STAY and
   let the Krauss follower brake until a gap opens.
2. **Clear ahead, but not in the leftmost lane** — drift left whenever the
   left lane is clear. This is the "come back up after passing" move: once
   the ego has slipped past the blocker it left behind, the lane above
   opens and it climbs back toward lane 2. The net pattern is dip-down /
   overtake / climb-back — a continuous, readable weave that stays in the
   upper lanes (so it never gets trapped behind the slow cars that pile up
   in lane 0).
3. **Cruising in lane 2 with open road** — STAY at top speed.

It is "greedy with a safety gate": eager to overtake, but it never changes
into a lane without a verified gap, so it doesn't cause collisions. Pairs
with the rush env's weave-blocker mechanism (`_manage_rush_traffic`), which
keeps one slow car in the ego's current lane so there is always something
to overtake.
"""

from __future__ import annotations

import numpy as np

from laneiq.agents.baselines.greedy_agent import _ego_lane_from_obs
from laneiq.env.actions import Action, valid_actions_mask
from laneiq.env.observations import OBS_DIM

_TOP_LANE: int = 2  # leftmost / "home" lane


class OvertakerAgent:
    """Lives in the upper lanes; dodges around close slow leaders and climbs
    back up after each pass."""

    name: str = "overtaker"

    def __init__(
        self,
        *,
        held_up_gap_norm: float = 0.65,    # leader within ~65 m → start dodging
        slower_dv_norm: float = -0.03,     # leader ≥ ~1.5 m/s slower than ego
        clear_front_norm: float = 0.35,    # target lane: ≥ ~35 m to its leader
        clear_rear_norm: float = 0.18,     # target lane: ≥ ~18 m to its follower
        roomier_gap_norm: float = 0.20,    # fallback dodge: ≥ ~20 m more room
        commit_steps: int = 15,            # hold a lane this long after a change
        cruise_ratio: float = 0.85,        # below this x lane limit behind a leader → held up
    ) -> None:
        self._held_up_gap = held_up_gap_norm
        self._slower_dv = slower_dv_norm
        self._clear_front = clear_front_norm
        self._clear_rear = clear_rear_norm
        self._roomier_gap = roomier_gap_norm
        self._commit_steps = commit_steps
        self._cruise_ratio = cruise_ratio
        self._cooldown = 0

    def reset(self, *, seed: int | None = None) -> None:
        self._cooldown = 0

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

        # Commit to the current lane for ~1.5 s after a change. Without this
        # the rule flip-flops every tick (dodge left, then the new lane's
        # leader triggers a dodge right, …) which both looks bad and can
        # walk the ego into a car sitting alongside it.
        if self._cooldown > 0:
            self._cooldown -= 1
            return int(Action.STAY)

        ego_lane = _ego_lane_from_obs(observation)

        if self._is_held_up(observation, ego_lane):
            # Dodge into an adjacent lane that is clear AND free of its own
            # blocker, preferring left (toward the fast side). Only a lane we
            # would not immediately want to leave again counts — this is the
            # hysteresis that stops the dodge-right / climb-left ping-pong
            # behind a single slow car.
            candidates = [
                (action, target)
                for action, target in (
                    (Action.CHANGE_LEFT, ego_lane + 1),
                    (Action.CHANGE_RIGHT, ego_lane - 1),
                )
                if action_mask[action]
                and 0 <= target <= _TOP_LANE
                and self._lane_is_clear(observation, target)
            ]
            for action, target in candidates:
                if not self._is_blocked(observation, target):
                    return self._commit(action)
            # Every clear lane has a blocker too: still dodge if one gives
            # meaningfully more room than the current leader.
            own_gap = float(observation[4 + ego_lane * 6])
            for action, target in candidates:
                if float(observation[4 + target * 6]) >= own_gap + self._roomier_gap:
                    return self._commit(action)
            return int(Action.STAY)  # boxed in → brake and wait for a gap

        # Not held up: climb back toward the top lane after a pass — but only
        # once the car we dodged is no longer a blocker up there. Climbing
        # into a lane where we'd be held up straight away is what used to
        # undo every overtake within a second.
        target = ego_lane + 1
        if (
            ego_lane < _TOP_LANE
            and action_mask[Action.CHANGE_LEFT]
            and self._lane_is_clear(observation, target)
            and not self._would_catch(observation, target)
        ):
            return self._commit(Action.CHANGE_LEFT)

        return int(Action.STAY)

    def _commit(self, action: Action) -> int:
        """Start the post-change cooldown and return `action`."""
        self._cooldown = self._commit_steps
        return int(action)

    def _is_held_up(self, observation: np.ndarray, lane: int) -> bool:
        """Dodge trigger for the ego's own lane: a close slower leader, OR
        a leader at any range that isn't pulling away while the ego is well
        under the lane limit. The second case is SUMO car-following parking
        the ego far behind a same-speed leader (rush: ~106 m at 33 m/s on a
        50 m/s road) — out of `held_up_gap` range, so the gap rule alone
        never fires and the ego cruises at traffic speed forever."""
        if self._is_blocked(observation, lane):
            return True
        off = 4 + lane * 6
        leader_present = float(observation[off + 2]) > -0.99
        return (
            leader_present
            and float(observation[0]) < self._cruise_ratio
            and float(observation[off + 1]) <= 0.02
        )

    def _would_catch(self, observation: np.ndarray, lane: int) -> bool:
        """Climb-back veto: `lane` has a leader at ANY range that is slower
        than the ego, or one that would hold it under cruise. Range-limited
        checks let the ego climb back in behind the car it just dodged
        (far, but still ahead), then dodge again ~2 s later."""
        off = 4 + lane * 6
        leader_present = float(observation[off + 2]) > -0.99
        return self._is_held_up(observation, lane) or (
            leader_present and float(observation[off + 1]) < self._slower_dv
        )

    def _is_blocked(self, observation: np.ndarray, lane: int) -> bool:
        """`lane` has a close, slower leader — the ego would be held up there.
        Used for the current lane (dodge trigger) and for target lanes
        (don't change into a lane you'd immediately want to leave)."""
        off = 4 + lane * 6
        leader_present = float(observation[off + 2]) > -0.99
        return (
            leader_present
            and float(observation[off]) < self._held_up_gap
            and float(observation[off + 1]) < self._slower_dv
        )

    def _lane_is_clear(self, observation: np.ndarray, lane: int) -> bool:
        """Both front AND rear gaps in `lane` exceed the (aggressive)
        thresholds — enough room to change in without a collision."""
        front_offset = 4 + lane * 6
        rear_offset = front_offset + 3
        front_gap = float(observation[front_offset])
        rear_gap = float(observation[rear_offset])
        return front_gap >= self._clear_front and rear_gap >= self._clear_rear
