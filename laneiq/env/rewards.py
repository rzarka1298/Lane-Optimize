"""Reward composer for LaneIQEnv.

Six weighted components, each with a clear motivation. Returns both the
scalar reward and a per-component dict so TensorBoard can log
`reward/components/<key>` per step — the single most useful diagnostic
when debugging reward-shape pathologies.

The component weights are tuned for the Week-4 Pareto sweep, which varies
them via `configs/reward/*.yaml` to produce the safety-vs-speed trade-off
plot. v1 defaults documented in `RewardConfig` below; full design context
in `Project-Documentation/concepts/02-reward-shaping.md`.

Why this is a *separate* module from `actions.py` and `highway_env.py`:

- Pure function of `(prev, action, next, config) -> (scalar, components)`.
  No TraCI, no env state, no I/O. Easy to test, easy to swap.
- Per-experiment config swaps (the Pareto sweep) only touch this module.
- Multi-agent RL (Week 5) reuses the same composer per-agent, possibly
  with a small shared bonus term added on top.

Invariant: this module **does not import the normalized observation**. It
operates on the raw `RewardState` slice the env extracts directly from
TraCI. Renormalizing later would couple the obs scale to the reward scale —
exactly the kind of silent dependency `Project-Documentation/setup.md`
warns against.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from laneiq.env.actions import Action

# ---------------------------------------------------------------------------
# Config + state dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RewardConfig:
    """Tunable reward weights and thresholds.

    Defaults are the v1 design from the master plan. The Week-4 Pareto
    sweep mutates these via `dataclasses.replace(config, w_speed=...)` to
    produce the safety-vs-speed trade-off plot.

    All weights have intuitive signs: speed and alive are positive
    incentives, the other four are penalties (negative).
    """

    # ---- Component weights (signs intentional) ----
    w_speed: float = 1.0          # +; reward fast cruising
    w_lane_cost: float = -0.05    # -; small per-step cost on any lane change
    w_jerk: float = -0.2          # -; penalize abrupt accel/decel changes
    w_unsafe_gap: float = -0.5    # -; penalize tailgating
    w_collision: float = -50.0    # -; terminal, dominant
    w_alive: float = 0.01         # +; small dense survival bonus

    # ---- Thresholds and normalizers ----
    v_max: float = 36.11          # m/s; lane speed limit (matches highway.edg.xml)
    unsafe_thw_s: float = 1.5     # s; below this time-headway → unsafe-gap penalty
    a_max_norm: float = 4.5       # m/s^2; jerk denominator (matches ego.decel)


# Module-level singleton — the same idiom as observations.DEFAULT_OBSERVATION_CONFIG.
# Avoids B008 ("function call in default arg") without sacrificing the convenience.
DEFAULT_REWARD_CONFIG: RewardConfig = RewardConfig()


@dataclass(frozen=True)
class RewardState:
    """Minimal slice of ego state the reward needs.

    Extracted by `LaneIQEnv.step()` from the raw TraCI snapshot — NOT from
    the normalized observation vector. Keeping reward inputs in raw units
    means future reward edits don't need de-normalization arithmetic.

    Attributes:
        speed: ego speed in m/s.
        accel: signed longitudinal acceleration in m/s^2.
        front_gap: m to leader in same lane; `math.inf` if no leader.
        front_thw: time-headway to leader = `front_gap / max(speed, eps)`;
            `math.inf` if no leader. Pre-computed by the env so the reward
            doesn't need to know the speed-eps constant.
        collided: True iff ego is in `getCollidingVehiclesIDList()` this step.
        lane_idx: integer lane index 0..2.
    """

    speed: float
    accel: float
    front_gap: float
    front_thw: float
    collided: bool
    lane_idx: int


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Stable order so the per-component dict iterates predictably (helpful for
# CSV logging and human eyeball-checking against TensorBoard).
COMPONENT_KEYS: tuple[str, ...] = (
    "speed",
    "lane_cost",
    "jerk",
    "unsafe_gap",
    "collision",
    "alive",
)


def compute_reward(
    prev: RewardState,
    action: Action | int,
    nxt: RewardState,
    *,
    config: RewardConfig = DEFAULT_REWARD_CONFIG,
) -> tuple[float, dict[str, float]]:
    """Compute the per-step reward.

    Args:
        prev: ego state before the action was applied. The env caches this
            from the previous step's snapshot (or constructs it once at
            reset for the first step).
        action: the integer action taken this step. STAY (0) is the no-op;
            CHANGE_LEFT (1) and CHANGE_RIGHT (2) trigger the lane-cost term.
            Accepts a raw int or `Action`; only the integer value matters.
        nxt: ego state after the action and the SUMO step.
        config: tunable weights and thresholds. Pin a `RewardConfig`
            instance once per experiment for reproducibility.

    Returns:
        `(total, components)` where:
        - `total` is the float scalar passed to the RL trainer.
        - `components` is a dict keyed by `COMPONENT_KEYS` with the signed
          contribution of each term. Sums exactly to `total` (no rounding).

    Notes:
        - When `nxt.collided` is True, the env should treat this step as
          terminal. The function still computes other components for logging
          completeness; the collision penalty dominates (-50 vs O(1)).
        - When there's no leader, `nxt.front_thw == math.inf` and the
          unsafe-gap term is 0 (inf > unsafe_thw_s).
        - On the very first env step, callers may set `prev = nxt`, which
          makes the jerk term exactly 0.
    """
    action_int = int(action)

    # Speed: positive for fast cruising. Cap the ratio at 1 for the
    # rare overshoot (aggressive vType has maxSpeed > lane limit).
    speed_ratio = min(nxt.speed / config.v_max, 1.0) if config.v_max > 0 else 0.0
    speed_term = config.w_speed * speed_ratio

    # Lane-change cost: small per-step penalty for any non-STAY action.
    lane_cost_term = config.w_lane_cost * (1.0 if action_int != Action.STAY else 0.0)

    # Jerk: rewards smooth driving. |Δa| normalized by a_max.
    jerk = abs(nxt.accel - prev.accel)
    jerk_term = config.w_jerk * (jerk / config.a_max_norm) if config.a_max_norm > 0 else 0.0

    # Unsafe gap: indicator on time-headway.
    is_unsafe = nxt.front_thw < config.unsafe_thw_s
    unsafe_gap_term = config.w_unsafe_gap * (1.0 if is_unsafe else 0.0)

    # Collision: large terminal penalty.
    collision_term = config.w_collision * (1.0 if nxt.collided else 0.0)

    # Alive: small dense bonus. Anti-suicide; ensures the policy doesn't
    # shortcut to "just collide and end the episode" when other rewards are
    # net-negative for a while.
    alive_term = config.w_alive

    components: dict[str, float] = {
        "speed": speed_term,
        "lane_cost": lane_cost_term,
        "jerk": jerk_term,
        "unsafe_gap": unsafe_gap_term,
        "collision": collision_term,
        "alive": alive_term,
    }
    total = sum(components.values())
    return total, components


def reward_state_at_rest(*, lane_idx: int = 1) -> RewardState:
    """Convenience constructor for a "neutral" RewardState.

    Useful as a `prev` placeholder on the first step of an episode (so the
    jerk term is 0) and in tests. Matches the state of an ego that just
    spawned at zero speed with no leader in front.
    """
    return RewardState(
        speed=0.0,
        accel=0.0,
        front_gap=math.inf,
        front_thw=math.inf,
        collided=False,
        lane_idx=lane_idx,
    )
