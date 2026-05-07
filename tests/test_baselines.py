"""Tests for the three rule-based baselines.

Pure-function tests (no SUMO needed) for each agent's logic, plus one
slow integration test per agent that drives a real `LaneIQEnv` for an
episode end-to-end.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from laneiq.agents.base import Policy
from laneiq.agents.baselines.greedy_agent import GreedyAgent
from laneiq.agents.baselines.random_agent import RandomAgent
from laneiq.agents.baselines.safe_rule_agent import SafeRuleAgent
from laneiq.env.actions import Action
from laneiq.env.observations import OBS_DIM

# ---------------------------------------------------------------------------
# Synthetic-observation helpers
# ---------------------------------------------------------------------------

# Layout reminders (see Project-Documentation/laneiq/env/observations.md):
#   obs[0]   = ego_speed / v_max
#   obs[1]   = ego_lane / 2
#   obs[4..6]   = lane0 (front gap, dv, vc)
#   obs[7..9]   = lane0 (rear  gap, dv, vc)
#   obs[10..12] = lane1 (front)
#   obs[13..15] = lane1 (rear)
#   obs[16..18] = lane2 (front)
#   obs[19..21] = lane2 (rear)
#   obs[23] = lane_validity_packed = (left*2 + right) / 3


def _empty_obs(*, ego_lane: int = 1, ego_speed_norm: float = 0.5) -> np.ndarray:
    """Build a 'no neighbors anywhere' observation in the given lane."""
    obs = np.zeros(OBS_DIM, dtype=np.float32)
    obs[0] = ego_speed_norm
    obs[1] = ego_lane / 2.0
    # Fill all 6 neighbor slots with the missing sentinel.
    for lane in (0, 1, 2):
        for slot_idx in range(2):
            offset = 4 + lane * 6 + slot_idx * 3
            obs[offset]     = 1.0   # gap
            obs[offset + 1] = 0.0   # dv
            obs[offset + 2] = -1.0  # vc → "missing"
    # Lane validity packed.
    left_ok = 1 if ego_lane < 2 else 0
    right_ok = 1 if ego_lane > 0 else 0
    obs[23] = (left_ok * 2 + right_ok) / 3.0
    obs[24] = 1.0  # THW = max → safe
    return obs


def _set_neighbor(
    obs: np.ndarray,
    lane: int,
    slot: str,  # "front" or "rear"
    *,
    gap_norm: float,
    dv_norm: float,
    vc_norm: float,
) -> None:
    """Mutate `obs` to place a neighbor in (lane, slot)."""
    slot_idx = 0 if slot == "front" else 1
    offset = 4 + lane * 6 + slot_idx * 3
    obs[offset]     = gap_norm
    obs[offset + 1] = dv_norm
    obs[offset + 2] = vc_norm


# ---------------------------------------------------------------------------
# Policy protocol — every baseline should be a Policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent",
    [RandomAgent(seed=0), GreedyAgent(), SafeRuleAgent()],
    ids=["random", "greedy", "safe_rule"],
)
def test_baseline_is_a_policy(agent) -> None:
    """Each baseline structurally satisfies the Policy protocol."""
    assert isinstance(agent, Policy)
    assert isinstance(agent.name, str) and agent.name


# ---------------------------------------------------------------------------
# RandomAgent
# ---------------------------------------------------------------------------


def test_random_returns_valid_action_in_range() -> None:
    obs = _empty_obs(ego_lane=1)
    agent = RandomAgent(seed=42)
    for _ in range(20):
        a = agent.act(obs)
        assert a in {0, 1, 2}


def test_random_respects_explicit_mask() -> None:
    """If the mask blocks LEFT, the agent must never pick LEFT."""
    obs = _empty_obs(ego_lane=1)
    mask = np.array([True, False, True])
    agent = RandomAgent(seed=42)
    picks = [agent.act(obs, action_mask=mask) for _ in range(50)]
    assert 1 not in picks
    assert set(picks) <= {0, 2}


def test_random_at_lane_0_never_picks_change_right() -> None:
    """Mask derived from obs[23] should prevent CHANGE_RIGHT in lane 0."""
    obs = _empty_obs(ego_lane=0)
    agent = RandomAgent(seed=42)
    picks = [agent.act(obs) for _ in range(50)]
    assert 2 not in picks  # CHANGE_RIGHT
    assert set(picks) <= {0, 1}


def test_random_distribution_roughly_uniform() -> None:
    """Over many samples the distribution should be ~uniform across valid actions."""
    obs = _empty_obs(ego_lane=1)
    agent = RandomAgent(seed=0)
    counts = Counter(agent.act(obs) for _ in range(3000))
    for a in (0, 1, 2):
        # Expect ~1000 each ± noise; very loose bound.
        assert 800 <= counts[a] <= 1200, f"action {a}: count={counts[a]}"


def test_random_seeding_reproducible() -> None:
    obs = _empty_obs(ego_lane=1)
    a1 = RandomAgent(seed=123)
    a2 = RandomAgent(seed=123)
    seq1 = [a1.act(obs) for _ in range(20)]
    seq2 = [a2.act(obs) for _ in range(20)]
    assert seq1 == seq2


def test_random_reset_with_seed_reproduces_sequence() -> None:
    obs = _empty_obs(ego_lane=1)
    a = RandomAgent(seed=99)
    seq1 = [a.act(obs) for _ in range(10)]
    a.reset(seed=99)
    seq2 = [a.act(obs) for _ in range(10)]
    assert seq1 == seq2


# ---------------------------------------------------------------------------
# GreedyAgent
# ---------------------------------------------------------------------------


def test_greedy_stays_when_all_lanes_open() -> None:
    """All lanes empty → all projected speeds equal → STAY (tie-break)."""
    obs = _empty_obs(ego_lane=1)
    agent = GreedyAgent()
    assert agent.act(obs) == int(Action.STAY)


def test_greedy_picks_left_when_left_lane_open_and_current_blocked() -> None:
    """Current lane has slow leader; left is open → CHANGE_LEFT."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.7)
    # Slow leader 30m ahead in current (lane 1), 5 m/s slower.
    _set_neighbor(obs, 1, "front", gap_norm=0.3, dv_norm=-0.14, vc_norm=0.5)  # slow vc
    # Left (lane 2) stays as missing → projected = 1.0.
    assert agent_choice(obs) == int(Action.CHANGE_LEFT)


def test_greedy_picks_right_when_only_right_is_better() -> None:
    """Left blocked by slow leader; current also blocked; right open."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.7)
    # Both current and left have slow leaders; right is open.
    _set_neighbor(obs, 1, "front", gap_norm=0.3, dv_norm=-0.14, vc_norm=0.5)
    _set_neighbor(obs, 2, "front", gap_norm=0.3, dv_norm=-0.14, vc_norm=0.5)
    assert agent_choice(obs) == int(Action.CHANGE_RIGHT)


def test_greedy_respects_mask_blocking_left() -> None:
    """Even if left looks better, an explicit mask=False forces a different choice."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.7)
    _set_neighbor(obs, 1, "front", gap_norm=0.3, dv_norm=-0.14, vc_norm=0.5)
    # Mask blocks left.
    mask = np.array([True, False, True])
    a = GreedyAgent().act(obs, action_mask=mask)
    assert a != int(Action.CHANGE_LEFT)


def test_greedy_does_not_change_when_current_lane_is_open() -> None:
    """Open current lane (no leader) → projected = 1.0 = max → STAY."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.5)
    # All lanes open already; no setup needed.
    assert GreedyAgent().act(obs) == int(Action.STAY)


def test_greedy_validates_obs_shape() -> None:
    with pytest.raises(ValueError, match="expected obs shape"):
        GreedyAgent().act(np.zeros(10, dtype=np.float32))


# Helper: shorthand for the parametrize tests above.
def agent_choice(obs: np.ndarray) -> int:
    return GreedyAgent().act(obs)


# ---------------------------------------------------------------------------
# SafeRuleAgent
# ---------------------------------------------------------------------------


def test_safe_stays_when_no_clear_faster_option() -> None:
    """All lanes equally open → no lane is *strictly faster* → STAY."""
    obs = _empty_obs(ego_lane=1)
    assert SafeRuleAgent().act(obs) == int(Action.STAY)


def test_safe_changes_left_when_left_clear_and_faster() -> None:
    """Slow leader in current; left is wide open → CHANGE_LEFT."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.5)
    _set_neighbor(obs, 1, "front", gap_norm=0.5, dv_norm=-0.3, vc_norm=0.5)
    # Lane 2 (left) is open by default.
    assert SafeRuleAgent().act(obs) == int(Action.CHANGE_LEFT)


def test_safe_does_not_change_left_when_left_front_too_close() -> None:
    """When left's front gap is below threshold, the agent must NOT pick LEFT.

    (The agent may still go right or stay, depending on right-lane state.)
    """
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.5)
    _set_neighbor(obs, 1, "front", gap_norm=0.5, dv_norm=-0.3, vc_norm=0.5)
    # Left has a faster leader but only 10m ahead → not clear.
    _set_neighbor(obs, 2, "front", gap_norm=0.10, dv_norm=+0.2, vc_norm=-0.5)
    assert SafeRuleAgent().act(obs) != int(Action.CHANGE_LEFT)


def test_safe_does_not_change_left_when_left_rear_too_close() -> None:
    """When left's rear gap is below threshold, the agent must NOT pick LEFT."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.5)
    _set_neighbor(obs, 1, "front", gap_norm=0.5, dv_norm=-0.3, vc_norm=0.5)
    # Left front open but rear has tailgater 10m back.
    _set_neighbor(obs, 2, "rear", gap_norm=0.10, dv_norm=+0.1, vc_norm=-0.5)
    assert SafeRuleAgent().act(obs) != int(Action.CHANGE_LEFT)


def test_safe_stays_when_no_target_clear_and_faster() -> None:
    """All neighbors blocking → no lane change improves things → STAY."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.5)
    # Current lane: slow leader.
    _set_neighbor(obs, 1, "front", gap_norm=0.5, dv_norm=-0.3, vc_norm=0.5)
    # Left blocked (close front).
    _set_neighbor(obs, 2, "front", gap_norm=0.10, dv_norm=+0.2, vc_norm=-0.5)
    # Right also has a slow leader (not faster than current's projected speed).
    _set_neighbor(obs, 0, "front", gap_norm=0.5, dv_norm=-0.4, vc_norm=0.5)
    assert SafeRuleAgent().act(obs) == int(Action.STAY)


def test_safe_picks_right_when_only_right_is_safe_and_faster() -> None:
    """Left is blocked, right is open and faster → CHANGE_RIGHT."""
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.5)
    _set_neighbor(obs, 1, "front", gap_norm=0.5, dv_norm=-0.3, vc_norm=0.5)
    # Left blocked (close front).
    _set_neighbor(obs, 2, "front", gap_norm=0.05, dv_norm=+0.2, vc_norm=-0.5)
    # Right is open by default in _empty_obs; explicitly leave it that way.
    assert SafeRuleAgent().act(obs) == int(Action.CHANGE_RIGHT)


def test_safe_respects_mask_blocking_left() -> None:
    obs = _empty_obs(ego_lane=1, ego_speed_norm=0.5)
    _set_neighbor(obs, 1, "front", gap_norm=0.5, dv_norm=-0.3, vc_norm=0.5)
    mask = np.array([True, False, True])
    assert SafeRuleAgent().act(obs, action_mask=mask) != int(Action.CHANGE_LEFT)


def test_safe_validates_obs_shape() -> None:
    with pytest.raises(ValueError, match="expected obs shape"):
        SafeRuleAgent().act(np.zeros(10, dtype=np.float32))


# ---------------------------------------------------------------------------
# Slow integration tests — each baseline drives a real LaneIQEnv
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_factory",
    [
        pytest.param(lambda: RandomAgent(seed=0), id="random"),
        pytest.param(GreedyAgent, id="greedy"),
        pytest.param(SafeRuleAgent, id="safe_rule"),
    ],
)
@pytest.mark.slow
@pytest.mark.sumo
def test_baseline_runs_one_episode(agent_factory) -> None:
    """Each baseline must drive at least one episode without crashing the env."""
    from laneiq.env.highway_env import LaneIQEnv

    env = LaneIQEnv(density="low", max_steps=30, warmup_steps=20, seed=42)
    agent = agent_factory()
    try:
        obs, _ = env.reset()
        agent.reset(seed=42)

        terminated = truncated = False
        steps = 0
        while not (terminated or truncated):
            mask = env.action_masks()
            action = agent.act(obs, action_mask=mask)
            assert mask[action], f"agent {agent.name} picked invalid action {action}"
            obs, reward, terminated, truncated, _info = env.step(action)
            assert env.observation_space.contains(obs)
            assert np.isfinite(reward)
            steps += 1
            if steps > 100:
                pytest.fail(f"agent {agent.name}: episode never ended")
    finally:
        env.close()
