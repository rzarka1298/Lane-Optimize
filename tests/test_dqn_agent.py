"""Tests for `DQNAgent` — fast unit tests; CartPole convergence lives in
`tests/test_dqn_smoke.py` (slow gate)."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

import numpy as np
import pytest

from laneiq.agents.base import Policy
from laneiq.agents.dqn.dqn_agent import DEFAULT_DQN_CONFIG, DQNAgent, DQNConfig

# ---------------------------------------------------------------------------
# Construction + Policy protocol
# ---------------------------------------------------------------------------


def test_default_config_is_singleton() -> None:
    assert DQNConfig() == DEFAULT_DQN_CONFIG


def test_dqn_is_a_policy() -> None:
    """DQNAgent structurally satisfies the shared Policy protocol."""
    agent = DQNAgent(obs_dim=4, n_actions=2, seed=0)
    assert isinstance(agent, Policy)
    assert agent.name == "dqn"


def test_constructor_initializes_target_net_to_q_net() -> None:
    """Right after init, target_net == q_net (parameter-equal)."""
    agent = DQNAgent(obs_dim=4, n_actions=2, seed=0)
    for p_q, p_t in zip(
        agent.q_net.parameters(), agent.target_net.parameters(), strict=True,
    ):
        assert (p_q == p_t).all()


def test_target_net_params_have_no_grad() -> None:
    agent = DQNAgent(obs_dim=4, n_actions=2, seed=0)
    for p in agent.target_net.parameters():
        assert not p.requires_grad


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------


def test_act_returns_int_in_range() -> None:
    agent = DQNAgent(obs_dim=4, n_actions=2, seed=0)
    obs = np.zeros(4, dtype=np.float32)
    a = agent.act(obs)
    assert isinstance(a, int)
    assert a in {0, 1}


def test_act_respects_action_mask() -> None:
    """If the only valid action is index 1, greedy must return 1."""
    agent = DQNAgent(obs_dim=4, n_actions=3, seed=0)
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([False, True, False])
    for _ in range(20):
        assert agent.act(obs, action_mask=mask) == 1


def test_epsilon_greedy_at_eps_1_is_random_uniform() -> None:
    """ε=1 → uniform random over valid actions, regardless of Q-values."""
    agent = DQNAgent(obs_dim=4, n_actions=3, seed=42)
    obs = np.zeros(4, dtype=np.float32)
    counts = Counter(agent.act_epsilon_greedy(obs, 1.0) for _ in range(3000))
    for a in (0, 1, 2):
        assert 800 <= counts[a] <= 1200, f"action {a}: count={counts[a]}"


def test_epsilon_greedy_at_eps_0_matches_greedy() -> None:
    """ε=0 → identical to greedy for the same observation."""
    agent = DQNAgent(obs_dim=4, n_actions=3, seed=0)
    obs = np.random.default_rng(0).standard_normal(4).astype(np.float32)
    greedy = agent.act(obs)
    for _ in range(5):
        assert agent.act_epsilon_greedy(obs, 0.0) == greedy


def test_epsilon_greedy_respects_mask_in_random_branch() -> None:
    """When ε=1 (always random) and mask blocks action 1, agent never picks 1."""
    agent = DQNAgent(obs_dim=4, n_actions=3, seed=0)
    obs = np.zeros(4, dtype=np.float32)
    mask = np.array([True, False, True])
    picks = [agent.act_epsilon_greedy(obs, 1.0, action_mask=mask) for _ in range(200)]
    assert 1 not in picks


def test_epsilon_schedule_is_linear() -> None:
    cfg = replace(DEFAULT_DQN_CONFIG, eps_start=1.0, eps_end=0.1, eps_decay_steps=100)
    agent = DQNAgent(obs_dim=4, n_actions=2, config=cfg, seed=0)
    assert agent.epsilon(0) == pytest.approx(1.0)
    assert agent.epsilon(50) == pytest.approx(0.55, abs=1e-6)  # halfway
    assert agent.epsilon(100) == pytest.approx(0.1)
    # Past decay steps, clamp at eps_end.
    assert agent.epsilon(1000) == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# Training step
# ---------------------------------------------------------------------------


def test_train_step_returns_none_when_buffer_too_small() -> None:
    cfg = replace(DEFAULT_DQN_CONFIG, learning_starts=100, batch_size=64)
    agent = DQNAgent(obs_dim=4, n_actions=2, config=cfg, seed=0)
    # Push 50 transitions — below learning_starts threshold.
    for _ in range(50):
        agent.push_transition(
            np.zeros(4, dtype=np.float32), 0, 0.0, np.zeros(4, dtype=np.float32), False,
        )
    assert agent.train_step() is None


def test_train_step_returns_metrics_when_ready() -> None:
    cfg = replace(DEFAULT_DQN_CONFIG, learning_starts=10, batch_size=4)
    agent = DQNAgent(obs_dim=4, n_actions=2, config=cfg, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(20):
        agent.push_transition(
            rng.standard_normal(4).astype(np.float32),
            int(rng.integers(0, 2)),
            float(rng.standard_normal()),
            rng.standard_normal(4).astype(np.float32),
            bool(rng.integers(0, 2)),
        )
    metrics = agent.train_step()
    assert metrics is not None
    for k in ("loss", "mean_q", "mean_target_q", "grad_norm"):
        assert k in metrics
        assert np.isfinite(metrics[k])


def test_train_step_actually_decreases_loss_on_fixed_data() -> None:
    """Sanity: many gradient steps on the same data → loss should decrease."""
    cfg = replace(DEFAULT_DQN_CONFIG, learning_starts=4, batch_size=4, lr=1e-2)
    agent = DQNAgent(obs_dim=4, n_actions=2, config=cfg, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(8):
        agent.push_transition(
            rng.standard_normal(4).astype(np.float32),
            int(rng.integers(0, 2)),
            1.0,
            rng.standard_normal(4).astype(np.float32),
            False,
        )
    losses = [agent.train_step()["loss"] for _ in range(50)]
    # Last 5 average should be << first 5 average.
    assert np.mean(losses[-5:]) < np.mean(losses[:5]) * 0.7


def test_target_sync_happens_on_cadence() -> None:
    cfg = replace(DEFAULT_DQN_CONFIG, target_update_freq=5)
    agent = DQNAgent(obs_dim=4, n_actions=2, config=cfg, seed=0)
    # Force a difference: take a gradient step so q_net diverges from target.
    cfg_small = replace(cfg, learning_starts=1, batch_size=2)
    agent = DQNAgent(obs_dim=4, n_actions=2, config=cfg_small, seed=0)
    rng = np.random.default_rng(0)
    for _ in range(4):
        agent.push_transition(
            rng.standard_normal(4).astype(np.float32),
            int(rng.integers(0, 2)),
            1.0,
            rng.standard_normal(4).astype(np.float32),
            False,
        )
    for _ in range(20):
        agent.train_step()

    # At this point q_net has trained; target_net is still init weights.
    p_q = next(agent.q_net.parameters()).detach().clone()
    p_t = next(agent.target_net.parameters()).detach().clone()
    assert not (p_q == p_t).all()

    # Triggering sync at the right step should re-match them.
    synced = agent.maybe_sync_target(5)
    assert synced is True
    p_t_after = next(agent.target_net.parameters()).detach()
    assert (p_q == p_t_after).all()

    # No-op at non-multiples.
    assert agent.maybe_sync_target(7) is False


# ---------------------------------------------------------------------------
# Save / load
# ---------------------------------------------------------------------------


def test_save_and_load_round_trip(tmp_path) -> None:
    agent = DQNAgent(obs_dim=4, n_actions=2, seed=0)
    obs = np.random.default_rng(0).standard_normal(4).astype(np.float32)
    a_before = agent.act(obs)

    ckpt = tmp_path / "dqn.pt"
    agent.save(ckpt)
    assert ckpt.exists()

    agent2 = DQNAgent(obs_dim=4, n_actions=2, seed=99)  # different init
    agent2.load(ckpt)
    a_after = agent2.act(obs)

    assert a_before == a_after
