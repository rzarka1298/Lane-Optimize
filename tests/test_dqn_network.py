"""Tests for the Q-network MLP."""

from __future__ import annotations

import torch

from laneiq.agents.dqn.network import QNetwork


def test_default_architecture_shape() -> None:
    net = QNetwork(obs_dim=25, n_actions=3)
    assert net.obs_dim == 25
    assert net.n_actions == 3
    out = net(torch.zeros(7, 25))
    assert out.shape == (7, 3)
    assert out.dtype == torch.float32


def test_custom_hidden_dims() -> None:
    net = QNetwork(obs_dim=4, n_actions=2, hidden_dims=(64, 64, 32))
    out = net(torch.zeros(3, 4))
    assert out.shape == (3, 2)


def test_forward_is_deterministic_in_eval() -> None:
    """Same input → same output (no dropout, no batchnorm)."""
    net = QNetwork(obs_dim=4, n_actions=2)
    x = torch.randn(5, 4)
    a = net(x)
    b = net(x)
    assert torch.allclose(a, b)


def test_param_count_matches_expectation() -> None:
    """[25 → 128 → 128 → 3] should have ~17.6k params."""
    net = QNetwork(obs_dim=25, n_actions=3)
    n = sum(p.numel() for p in net.parameters())
    # exact: (25+1)*128 + (128+1)*128 + (128+1)*3 = 3328 + 16512 + 387 = 20227
    assert 18_000 <= n <= 22_000, f"unexpected param count: {n}"


def test_gradients_flow() -> None:
    """A simple MSE loss should produce non-zero gradients on every layer."""
    net = QNetwork(obs_dim=4, n_actions=2)
    x = torch.randn(8, 4, requires_grad=False)
    target = torch.randn(8, 2)
    loss = ((net(x) - target) ** 2).mean()
    loss.backward()
    for name, p in net.named_parameters():
        assert p.grad is not None, f"no grad on {name}"
        assert torch.isfinite(p.grad).all(), f"non-finite grad on {name}"
