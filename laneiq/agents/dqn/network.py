"""Q-network for DQN.

Two-hidden-layer MLP, ReLU activations, no output activation (raw Q-values).
Tiny by neural-net standards (~17k parameters) — the whole point is that
the architecture isn't the interesting variable. The interesting variable
is the *algorithm* (replay, target net, ε-greedy) — see
`Project-Documentation/concepts/03-dqn-math.md`.

Why a separate file: testing and reasoning are easier when the model
is decoupled from the training loop. `dqn_agent.py` can swap in any
`nn.Module` with the right input/output shape.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """MLP mapping observation → Q-values per discrete action.

    Default architecture: `[obs_dim → 128 → 128 → n_actions]`. ReLU
    activations on the two hidden layers; no activation on the output
    (Q-values are unbounded reals).

    Weight init: PyTorch default (Kaiming uniform for Linear). Adequate
    for an MLP of this size; exotic init schemes don't help here.
    """

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU(inplace=True))
            in_dim = h
        layers.append(nn.Linear(in_dim, n_actions))
        self.net = nn.Sequential(*layers)
        self._obs_dim = obs_dim
        self._n_actions = n_actions

    @property
    def obs_dim(self) -> int:
        return self._obs_dim

    @property
    def n_actions(self) -> int:
        return self._n_actions

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            obs: shape `(batch, obs_dim)` float tensor.

        Returns:
            shape `(batch, n_actions)` Q-values.
        """
        return self.net(obs)
