"""Shared agent interface.

Every policy in this project — `RandomAgent`, `GreedyAgent`, `SafeRuleAgent`,
the from-scratch DQN, the SB3 PPO wrapper, the multi-agent shared PPO — implements
the `Policy` protocol below. The eval harness in `laneiq.eval` drives them all
through the same call surface, which is what makes the agents-vs-baselines
result table possible.

Why a `Protocol` (structural typing) instead of an ABC:

- Each agent already inherits from something (PyTorch `nn.Module`, SB3's
  `BaseAlgorithm`, etc.). Protocols compose without metaclass conflicts.
- Tests can `isinstance(agent, Policy)` because the protocol is
  `runtime_checkable`.
- Conformance is mechanical: an agent with the right method signatures IS a
  `Policy`, no inheritance gymnastics required.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Policy(Protocol):
    """Minimum contract every agent must satisfy.

    Implementations must expose a `name: str` attribute (used in eval-result
    tables and TensorBoard tags) and the two methods below.
    """

    name: str

    def reset(self, *, seed: int | None = None) -> None:
        """Clear any per-episode state. Called at every `env.reset()`.

        Stateless agents (greedy, safe-rule) implement this as a no-op.
        Stateful agents (random with an RNG, DQN with an exploration counter)
        use it to re-seed or reset.
        """
        ...

    def act(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Pick an action given the current observation.

        Args:
            observation: shape-(25,) float32 array from `build_observation`.
            action_mask: optional shape-(3,) bool array. If provided, the
                returned action MUST satisfy `mask[action] is True`. If
                omitted, the policy may recover the mask from the observation
                via `valid_actions_mask(observation)`.

        Returns:
            Integer action ∈ {0, 1, 2} = {STAY, CHANGE_LEFT, CHANGE_RIGHT}.
        """
        ...
