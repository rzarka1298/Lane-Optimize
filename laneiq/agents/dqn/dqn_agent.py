"""From-scratch DQN agent.

Implements the algorithm in `Project-Documentation/concepts/03-dqn-math.md`:

- ε-greedy action selection with optional action masking.
- Replay-buffer-driven training step with Bellman backup.
- Target network with periodic hard sync.
- Optional Double DQN (`use_double_dqn=True` by default).
- Gradient clipping (DQN losses can spike on early high-TD samples).

Implements the `Policy` protocol from `laneiq.agents.base` so the eval
harness drives DQN identically to the rule-based baselines.

The class is stateful but device-agnostic; tests instantiate with
`device="cpu"` and the trainer picks `mps` / `cuda` / `cpu` based on
availability.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from laneiq.agents.dqn.network import QNetwork
from laneiq.agents.dqn.replay_buffer import ReplayBatch, ReplayBuffer

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DQNConfig:
    """Hyperparameters for `DQNAgent`. Defaults tuned for LaneIQEnv-25-dim;
    overridden per-env for the CartPole gate.

    Mutate via `dataclasses.replace(cfg, lr=...)` for ablations.
    """

    # Architecture
    hidden_dims: tuple[int, ...] = (128, 128)

    # Optimization
    lr: float = 1e-3
    gamma: float = 0.99
    gradient_clip: float = 10.0

    # Replay buffer
    buffer_size: int = 100_000
    batch_size: int = 64
    learning_starts: int = 1_000
    train_freq: int = 4

    # Target network (hard sync)
    target_update_freq: int = 1_000

    # ε-greedy schedule (linear)
    eps_start: float = 1.0
    eps_end: float = 0.05
    eps_decay_steps: int = 50_000

    # Double DQN trick
    use_double_dqn: bool = True


DEFAULT_DQN_CONFIG: DQNConfig = DQNConfig()


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DQNAgent:
    """Vanilla DQN (with optional Double DQN) on a discrete action space.

    Implements the `Policy` protocol — `name`, `reset()`, `act()`. Plus
    training-side methods `train_step()`, `push_transition()`,
    `epsilon(global_step)`, and `save()/load()`.

    Action-mask handling: `act()` accepts an optional `action_mask`. If
    provided, invalid actions are forbidden in both the random-exploration
    branch and the greedy branch. If omitted, the agent acts as if all
    actions are valid (matches Gym's contract on action spaces with no
    masking, e.g. CartPole).
    """

    name: str = "dqn"

    def __init__(
        self,
        obs_dim: int,
        n_actions: int,
        *,
        config: DQNConfig = DEFAULT_DQN_CONFIG,
        device: str | torch.device = "cpu",
        seed: int | None = None,
    ) -> None:
        self._cfg = config
        self._device = torch.device(device)
        self._n_actions = n_actions
        self._obs_dim = obs_dim

        if seed is not None:
            torch.manual_seed(seed)

        self.q_net = QNetwork(obs_dim, n_actions, hidden_dims=config.hidden_dims).to(self._device)
        self.target_net = QNetwork(obs_dim, n_actions, hidden_dims=config.hidden_dims).to(self._device)
        # Initial sync.
        self.target_net.load_state_dict(self.q_net.state_dict())
        # Target net is never trained directly.
        for p in self.target_net.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=config.lr)
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, seed=seed)

        # Action-selection RNG, separate from torch's so torch ops don't
        # advance it (helps reproducibility).
        self._action_rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Policy protocol
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> None:
        """Per-episode hook from the `Policy` protocol.

        Stateful in this implementation only via the action-selection RNG
        (which we re-seed if a seed is provided). Network weights, replay
        buffer, and target net are NOT reset — those persist across
        episodes (training is multi-episode).
        """
        if seed is not None:
            self._action_rng = np.random.default_rng(seed)

    def act(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Greedy action under the current Q-network. Used at eval time.

        For training-time exploration, use `act_epsilon_greedy()` instead.
        """
        return self._greedy_action(observation, action_mask=action_mask)

    # ------------------------------------------------------------------
    # Training-time API
    # ------------------------------------------------------------------

    def epsilon(self, global_step: int) -> float:
        """Linear schedule: `eps_start` → `eps_end` over `eps_decay_steps`."""
        if self._cfg.eps_decay_steps <= 0:
            return self._cfg.eps_end
        frac = min(1.0, global_step / self._cfg.eps_decay_steps)
        return self._cfg.eps_start + frac * (self._cfg.eps_end - self._cfg.eps_start)

    def act_epsilon_greedy(
        self,
        observation: np.ndarray,
        epsilon: float,
        *,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """ε-greedy: with prob `epsilon`, random valid action; else greedy."""
        if self._action_rng.random() < epsilon:
            return self._random_action(action_mask)
        return self._greedy_action(observation, action_mask=action_mask)

    def push_transition(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        """Append one transition to the replay buffer."""
        self.buffer.push(obs, action, reward, next_obs, done)

    def train_step(self) -> dict[str, float] | None:
        """Sample a minibatch and take one gradient step.

        Returns:
            Per-step diagnostics dict (loss, mean_q, mean_target_q, ...)
            or `None` if the buffer is too small (caller should noop).
        """
        if len(self.buffer) < max(self._cfg.batch_size, self._cfg.learning_starts):
            return None

        batch = self.buffer.sample(self._cfg.batch_size)
        return self._gradient_update(batch)

    def maybe_sync_target(self, global_step: int) -> bool:
        """Hard-sync the target net every `target_update_freq` steps.

        Returns:
            True if a sync happened on this call.
        """
        if global_step > 0 and global_step % self._cfg.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())
            return True
        return False

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save Q-net + target-net + optimizer + config to a single `.pt` file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "q_net": self.q_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "config": self._cfg,
                "obs_dim": self._obs_dim,
                "n_actions": self._n_actions,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """Load weights from a checkpoint produced by `save()`."""
        ckpt: dict[str, Any] = torch.load(path, map_location=self._device, weights_only=False)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _greedy_action(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | None,
    ) -> int:
        with torch.no_grad():
            obs_t = torch.from_numpy(observation).to(self._device).float().unsqueeze(0)
            q = self.q_net(obs_t).squeeze(0).cpu().numpy()
        if action_mask is not None:
            # -inf masking; argmax on the masked array.
            masked = np.where(action_mask, q, -np.inf)
            return int(np.argmax(masked))
        return int(np.argmax(q))

    def _random_action(self, action_mask: np.ndarray | None) -> int:
        if action_mask is None:
            return int(self._action_rng.integers(0, self._n_actions))
        valid = np.flatnonzero(action_mask)
        if valid.size == 0:
            return 0  # pathological fallback (STAY for LaneIQ; arbitrary elsewhere)
        return int(self._action_rng.choice(valid))

    def _gradient_update(self, batch: ReplayBatch) -> dict[str, float]:
        device = self._device

        obs = torch.from_numpy(batch.obs).to(device)
        next_obs = torch.from_numpy(batch.next_obs).to(device)
        actions = torch.from_numpy(batch.actions).to(device)
        rewards = torch.from_numpy(batch.rewards).to(device)
        dones = torch.from_numpy(batch.dones).to(device).float()

        # Q(s, a) for the selected action.
        q_all = self.q_net(obs)                                # (B, n_actions)
        q_taken = q_all.gather(1, actions.unsqueeze(1)).squeeze(1)   # (B,)

        # Bootstrap target.
        with torch.no_grad():
            if self._cfg.use_double_dqn:
                # Double DQN: online net picks argmax; target net evaluates.
                online_argmax = self.q_net(next_obs).argmax(dim=1, keepdim=True)
                next_q = self.target_net(next_obs).gather(1, online_argmax).squeeze(1)
            else:
                # Vanilla DQN.
                next_q = self.target_net(next_obs).max(dim=1).values
            target = rewards + (1.0 - dones) * self._cfg.gamma * next_q

        # MSE Bellman loss.
        loss = nn.functional.mse_loss(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.q_net.parameters(), self._cfg.gradient_clip,
        )
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_taken.mean().item()),
            "mean_target_q": float(target.mean().item()),
            "grad_norm": float(grad_norm),
        }


