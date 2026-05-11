"""PPO via Stable-Baselines3's MaskablePPO (sb3-contrib).

PPO is a policy-gradient method — fundamentally different family from
DQN (value iteration). We use the library implementation (not from-scratch)
because the "from-scratch" interview story already lives in the DQN code;
PPO's role here is to validate that a different RL family hits the same
or higher performance, supporting the "RL learns the Pareto frontier"
narrative.

`MaskablePPO` (from `sb3-contrib`) extends standard PPO with action
masking — invalid actions get zero probability before sampling. This
matches the contract our `LaneIQEnv.action_masks()` was built for.

`PPOAgent` wraps the SB3 model so the eval harness drives it identically
to DQN and the rule-based baselines (Policy protocol).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sb3_contrib import MaskablePPO

from laneiq.env.actions import valid_actions_mask
from laneiq.env.observations import OBS_DIM


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for `PPOAgent`. SB3's defaults are well-tuned;
    these mostly mirror them with one or two task-specific changes.

    Mutate via `dataclasses.replace(cfg, lr=...)` for ablations.
    """

    # Architecture (match DQN for fair comparison).
    hidden_dims: tuple[int, ...] = (128, 128)

    # Optimization
    lr: float = 3e-4                 # SB3 default; PPO is sensitive — don't go higher
    gamma: float = 0.99
    n_steps: int = 2_048             # rollout buffer size before each update
    batch_size: int = 64
    n_epochs: int = 10               # gradient passes over each rollout
    gae_lambda: float = 0.95         # GAE smoothing
    clip_range: float = 0.2          # PPO clipping epsilon
    ent_coef: float = 0.01           # entropy bonus encourages exploration
    vf_coef: float = 0.5             # value-loss weight
    max_grad_norm: float = 0.5       # tighter than DQN; PPO is stable enough

    # Logging
    verbose: int = 1                  # SB3 print level


DEFAULT_PPO_CONFIG: PPOConfig = PPOConfig()


class PPOAgent:
    """Wraps `sb3_contrib.MaskablePPO` to implement the Policy protocol.

    Construction does NOT train; call `learn(env, total_timesteps)` to
    train, then `act(obs, action_mask)` for eval-time inference. The
    underlying SB3 model is exposed as `self.model` for advanced use
    (e.g., raw `predict()` with custom flags).
    """

    name: str = "ppo"

    def __init__(
        self,
        env=None,                                            # noqa: ANN001 — gym.Env or vec env
        *,
        config: PPOConfig = DEFAULT_PPO_CONFIG,
        device: str | torch.device = "cpu",
        seed: int | None = None,
        tensorboard_log: str | Path | None = None,
    ) -> None:
        """Construct the underlying MaskablePPO model.

        Args:
            env: a Gymnasium env exposing `action_masks()`. Required for
                training; can be a dummy env for eval-time-only use as long
                as observation/action spaces match.
            config: PPO hyperparameters.
            device: 'cpu' / 'mps' / 'cuda'. CPU recommended for our small net.
            seed: torch + SB3 seed for reproducibility.
            tensorboard_log: optional path for SB3's TensorBoard scalars.
        """
        self._cfg = config

        if env is None:
            # Build a placeholder env so MaskablePPO can read the spaces.
            # Only used in eval-time-only flows that load a checkpoint.
            from laneiq.env.highway_env import LaneIQEnv

            env = LaneIQEnv(density="medium", max_steps=10, warmup_steps=5)

        # SB3 builds the actor-critic MLP from net_arch.
        policy_kwargs = {
            "net_arch": list(config.hidden_dims),
        }

        self.model = MaskablePPO(
            "MlpPolicy",
            env,
            learning_rate=config.lr,
            gamma=config.gamma,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            verbose=config.verbose,
            seed=seed,
            device=str(device),
            policy_kwargs=policy_kwargs,
            tensorboard_log=str(tensorboard_log) if tensorboard_log else None,
        )

    # ------------------------------------------------------------------
    # Policy protocol — eval-time
    # ------------------------------------------------------------------

    def reset(self, *, seed: int | None = None) -> None:  # noqa: ARG002
        """No-op — SB3 PPO is stateless across episodes."""
        return

    def act(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | None = None,
    ) -> int:
        """Greedy action under the current PPO policy."""
        if action_mask is None:
            action_mask = valid_actions_mask(observation)
        # SB3's predict() returns (action, state). We use deterministic=True
        # for eval; the action_masks kwarg ensures invalid actions stay
        # zero-prob even when stochastic.
        action, _state = self.model.predict(
            observation,
            action_masks=action_mask,
            deterministic=True,
        )
        return int(action)

    # ------------------------------------------------------------------
    # Training-time
    # ------------------------------------------------------------------

    def learn(
        self,
        total_timesteps: int,
        *,
        tb_log_name: str = "MaskablePPO",
        progress_bar: bool = False,
    ) -> None:
        """Train for `total_timesteps`. SB3 handles env stepping + the loop."""
        self.model.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            progress_bar=progress_bar,
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save to a .zip checkpoint (SB3's native format)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)

    def load(self, path: str | Path, env=None) -> None:  # noqa: ANN001
        """Load weights from a checkpoint produced by `save()`."""
        self.model = MaskablePPO.load(path, env=env, device=self.model.device)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        device: str | torch.device = "cpu",
        env=None,                                            # noqa: ANN001
    ) -> "PPOAgent":
        """Eval-only constructor: build agent from a saved checkpoint.

        No env required for the act() path because MaskablePPO stores the
        observation/action spaces. Pass `env` only if you plan to continue
        training (model.learn requires it).
        """
        # Construct a placeholder, then replace the model entirely.
        agent = cls.__new__(cls)
        agent._cfg = DEFAULT_PPO_CONFIG
        agent.model = MaskablePPO.load(path, env=env, device=str(device))
        return agent
