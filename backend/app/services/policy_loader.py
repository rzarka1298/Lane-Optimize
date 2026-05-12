"""Discover + load checkpoints under `checkpoints/final/`.

Backend startup scans the folder; `GET /runs` enumerates what's
available; `POST /sim/start` resolves the requested agent name to a
policy instance.

Naming convention (case-insensitive):
  - filename starts with 'dqn_'  → algorithm = "dqn", load via DQNAgent
  - filename starts with 'ppo_'  → algorithm = "ppo", load via PPOAgent
  - special names {'random', 'greedy', 'safe_rule'} → builtin baselines

If the conventions ever shift, swap the prefix table here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from backend.app.schemas import RunListEntry
from laneiq.agents.base import Policy
from laneiq.agents.baselines.greedy_agent import GreedyAgent
from laneiq.agents.baselines.random_agent import RandomAgent
from laneiq.agents.baselines.safe_rule_agent import SafeRuleAgent

# Repo root → resolves relative to this file.
_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
_CHECKPOINT_DIR: Path = _REPO_ROOT / "checkpoints" / "final"

_BASELINE_FACTORIES: dict[str, callable] = {
    "random": lambda: RandomAgent(seed=0),
    "greedy": GreedyAgent,
    "safe_rule": SafeRuleAgent,
}

AlgoT = Literal["dqn", "ppo"]


def _detect_algo(path: Path) -> AlgoT | None:
    name = path.name.lower()
    if name.startswith("dqn_") or name.endswith(".pt"):
        return "dqn"
    if name.startswith("ppo_") or name.endswith(".zip"):
        return "ppo"
    return None


def list_runs() -> list[RunListEntry]:
    """Discover checkpoints under `checkpoints/final/`."""
    out: list[RunListEntry] = []
    if not _CHECKPOINT_DIR.is_dir():
        return out

    for p in sorted(_CHECKPOINT_DIR.iterdir()):
        if p.name == ".gitkeep" or p.name.startswith("."):
            continue
        if not p.is_file():
            continue
        algo = _detect_algo(p)
        if algo is None:
            continue
        out.append(RunListEntry(
            name=p.stem,
            algorithm=algo,
            path=str(p),
            size_bytes=p.stat().st_size,
        ))
    return out


def load_baseline(name: str) -> Policy:
    """Construct one of the builtin rule-based baselines."""
    if name not in _BASELINE_FACTORIES:
        raise ValueError(
            f"unknown baseline {name!r}; valid: {sorted(_BASELINE_FACTORIES)}",
        )
    return _BASELINE_FACTORIES[name]()


def load_checkpoint(name: str) -> Policy:
    """Resolve `name` to a checkpoint and load the matching agent.

    `name` matches `RunListEntry.name` (the file stem). If multiple files
    share the same stem (.pt and .zip), the algorithm is inferred from
    the file extension.

    Raises:
        FileNotFoundError: if no matching checkpoint exists.
        ValueError: if the file can't be classified as DQN or PPO.
    """
    matches = list(_CHECKPOINT_DIR.glob(f"{name}.*"))
    matches = [m for m in matches if m.suffix in (".pt", ".zip")]
    if not matches:
        raise FileNotFoundError(
            f"no checkpoint matching {name!r} under {_CHECKPOINT_DIR}",
        )
    if len(matches) > 1:
        # Prefer .zip (PPO) if both exist — shouldn't happen in practice.
        matches = sorted(matches, key=lambda p: (p.suffix != ".zip"))
    ckpt = matches[0]
    algo = _detect_algo(ckpt)

    if algo == "ppo":
        from laneiq.agents.ppo_sb3 import PPOAgent
        return PPOAgent.from_checkpoint(ckpt, device="cpu")
    if algo == "dqn":
        from laneiq.agents.dqn.dqn_agent import DQNAgent
        from laneiq.env.observations import OBS_DIM
        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=3, device="cpu", seed=0)
        agent.load(ckpt)
        return agent
    raise ValueError(f"cannot classify checkpoint {ckpt}")


def resolve(name: str) -> Policy:
    """Resolve `name` to a Policy. Tries baselines first, then checkpoints.

    Used by `POST /sim/start` — frontend can pass either 'random' (baseline)
    or 'ppo_v1_mixed_500k' (checkpoint) and we figure it out.
    """
    if name in _BASELINE_FACTORIES:
        return load_baseline(name)
    return load_checkpoint(name)
