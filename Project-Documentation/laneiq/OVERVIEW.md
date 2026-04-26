# `laneiq/` — main Python package

## Purpose

The `laneiq` Python package contains everything that is *not* a SUMO scenario,
a backend service, or a frontend app. It owns the Gym env, the from-scratch
DQN, the SB3 PPO wrapper, the rule-based baselines, the eval harness, and the
multi-agent training code.

It is published as `laneiq` in `pyproject.toml` (Hatchling backend), so all
imports look like `from laneiq.env import LaneIQEnv` — clean read for recruiters.

## Status

**Scaffold only.** Every subpackage exists with an empty `__init__.py`. No
runtime code has been written yet.

## Key files

```
laneiq/
├── __init__.py          # exposes __version__ = "0.1.0"
├── env/__init__.py      # (empty)  ← LaneIQEnv lands here in Week 2
├── agents/
│   ├── __init__.py      # (empty)
│   ├── dqn/__init__.py  # (empty)  ← from-scratch DQN, Week 2-3
│   └── baselines/__init__.py  # (empty)  ← random/greedy/safe_rule, Week 2
├── eval/__init__.py     # (empty)
├── viz/__init__.py      # (empty)
├── multi_agent/__init__.py  # (empty)
└── utils/__init__.py    # (empty)
```

## Public API

Today: `laneiq.__version__` only.

Eventual public surface (planned):
- `laneiq.env.LaneIQEnv` — single-agent Gymnasium env.
- `laneiq.env.LaneIQParallelEnv` — multi-agent PettingZoo env.
- `laneiq.agents.dqn.DQNAgent` — from-scratch DQN.
- `laneiq.agents.baselines.{RandomAgent, GreedyAgent, SafeRuleAgent}`.
- `laneiq.eval.run_eval_matrix(...)` — agents × densities × seeds → CSV.

## Invariants & gotchas

- **Subpackage `__init__.py` files are intentionally empty.** Don't dump
  re-exports there — keep the import surface narrow and explicit. The user's
  IDE imports should auto-suggest concrete classes, not anything from
  `laneiq.*` directly.
- **Python 3.11 only.** `requires-python = ">=3.11,<3.12"` is pinned in
  `pyproject.toml` because libsumo wheels lag a release behind Python; do not
  bump to 3.12 without verifying `libsumo` ships an arm64 wheel.
- **No top-level `import torch` or `import libsumo` at package import time.**
  They're heavy and create import-time errors when SUMO/torch aren't
  installed (e.g., during a fresh `git clone` before `uv sync`). Lazy-import
  them inside functions/methods that need them.

## Last updated

2026-04-26 · `be30837` (scaffold)
