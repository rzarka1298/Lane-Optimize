# `laneiq/agents/` — RL agents and rule-based baselines

## Purpose

Everything that *acts in* the env lives here. The shared `Policy` protocol
in `base.py` is what every agent — random, greedy, safe-rule, DQN, PPO,
MAPPO — implements. The `laneiq.eval` harness drives them all through the
same call surface, which is what makes the agent-vs-baseline result table
in `docs/results.md` mechanically possible.

## Status

🚧 **Baselines complete.** The three rule-based agents from the PRD are in
and tested. The from-scratch DQN (Task #12), the SB3 PPO wrapper, and the
multi-agent shared PPO are still pending.

## Files in this folder

```
laneiq/agents/
├── __init__.py             empty (parent invariant)
├── base.py                 Policy Protocol — the shared interface ✅
├── baselines/              Three rule-based agents (PRD baselines)
│   ├── __init__.py         empty
│   ├── random_agent.py     RandomAgent (uniform over valid actions) ✅
│   ├── greedy_agent.py     GreedyAgent (highest projected speed) ✅
│   └── safe_rule_agent.py  SafeRuleAgent (clear-and-faster only) ✅
├── ppo_sb3.py              ⏳ Week 4 — SB3 PPO wrapper
└── dqn/                    ⏳ Tasks #12-13 — from-scratch DQN
```

Per-file docs:

- **`base.md`** — folded into this OVERVIEW (file is 30 lines, single Protocol).
- **[`baselines/OVERVIEW.md`](baselines/OVERVIEW.md)** — folder summary.
- **[`baselines/random_agent.md`](baselines/random_agent.md)** ✅
- **[`baselines/greedy_agent.md`](baselines/greedy_agent.md)** ✅
- **[`baselines/safe_rule_agent.md`](baselines/safe_rule_agent.md)** ✅

## The `Policy` protocol (from `base.py`)

```python
from typing import Protocol, runtime_checkable
import numpy as np

@runtime_checkable
class Policy(Protocol):
    name: str

    def reset(self, *, seed: int | None = None) -> None: ...

    def act(
        self,
        observation: np.ndarray,
        *,
        action_mask: np.ndarray | None = None,
    ) -> int: ...
```

**Why a Protocol, not an ABC.** Each agent already inherits from something
(SB3's `BaseAlgorithm`, PyTorch's `nn.Module`, etc.). Protocols compose
without metaclass conflicts. `runtime_checkable` lets tests do
`isinstance(agent, Policy)`.

**`name`** is consumed by the eval harness as the row label in the results
table and the TensorBoard scalar tag prefix.

**`action_mask` is optional** — if omitted, the policy is expected to
recover the mask from the observation via `valid_actions_mask(obs)`. The
env's `action_masks()` method returns the same thing; passing it in just
saves a re-computation.

## Public API (today)

```python
from laneiq.agents.base import Policy
from laneiq.agents.baselines.random_agent import RandomAgent
from laneiq.agents.baselines.greedy_agent import GreedyAgent
from laneiq.agents.baselines.safe_rule_agent import SafeRuleAgent
```

## Folder-wide invariants

- **All agents implement `Policy`.** Tested via
  `test_baseline_is_a_policy` in `tests/test_baselines.py`. The DQN class
  (Task #12) will need to expose `name` and `act`/`reset` matching this
  signature, even though it also exposes training methods.
- **`act()` must return an action where `mask[action] is True`** when a
  mask is provided. Tested per agent.
- **Agents do NOT touch SUMO directly.** They consume the observation
  (numpy array) and emit an integer. The env owns the TraCI connection.
- **`act()` is pure-ish** — no I/O, no logging at INFO/WARNING level. The
  env loop calls `act()` thousands of times per episode; per-call logging
  would drown TensorBoard.

## Last updated

2026-05-07 · pending commit (baselines + Policy protocol; DQN pending Task #12)
