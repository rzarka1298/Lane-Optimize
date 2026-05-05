# `laneiq/env/` — Gymnasium environment

## Purpose

Holds everything that turns SUMO into a Gymnasium-compatible RL environment:
the TraCI lifecycle, the observation builder, the action applier, the reward
composer, the episode controller, and (later) the multi-agent PettingZoo
wrapper. Anything that *is the environment* lives here; anything that *acts
in the environment* lives in `laneiq/agents/`.

## Status

✅ **Single-agent env complete.** All five files (`sumo_runtime`,
`observations`, `actions`, `rewards`, `highway_env`) are in and tested.
`LaneIQEnv` passes `gymnasium.utils.env_checker.check_env`. The PettingZoo
multi-agent wrapper lands in Week 5.

## Files in this folder

Per-file docs live alongside this OVERVIEW.md:

- **[`sumo_runtime.md`](sumo_runtime.md)** — Process lifecycle: `build_argv`,
  `sumo_session`, `sumo_binary`. Single seam for libsumo swap. ✅
- **[`observations.md`](observations.md)** — Fixed-length 25-dim obs vector;
  `build_observation` + `ObservationBuilder`. ✅
- **[`actions.md`](actions.md)** — `Action` enum, `apply_action`,
  `valid_actions_mask`, `disable_sumo_lane_change`. ✅
- **[`rewards.md`](rewards.md)** — `RewardConfig`, `RewardState`,
  `compute_reward` (returns scalar + per-component dict). ✅
- **[`highway_env.md`](highway_env.md)** — `LaneIQEnv(gymnasium.Env)` —
  the integrated env. Passes `env_checker`. ✅
- **`multi_agent_env.md`** — `LaneIQParallelEnv`. ⏳ Week 5.

## Public API (today)

```python
# SUMO process lifecycle
from laneiq.env.sumo_runtime import (
    DEFAULT_SCENARIO_DIR, DEFAULT_SUMOCFG,
    sumo_binary, build_argv, sumo_session,
)

# Observation building
from laneiq.env.observations import (
    OBS_DIM, OBS_NAMES, OBSERVATION_SPEC,
    DEFAULT_OBSERVATION_CONFIG,
    ObservationConfig, ObservationSpec,
    build_observation, ego_present, ObservationBuilder,
)

# Discrete actions + masking
from laneiq.env.actions import (
    Action, N_ACTIONS, ACTION_SPACE, ACTION_NAMES,
    disable_sumo_lane_change,
    apply_action, valid_actions_mask, mask_invalid_q_values,
)

# Reward composition
from laneiq.env.rewards import (
    RewardConfig, DEFAULT_REWARD_CONFIG, RewardState,
    COMPONENT_KEYS, compute_reward, reward_state_at_rest,
)

# The integrated env
from laneiq.env.highway_env import LaneIQEnv
```

Full schemas and examples in the per-file docs:
[`sumo_runtime.md`](sumo_runtime.md), [`observations.md`](observations.md),
[`actions.md`](actions.md), [`rewards.md`](rewards.md),
[`highway_env.md`](highway_env.md).

## Folder-wide invariants

- **TraCI only — `libsumo` is intentionally absent** on macOS arm64. If
  Week 3 profiling justifies it, the swap point is *just* `sumo_runtime.py`.
  See `Project-Documentation/setup.md` for the decision and recovery path.
- **TraCI is single-connection per label.** Default label `"laneiq"`.
  Multi-agent training (Week 5) keeps using ONE connection — multiple egos
  in one SUMO process.
- **Heavy imports (`traci`, `sumolib`, `torch`) inside this folder are OK at
  module top-level**, but `laneiq/__init__.py` and `laneiq/env/__init__.py`
  must stay empty so plain `import laneiq` doesn't transitively load these.
  See parent invariant in `../OVERVIEW.md`.

## Last updated

2026-05-05 · `7c6a6e9` (highway_env.py landed; env_checker passes);
prior `ee53ff5` (rewards); `b4d7fd1` (actions); `0477f0d` (observations);
`1f423f2` (sumo_runtime); doc split at `d9d8601`
