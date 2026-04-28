# `laneiq/env/` — Gymnasium environment

## Purpose

Holds everything that turns SUMO into a Gymnasium-compatible RL environment:
the TraCI lifecycle, the observation builder, the action applier, the reward
composer, the episode controller, and (later) the multi-agent PettingZoo
wrapper. Anything that *is the environment* lives here; anything that *acts
in the environment* lives in `laneiq/agents/`.

## Status

🚧 **Foundation laid.** `sumo_runtime.py` is in and tested. Next files due in
Week 2: `observations.py`, `actions.py`, `rewards.py`, `highway_env.py`. The
PettingZoo wrapper lands in Week 5.

## Files in this folder

Per-file docs live alongside this OVERVIEW.md:

- **[`sumo_runtime.md`](sumo_runtime.md)** — Process lifecycle: `build_argv`,
  `sumo_session`, `sumo_binary`. Single seam for libsumo swap. ✅
- **`observations.md`** — Fixed-length 24-dim obs vector. ⏳ Task #7.
- **`actions.md`** — Action enum + TraCI applier. ⏳ Task #8.
- **`rewards.md`** — Reward composer with config. ⏳ Task #9.
- **`highway_env.md`** — `LaneIQEnv(gymnasium.Env)`. ⏳ Task #10.
- **`spawning.md`** — Ego respawn logic (with `highway_env.py`). ⏳ Task #10.
- **`multi_agent_env.md`** — `LaneIQParallelEnv`. ⏳ Week 5.

## Public API (today)

```python
from laneiq.env.sumo_runtime import (
    DEFAULT_SCENARIO_DIR,    # Path to sumo_scenarios/highway_3lane/
    DEFAULT_SUMOCFG,         # Path to highway.sumocfg
    sumo_binary,             # (gui: bool) -> str
    build_argv,              # construct argv for traci.start
    sumo_session,            # context manager
)
```

`sumo_session()` is for tests and ad-hoc scripts. The Gym env (landing in
Week 2) will use `build_argv()` directly because it manages its TraCI
connection lifetime across many `reset()` calls inside its own class.

Full schema and examples in [`sumo_runtime.md`](sumo_runtime.md).

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

2026-04-26 · `d9d8601` (per-file doc split; sumo_runtime.py at `1f423f2`)
