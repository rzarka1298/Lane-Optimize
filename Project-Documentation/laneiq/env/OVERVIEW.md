# `laneiq/env/` — Gymnasium environment

## Purpose

Holds everything that turns SUMO into a Gymnasium-compatible RL environment:
the TraCI lifecycle, the observation builder, the action applier, the reward
composer, the episode controller, and (later) the multi-agent PettingZoo
wrapper. Anything that *is the environment* lives here; anything that *acts in
the environment* lives in `laneiq/agents/`.

## Status

🚧 **Foundation laid.** `sumo_runtime.py` is in and tested (process lifecycle
+ argv builder + context manager). Next files due in Week 2: `observations.py`,
`actions.py`, `rewards.py`, `highway_env.py`. The PettingZoo wrapper lands in
Week 5.

## Key files

```
laneiq/env/
├── __init__.py             (empty — see invariants in laneiq/OVERVIEW.md)
├── sumo_runtime.py         ✅ Process lifecycle: build_argv, sumo_session,
│                              sumo_binary. Single seam for libsumo swap.
├── observations.py         ⏳ Fixed-length 24-dim obs vector (Task #7)
├── actions.py              ⏳ Action enum + TraCI applier (Task #8)
├── rewards.py              ⏳ Reward composer (Task #9)
├── highway_env.py          ⏳ LaneIQEnv (Task #10)
├── spawning.py             ⏳ Ego respawn logic (with highway_env)
└── multi_agent_env.py      ⏳ Week 5
```

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

`sumo_session()` is for tests and ad-hoc scripts. The Gym env (`highway_env.py`,
landing next) will use `build_argv()` directly because it manages its TraCI
connection lifetime across many `reset()` calls inside its own class.

## Invariants & gotchas

- **TraCI only — `libsumo` is intentionally absent.** No macOS arm64 wheel as
  of 1.26. If Week 3 profiling justifies it, the swap point is *just*
  `sumo_runtime.py` (change `import traci` → conditional `import libsumo as traci`
  with the same call surface). See `Project-Documentation/setup.md`.
- **Locked CLI flags in `_LOCKED_FLAGS`.** `--collision.action=warn`,
  `--collision.mingap-factor=0`, `--time-to-teleport=-1`, `--step-length=0.1`.
  These are load-bearing for RL. Don't change one without understanding all of
  them — `Project-Documentation/sumo_scenarios/OVERVIEW.md` explains why each.
- **`sumo_session()` is for tests/scripts; do NOT wrap a `LaneIQEnv` in it.**
  The env owns its own TraCI connection across episodes. Wrapping creates
  double start/close and races SUMO process teardown.
- **TraCI is single-connection per label.** Default label `"laneiq"`.
  Multi-agent training (Week 5) keeps using ONE connection — multiple egos
  control multiple vehicles in the same SUMO process.
- **`build_argv()` extras append last** so any caller can override defaulted
  flags. CLI flags override `.sumocfg` values; flag-after-flag-of-same-name
  is the last-wins.
- **Heavy imports at module top.** `import traci` and `import sumolib` happen
  at import time of `sumo_runtime.py`. This is OK because:
  (a) the module's name signals intent — anyone importing it wants SUMO;
  (b) `laneiq/env/__init__.py` is empty, so plain `import laneiq` does NOT
       transitively load TraCI.
  Don't change this without re-checking the parent invariant in
  `Project-Documentation/laneiq/OVERVIEW.md`.

## Tests

`tests/test_sumo_runtime.py` (7 tests, all green):

- 5 fast unit tests on `build_argv()` — locked-flag presence, override
  ordering, path resolution.
- 1 fast test that `sumo_binary()` returns existing paths.
- 1 `@slow @sumo` integration test that actually launches SUMO via the
  context manager, steps 100 sim-ticks, asserts time advanced to 10.0 s,
  and confirms the connection closes cleanly.

The slow integration test runs in <0.5 s on Apple Silicon — startup is fast
enough that we can run it in default `pytest` (no `not slow` filter needed
yet at this scale).

## Last updated

2026-04-26 · `1f423f2` (sumo_runtime.py landed; 7 tests green; lint clean)
