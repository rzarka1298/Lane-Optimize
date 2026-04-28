# `laneiq/env/sumo_runtime.py`

## What

Process-level lifecycle for the SUMO simulator. The single place that owns:

- Resolving the SUMO binary inside the bundled `eclipse-sumo` wheel.
- Building the long argv (config path + locked flags + optional overrides)
  that every SUMO launch uses.
- Starting/closing the TraCI connection, with cleanup guarantees.

If the project ever swaps from TraCI to libsumo (Week 3 profiling decides),
**this is the only file that changes.**

## Public API

```python
from laneiq.env.sumo_runtime import (
    DEFAULT_SCENARIO_DIR,    # Path → sumo_scenarios/highway_3lane/
    DEFAULT_SUMOCFG,         # Path → highway.sumocfg
    sumo_binary,             # (gui: bool = False) → str
    build_argv,              # full signature below
    sumo_session,            # context manager; full signature below
)
```

### `sumo_binary(gui: bool = False) -> str`

Returns the absolute path to the `sumo` (headless) or `sumo-gui` binary
inside the venv (`.venv/lib/python3.11/site-packages/sumo/bin/`). Resolved
via `sumolib.checkBinary()`, which is wheel-aware.

### `build_argv(sumocfg, *, gui, seed, route_file, end_time, extra_flags) -> list[str]`

Composes the argv list ready for `traci.start(...)` or a direct subprocess.

| Param | Type | Default | Effect |
| --- | --- | --- | --- |
| `sumocfg` | path | `DEFAULT_SUMOCFG` | The `.sumocfg` file. |
| `gui` | bool | `False` | If True, launch `sumo-gui`. |
| `seed` | int \| None | None | Reproducibility seed (passed as `--seed N`). |
| `route_file` | path \| None | None | Override the routes file (`--route-files`). |
| `end_time` | float \| None | None | Override the sim end time (`--end T`). |
| `extra_flags` | list[str] \| None | None | Tail-appended raw flags. Override anything. |

The locked flags (always present, regardless of args):

```
--no-step-log true              quieter logs during training
--time-to-teleport -1            never silently teleport stuck cars
--collision.action warn          let TraCI detect collisions; don't auto-remove
--collision.mingap-factor 0      collisions register at true zero overlap
--step-length 0.1                10 Hz sim
```

These are load-bearing for RL — see
`Project-Documentation/sumo_scenarios/config.md` for the rationale.

### `sumo_session(...) -> ContextManager[None]`

Same args as `build_argv` plus `label: str = "laneiq"` for the TraCI
connection name. Yields no value (use the global `traci` API after entering).

```python
import traci
from laneiq.env.sumo_runtime import sumo_session

with sumo_session(seed=42, end_time=60.0):
    for _ in range(100):
        traci.simulationStep()
    assert abs(traci.simulation.getTime() - 10.0) < 0.01
```

## Design notes

### Why a dedicated module instead of inlining

The Gym env (`highway_env.py`, landing in Week 2) needs the same launch
logic but manages its connection lifetime across many `reset()` calls. Tests
and ad-hoc scripts also need launch-and-cleanup. A module-level seam keeps
all three on the same code path.

### Why the context manager + `try/finally` + `suppress`

RL training calls `reset()` thousands of times. A leaked TraCI socket
eventually exhausts the local port range and SUMO refuses new connections
with cryptic errors. The `finally` guarantees `traci.close(False)` is
called.

`contextlib.suppress(traci.exceptions.FatalTraCIError)` around the close
exists because if SUMO died inside the `with` block (e.g., bad sumocfg,
process killed), `traci.close()` raises `FatalTraCIError`. Letting that
propagate would mask the *original* exception with a misleading close
error. Suppressing only this specific exception is safer than a bare
`except Exception`.

### `gui` flag semantics

The same code path can launch `sumo-gui` for visual debugging. The Gym env
will pass `gui=False` always (RL doesn't render); the `make sumo-gui-*`
targets bypass this module entirely and call the binary directly.

### `extra_flags` appended last

CLI flags that appear later in argv override earlier ones. By appending
caller-provided extras last, we let any caller override even our
"locked" flags if they really need to (e.g., a test that wants
`--collision.action remove` to verify env reaction to a different policy).

## Invariants & gotchas

- **`sumo_session()` is for tests/scripts only.** Do NOT wrap a `LaneIQEnv`
  in it — the env owns its connection. Wrapping creates a double-start race.
- **Single connection per label.** The default `label="laneiq"` is fine for
  everything in this repo; multi-agent training uses one SUMO with multiple
  egos, not multiple connections.
- **Heavy imports at top of module.** `import traci` and `import sumolib`
  fire at module import time. This is intentional — anyone importing
  `sumo_runtime` signals intent to use SUMO. It's safe because
  `laneiq/__init__.py` and `laneiq/env/__init__.py` are empty (per the
  invariant in `laneiq/OVERVIEW.md`), so plain `import laneiq` does NOT
  transitively load TraCI.
- **`build_argv` resolves paths to absolute.** Callers can pass relative
  paths and they'll be resolved against `cwd` at call time. If you want
  repo-relative behavior, pass an absolute path (e.g., `DEFAULT_SUMOCFG`).

## Tests

`tests/test_sumo_runtime.py` (7 tests):

| Test | What it verifies | Speed |
| --- | --- | --- |
| `test_default_paths_resolve` | `DEFAULT_SCENARIO_DIR` and `DEFAULT_SUMOCFG` point at real files | fast |
| `test_sumo_binary_returns_existing_path` | binary path exists for both `gui=True` and `gui=False` | fast |
| `test_build_argv_default_includes_locked_flags` | all `_LOCKED_FLAGS` appear in default argv | fast |
| `test_build_argv_with_seed_and_route_override` | `--seed`, `--route-files`, `--end` override correctly | fast |
| `test_build_argv_extra_flags_appended_last` | caller extras override defaulted flags | fast |
| `test_sumo_session_steps_and_closes` | actually launches SUMO, steps 100 ticks, asserts time advanced, confirms clean close | slow + sumo |

Run all of them:

```bash
uv run pytest tests/test_sumo_runtime.py -v
```

## Last updated

2026-04-26 · `1f423f2` (initial implementation; 7 tests green; lint clean)
