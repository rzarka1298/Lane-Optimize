# Setup — dev environment

Cross-cutting notes about how this project is installed and run locally.
**Read this before:** changing `pyproject.toml`, modifying anything that touches
SUMO at the runtime/binary level, debugging install failures, or onboarding a
new machine.

## Stack

| Tool | Pin | Why |
| --- | --- | --- |
| Python | `>=3.11,<3.12` (pinned in `pyproject.toml`) | libsumo wheels lag a Python release; `eclipse-sumo` 1.26 ships arm64 only for 3.10/3.11. Don't bump to 3.12 without checking PyPI. |
| uv | latest | One-command sync (`uv sync`), reproducible via `uv.lock`. |
| Hatchling | build backend | Lightweight; no Poetry/Hatch ceremony. |

## Install

```bash
uv sync --extra dev --extra sumo --extra rl --extra multiagent
```

Extras are split so each phase only needs what it uses:

| Extra | Contains | Used from |
| --- | --- | --- |
| `dev` | pytest, ruff, mypy | Always |
| `sumo` | eclipse-sumo, traci, sumolib | Week 1 onward |
| `rl` | stable-baselines3, sb3-contrib | Week 4 onward |
| `multiagent` | pettingzoo, supersuit | Week 5 onward |

## SUMO

### How SUMO is installed

The `eclipse-sumo` PyPI wheel **bundles all SUMO binaries** (`sumo`, `sumo-gui`,
`netconvert`, `netedit`, `polyconvert`, `duarouter`, …) inside
`.venv/lib/python3.11/site-packages/sumo/bin/`. It also installs entry-point
shims into `.venv/bin/`, so `uv run sumo-gui` Just Works — no `SUMO_HOME` to
set, no Homebrew tap to add, no system-level install.

Cold-clone path:

```bash
git clone <repo>
cd Lane-Optimize
uv sync --extra sumo
uv run sumo --version    # confirms install
```

This is bulletproof for recruiter screen-shares and CI.

### libsumo: intentionally absent

The original plan called for `libsumo` (in-process, ~8× faster than the
TCP/socket-based `traci`) for training, with `traci` reserved for the live
demo. **`libsumo` 1.26 ships no macOS arm64 wheel on PyPI** (only manylinux and
win_amd64). Building it from source via the DLR-TS Homebrew tap takes
~30–60 min and adds CMake + FOX-toolkit prerequisites to the README quickstart.

**Decision (2026-04-26):** use TraCI for everything until Week 3 profiling shows
env step rate is the training bottleneck (i.e., env-steps/sec significantly
lower than gradient-steps/sec). Single runtime path, simpler env code.

If Week 3 profiling justifies it, the path back to libsumo is:

```bash
brew tap dlr-ts/sumo
brew install --cask sumo-gui    # optional GUI
brew install dlr-ts/sumo/sumo   # builds with libsumo Python bindings
# ... then add libsumo bindings to PYTHONPATH and add a runtime toggle
# in laneiq/env/sumo_runtime.py
```

This is captured as Task #5 in the live task list.

## Get the project running (full walkthrough)

Use this section any time you want to confirm the project is healthy from
scratch — after pulling new commits, switching machines, or clearing caches.
Steps are ordered: each one builds on the prior. If one fails, stop and fix
before moving on.

### 0. Prereqs

You need:

- macOS (Apple Silicon tested) or Linux. Windows untested.
- [`uv`](https://docs.astral.sh/uv/) on your `PATH`. If missing:
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Python 3.11 — `uv` will fetch it automatically if you don't have it.

### 1. Clone (skip if already done)

```bash
git clone <repo-url> Lane-Optimize
cd Lane-Optimize
```

### 2. Sync dependencies

```bash
uv sync --extra dev --extra sumo
```

Adding `--extra rl` and `--extra multiagent` is fine but not needed until
Week 4–5. The sync downloads ~150 MB (the eclipse-sumo wheel) the first
time; subsequent runs are seconds.

You should see something like:

```
Resolved 80 packages
Installed N packages
+ eclipse-sumo==1.26.0
+ traci==1.26.0
+ sumolib==1.26.0
+ ...
```

### 3. Confirm the SUMO bindings + binaries

```bash
uv run sumo --version
```

Expected: `Eclipse SUMO sumo 1.26.0` followed by a build-features line. If
this errors with "command not found", the venv didn't create entry-point
shims — re-run `uv sync --extra sumo` and check `ls .venv/bin/sumo*`.

```bash
uv run python -c "import traci, sumolib; print('bindings OK')"
```

Expected: `bindings OK`. Any ImportError means the wheel didn't install —
clear the venv (`rm -rf .venv && uv sync --extra dev --extra sumo`) and
retry.

### 4. Lint everything

```bash
make lint
```

or directly:

```bash
uv run ruff check laneiq tests scripts
```

Expected: `All checks passed!`. If anything fails, it's likely a doc edit
that changed line counts in code blocks — fix and retry.

### 5. Run the test suite

The full suite (currently 7 tests) takes ~1.2 s and includes one
integration test that actually launches SUMO. To run it all:

```bash
uv run pytest
```

Expected output:

```
tests/test_smoke.py .                                  [ 14%]
tests/test_sumo_runtime.py ......                      [100%]

============================== 7 passed in ~1.2s ===============================
```

For just the fast tests (skip the SUMO-launching integration test):

```bash
make test            # equivalent to: uv run pytest -m "not slow"
```

If the slow `test_sumo_session_steps_and_closes` fails, the most common
cause is a stuck SUMO process from a previous run holding the TraCI port —
`pkill -f sumo` then retry.

### 6. Eyeball the highway scenario in `sumo-gui`

The headless tests confirm the scenario *parses* and *runs*. To confirm it
*looks right*:

```bash
make sumo-gui-medium       # default density (~20 cars on screen)
# or
make sumo-gui-low          # ~7-8 cars
make sumo-gui-high         # ~40-45 cars
```

A native macOS window opens titled `highway.sumocfg - SUMO 1.26.0`. The
sim is paused at t=0. Recipe to verify:

1. **Set delay.** Top toolbar → "Delay (ms)" field → type `100` → Enter.
   (`0` runs as fast as the GPU can render — too fast to watch.)
2. **Press play** (the green ▶ in the top-left).
3. **Watch for ~30s of sim time:**
   - Cars enter from the left, exit at the right.
   - **Red** (aggressive) cars overtake on either side; rarely yield.
   - **Blue** (slow) cars stick to the rightmost lane.
   - **White** (normal) cars: balanced mix.
   - The bottom message panel stays quiet (no "collision" / "teleport"
     warnings during normal flow).
4. **Inspect a car.** Right-click any vehicle → "Show Parameter" → live
   `speed`, `lane`, `accel`, `gap to leader`. Useful for verifying the
   vTypes are behaving as designed (aggressive cars tailgate; slow cars
   keep distance).

Full visual-debug recipe: `Project-Documentation/sumo_scenarios/OVERVIEW.md`.

### 7. Quick TraCI sanity script (optional)

To confirm you can drive the simulator from Python end-to-end:

```bash
uv run python -c "
import traci
from laneiq.env.sumo_runtime import sumo_session

with sumo_session(seed=42, end_time=60.0):
    for _ in range(50):
        traci.simulationStep()
    print('time:', traci.simulation.getTime(), 'sec')
    print('vehicles:', len(traci.vehicle.getIDList()))
"
```

Expected: `time: 5.0 sec` plus a vehicle count between 1 and 4. This is the
exact pattern the Gym env will use, just simpler.

### 8. (Optional) Re-generate the SUMO network

If you edit `highway.nod.xml` or `highway.edg.xml`, regenerate `highway.net.xml`:

```bash
make scenario
```

This runs `sumo_scenarios/highway_3lane/build.sh`, which calls `netconvert`
via `uv run`. The generated file is committed for cold-clone repro; a
re-run only changes it if your edits actually altered the topology.

## Common failures, fixed

| Symptom | Cause | Fix |
| --- | --- | --- |
| `sumo: command not found` after `uv sync` | venv shims missing | `rm -rf .venv && uv sync --extra dev --extra sumo` |
| `pytest` hangs forever during slow test | stale SUMO process | `pkill -f sumo` |
| `sumo-gui` opens but window is blank | net not generated yet | `make scenario` |
| `'--' sequence is illegal in comment` | XML comment with `--` inside | edit the offending file; `--` is forbidden in XML comment bodies |
| `MPS backend out of memory` | (Week 2+) batch too large | drop batch size or move to CPU; see Apple Silicon notes below |

## Verifying the install (fast smoke-only)

If you just want a quick "is it broken" check after a small change:

```bash
make lint && make test
```

That's ~3 seconds total and covers the green-CI gate.

## Apple Silicon / MPS notes

- Always set `PYTORCH_ENABLE_MPS_FALLBACK=1` before running torch code that
  exercises niche ops. The DQN training script (Week 2+) sets this in its
  Python entrypoint.
- Network sizes that fit MPS comfortably: MLPs up to ~512×512, batches ≤ 256.
  Plenty for a 24-dim observation. Don't try CNNs.
- If you suspect silent miscompute (MPS optimizer-update bugs surface as "loss
  doesn't decrease but no error"): rerun on CPU and compare. If returns differ
  by >5% at convergence, push the optimizer step to CPU and keep the network
  forward pass on MPS.

## Last updated

2026-04-26 · `2a5f8ff` (SUMO Python bindings installed, libsumo dropped)
