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

## Verifying the install

Smoke checks that should pass after every fresh sync:

```bash
uv run pytest                                          # tests green
uv run ruff check laneiq tests scripts                 # lint green
uv run python -c "import traci, sumolib; print('OK')"  # SUMO bindings importable
uv run sumo --version                                  # SUMO CLI runs
```

The `tests/test_smoke.py` suite asserts the `laneiq` package imports cleanly —
a fast guard against scaffolding regressions.

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
