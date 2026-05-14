# Architecture

LaneIQ is built in five layers. Each layer is independently testable
and depends only on the layers below it.

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5 — Live demo                                             │
│  React dashboard (Vite + TS + Tailwind + Recharts) ← WS ← FastAPI│
│  frontend/                                            backend/    │
└─────────────────────────────────────────────────────────────────┘
              ▲                                       ▲
              │                                       │ uses
              │ consumes                              │
              │                                       │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4 — Evaluation                                            │
│  Reproducible eval matrix: agents × densities × seeds → CSV      │
│  laneiq/eval/                                                    │
└─────────────────────────────────────────────────────────────────┘
                                                      ▲
                                                      │ drives
                                                      │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3 — Agents                                                │
│  DQN (from scratch) + PPO (sb3-contrib MaskablePPO) + MAPPO     │
│  laneiq/agents/                                                  │
└─────────────────────────────────────────────────────────────────┘
                                                      ▲
                                                      │ consumes
                                                      │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2 — Gymnasium env                                         │
│  LaneIQEnv + MixedDensityEnv + LaneIQParallelEnv (PettingZoo)   │
│  laneiq/env/                                                     │
└─────────────────────────────────────────────────────────────────┘
                                                      ▲
                                                      │ wraps
                                                      │
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — SUMO scenario                                         │
│  3-lane 1km highway + 4 vTypes + 3 density route files (XML)    │
│  sumo_scenarios/highway_3lane/                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1 — SUMO scenario

Hand-authored XML defining the simulated world. A single 1 km edge
with 3 lanes, four vehicle "personalities" (`aggressive`, `normal`,
`slow`, `ego`), and three Poisson flow densities (`low`, `medium`,
`high`).

| File | What |
| --- | --- |
| `highway.nod.xml` + `highway.edg.xml` | Topology: 1 km edge, 3 lanes, 36.11 m/s speed limit |
| `highway.net.xml` | Generated from the above by `netconvert` (committed for cold-clone reproducibility) |
| `vtypes.add.xml` | The 4 vehicle profiles (Krauss + LC2013 parameters) |
| `routes_{low,medium,high}.rou.xml` | Poisson flows: 0.25 / 0.65 / 1.45 vehicles/s |
| `highway.sumocfg` | Top-level config; sets the load-bearing collision policy |

Background traffic is spawned via SUMO `<flow>` elements with
`period="exp(λ)"`. The ego is **not** in any route file — the Gym env
adds it at runtime via `traci.vehicle.add(...)` after a 5-second warmup
that populates background traffic.

## Layer 2 — Gymnasium environment

`laneiq/env/highway_env.py:LaneIQEnv` wraps SUMO via TraCI and exposes
the standard Gymnasium API. **Passes `gymnasium.utils.env_checker.check_env`.**

| Module | What |
| --- | --- |
| `sumo_runtime.py` | TraCI lifecycle (`build_argv`, `sumo_session`, `sumo_binary`). Manages SUMO process startup/teardown. |
| `observations.py` | 25-dimensional observation vector (ego state + per-lane neighbors + lane-validity mask). Pure function over a TraCI snapshot. |
| `actions.py` | 3 discrete actions (`STAY`, `CHANGE_LEFT`, `CHANGE_RIGHT`); `valid_actions_mask` enforces physical constraints. |
| `rewards.py` | 6-component reward (`speed`, `lane_cost`, `jerk`, `unsafe_gap`, `collision`, `alive`); pure function decoupled from observations. |
| `highway_env.py` | The integrated `LaneIQEnv` class — the seam every agent and trainer talks to. |
| `mixed_density_env.py` | Wrapper that samples `low`/`medium`/`high` per `reset()` for robust training. |
| `multi_agent_env.py` | PettingZoo `ParallelEnv` with N concurrent egos. Used by MAPPO training. |

**Why TraCI everywhere (no libsumo):** `libsumo` (the in-process variant)
doesn't ship a macOS arm64 wheel. TraCI runs at 200–310 env-steps/sec —
fast enough that the gradient step is the bottleneck for our 17k-parameter
network. Single runtime path keeps the code simple.

## Layer 3 — Agents

Three RL families plus three rule-based baselines, all implementing the
same `Policy` protocol so the eval harness can drive them
interchangeably.

| Agent | Type | Implementation |
| --- | --- | --- |
| `RandomAgent`, `GreedyAgent`, `SafeRuleAgent` | rule-based | `laneiq/agents/baselines/` |
| `DQNAgent` | value-based RL | `laneiq/agents/dqn/` (from scratch in PyTorch) |
| `PPOAgent` | policy-gradient RL | `laneiq/agents/ppo_sb3.py` (wraps `sb3_contrib.MaskablePPO`) |
| MAPPO (parameter-sharing) | multi-agent RL | `scripts/train_mappo.py` (plain `stable_baselines3.PPO` over a supersuit-wrapped `LaneIQParallelEnv`) |

**Why DQN from scratch but PPO from a library:** the DQN code is the
"I built this" portfolio piece — replay buffer, target network, ε-greedy,
Double DQN, Huber loss. PPO is the "I can pick the right tool" piece —
SB3's MaskablePPO is the production-grade impl with action masking
support, faster to train and more debuggable than a from-scratch reimpl.

**Why MAPPO uses plain PPO not MaskablePPO:** supersuit's vec-env
wrapping doesn't propagate action masks. We documented the trade-off in
the codebase and rely on the env's silent-no-op-on-invalid-lane behavior.

## Layer 4 — Evaluation

`laneiq/eval/` is the reproducible evaluation harness. The matrix CLI
takes `agents × densities × seeds`, runs each cell, and writes an
aggregated CSV (one row per cell, mean ± std) plus an optional
per-episode CSV.

| Module | What |
| --- | --- |
| `metrics.py` | `EpisodeMetrics` dataclass (15 fields per episode: speed, collision, lc/min, jerk, near-miss rate, return…). Pure observer fed by the runner. |
| `runner.py` | Drives one episode end-to-end. |
| `compare.py` | The matrix CLI. Produces all `results/*_eval_matrix.csv` files. |

The Week-7 final eval at 20 seeds (5 agents × 3 densities × 20 seeds =
300 episodes) takes ~30 minutes wall-clock and lives at
`results/final_eval_matrix.csv`.

## Layer 5 — Live demo

A FastAPI backend that owns one running SUMO simulation per process
streams per-tick state to a React frontend over WebSocket.

| Endpoint | What |
| --- | --- |
| `POST /sim/start` | Start a live sim with an agent + density + seed |
| `POST /sim/start_replay` | Stream a pre-recorded JSON instead of running SUMO |
| `POST /sim/stop` | Idempotent stop |
| `GET /sim/status` | Current run state |
| `GET /runs` | Enumerate `checkpoints/final/` |
| `GET /replays` | Enumerate `results/replays/*.json` |
| `WS /sim/stream` | Per-tick `SimTick` payloads at 10 Hz (same envelope for live AND replay) |

The frontend (Vite + React 18 + TypeScript + Tailwind v3) consumes
these via a typed API client and renders four panels:

- **HighwayCanvas** — Canvas2D top-down road with rAF-interpolated vehicles
- **DecisionTicker** — chip with the AI's chosen action + English explanation + DQN Q-bars
- **MetricsPanel** — stat cards + Recharts cumulative-return + per-component reward charts
- **Diagnostics + collapsible "How it learned" / "How it uses lanes"** panels

**Why the WS envelope is unified live-vs-replay:** the recording script
(`scripts/record_replay.py`) emits ticks via the same `build_tick`
function the live simulator uses. The frontend doesn't care whether
it's watching a live SUMO process or a JSON playback — same parser,
same renderer.

## Why these choices

A handful of architectural decisions have load-bearing rationale:

- **TraCI not libsumo** — no macOS arm64 wheel; throughput is fine.
- **CPU not MPS for training** — MPS is 8× slower for our 17k-param
  net (kernel-launch overhead dominates).
- **Huber loss for DQN, MSE for CartPole** — sparse `-50` collision
  reward exploded MSE gradients (loss 935× too high); Huber's linear
  large-error regime fixed it. CartPole's bounded `+1/step` reward
  doesn't have that pathology.
- **`max_steps=1000` per episode** — 100 sim seconds, long enough for
  the 1 km road to be traversable, short enough that DQN replay
  buffers don't blow up.
- **Single sim per backend process** — TraCI's connection state is
  module-level globals; can't run two sims in one process. Fine for
  a single-viewer recruiter demo.
