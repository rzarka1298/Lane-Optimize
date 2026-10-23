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
│  DQN (from scratch) + PPO (MaskablePPO) + MAPPO + rule baselines│
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
│  3-lane 1km highway + 4 vTypes + 5 scenario route files (XML)   │
│  sumo_scenarios/highway_3lane/                                   │
└─────────────────────────────────────────────────────────────────┘
```

## Layer 1 — SUMO scenario

Hand-authored XML defining the simulated world. A single 1 km edge
with 3 lanes, four vehicle "personalities" (`aggressive`, `normal`,
`slow`, `ego`), three Poisson flow densities (`low`, `medium`,
`high`), plus two special scenarios: `gauntlet` (the ego starts in
lane 0 behind a slow wall) and `rush` (a 10 km road with a 50 m/s
limit, used by the Aggressive demo).

| File | What |
| --- | --- |
| `highway.nod.xml` + `highway.edg.xml` | Topology: 1 km edge, 3 lanes, 36.11 m/s speed limit |
| `highway.net.xml` | Generated from the above by `netconvert` (committed for cold-clone reproducibility) |
| `highway_long.*` | The 10 km variant used by `rush` |
| `vtypes.add.xml` | The 4 vehicle profiles (Krauss + LC2013 parameters). The ego pins `speedFactor="1" speedDev="0"` so its top speed is the lane limit on every seed. |
| `routes_{low,medium,high}.rou.xml` | Poisson flows: 0.25 / 0.65 / 1.45 vehicles/s |
| `routes_{gauntlet,rush}.rou.xml` | Scenario-specific traffic |
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
| `observations.py` | 25-dimensional observation vector (ego state + per-lane neighbors + lane-validity mask), built from a TraCI snapshot. A real-SUMO test pins the side-lane slots to the true lanes. |
| `actions.py` | 3 discrete actions (`STAY`, `CHANGE_LEFT`, `CHANGE_RIGHT`); `valid_actions_mask`; `LaneChangeCooldown` (1.5 s between executed changes); the ego lane-change modes (`768` default = SUMO refuses unsafe requests, `0` = unchecked). |
| `rewards.py` | 6-component reward (`speed`, `lane_cost`, `jerk`, `unsafe_gap`, `collision`, `alive`); pure function decoupled from observations. |
| `highway_env.py` | The integrated `LaneIQEnv` class — the seam every agent and trainer talks to. Reports `info["executed_action"]`, the lane change SUMO actually made. |
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
| `OvertakerAgent` | rule-based, demo only | `laneiq/agents/baselines/overtaker_agent.py` — the rush weaver |
| `DQNAgent` | value-based RL | `laneiq/agents/dqn/` (from scratch in PyTorch) |
| `PPOAgent` | policy-gradient RL | `laneiq/agents/ppo_sb3.py` (wraps `sb3_contrib.MaskablePPO`) |
| MAPPO (parameter-sharing) | multi-agent RL | `scripts/train_mappo.py` (plain `stable_baselines3.PPO` over a supersuit-wrapped `LaneIQParallelEnv`) |

**Why DQN from scratch but PPO from a library:** the DQN code is the
"I built this" portfolio piece — replay buffer, target network, ε-greedy,
Double DQN, Huber loss. PPO is the "I can pick the right tool" piece —
SB3's MaskablePPO is the production-grade impl with action masking
support, faster to train and more debuggable than a from-scratch reimpl.

**Why MAPPO uses plain PPO not MaskablePPO:** supersuit's vec-env
wrapping doesn't propagate action masks. The env turns an invalid or
cooling-down request into `STAY`, and observation slot 23 shows each ego
when a change isn't available, so the policy can still learn it.

## Layer 4 — Evaluation

`laneiq/eval/` is the reproducible evaluation harness. The matrix CLI
takes `agents × densities × seeds`, runs each cell, and writes an
aggregated CSV (one row per cell, mean ± std) plus an optional
per-episode CSV.

| Module | What |
| --- | --- |
| `metrics.py` | `EpisodeMetrics` dataclass (speed, collision, episode length, lc/min, jerk, near-miss rate, return…). Pure observer fed by the runner. |
| `runner.py` | Drives one episode end-to-end; counts lane changes from `info["executed_action"]`. |
| `compare.py` | The matrix CLI. `--ego-lane-change-mode` overrides the env default. |

The current 20-seed eval (5 agents × 3 densities × 20 seeds = 300
episodes) lives at
`results/ctrl_eval_matrix_existing.csv`, with the gauntlet at
`results/ctrl_eval_gauntlet_existing.csv`. Drivers are ranked by
**finish time**: on a fixed 1 km episode the return can't tell a fast
driver from a slow one (see [`rl_design.md`](rl_design.md)).

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

- **HighwayCanvas** — Canvas2D top-down road, follow-camera on the ego
- **DecisionTicker** — chip with the action the car actually took (`decision.action` is the executed action, so a refused change shows as "stay") + English explanation
- **MetricsPanel** — stat cards + Recharts cumulative-return + per-component reward charts
- **Diagnostics + collapsible "How it learned" / "How it uses lanes"** panels

**Why the canvas doesn't stutter:** ticks go into a `TickBuffer`
(`frontend/src/api/tickBuffer.ts`) outside React. The draw loop renders
200 ms behind the newest tick and interpolates positions by sim time
`t`, against a smoothed estimate of when each tick arrived. Network or
render jitter therefore never reaches the picture. React state (charts,
ticker) is committed at most every 250 ms, and immediately on an
episode end or an action change, so chart renders don't compete with
the 60 fps draw loop. SUMO moves a car a whole lane in one step; the
canvas eases lane changes over ~0.5 s and fades cars in and out at the
window edges. All of it is display-only and the simulation is untouched.

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
