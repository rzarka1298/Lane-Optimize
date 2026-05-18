# LaneIQ

> **An RL agent that learns to drive a highway — trained from scratch,
> evaluated over 300 episodes, and streamed live to a browser.**

<!--
  HERO GIF — drop the recording here.
  Record the 30-second cut from docs/demo_storyboard.md (the gauntlet
  payoff: agent starts stuck behind slow traffic, then escapes), export
  to docs/images/demo.gif (keep it under ~8 MB, ~12 fps), and the line
  below will render it. Until then it shows a broken-image placeholder
  — that's expected pre-recording.
-->
![LaneIQ live demo — the gauntlet agent escaping slow traffic](docs/images/demo.gif)

End-to-end reinforcement learning for highway lane optimization in
[SUMO](https://eclipse.dev/sumo/), with a live React + FastAPI
dashboard streaming a trained agent's decisions in real time.

A SUMO 3-lane highway populated with Poisson background traffic at
three density levels. The ego vehicle is controlled by an RL agent
that picks one of `STAY` / `CHANGE_LEFT` / `CHANGE_RIGHT` every 100 ms
to maximize a 6-component reward trading off speed, safety, and
stability.

**Three RL families compared:**

- **DQN** — implemented from scratch in PyTorch (replay buffer, target
  network, ε-greedy, Double DQN, Huber loss)
- **PPO** — via [`sb3-contrib`](https://github.com/Stable-Baselines3/stable-baselines3-contrib) `MaskablePPO`
- **Multi-agent parameter-sharing PPO** over a PettingZoo
  `ParallelEnv` (4 cooperating egos in one scene)

A 5-component live dashboard renders the SUMO state in real time —
top-down Canvas2D road, AI decision narration, per-step reward
breakdown — and supports both live SUMO playback and pre-recorded
JSON replays through the same WebSocket path.

## Headline numbers

5 RL agents, 20 seeds × 3 densities = 300 episodes per agent. The
canonical PPO (`ppo_v2_safety_heavy_150k`) is the **only single-agent
policy with 0% collisions across all densities** AND the highest
total return (693), trained in 30% of the budget of the alternative
PPO variant.

| Agent | Total return | Worst-case collisions |
| --- | --- | --- |
| `random` | 233 | 90% (high) |
| `greedy` | 509 | 45% (high) |
| `safe_rule` | 674 | 15% (high) |
| `dqn_v3_mixed_200k` | 627 | 25% (high) |
| **`ppo_v2_safety_heavy_150k`** | **693** | **0% everywhere** |

A 5-seed eval (Week 4) suggested the alternative PPO was equally
safe; the 20-seed re-run revealed 15% high-density collisions hiding
under sampling noise. **Reward shaping outweighed training duration**
for safety-critical RL.

Multi-agent MAPPO over 4 concurrent egos: shared policy keeps **80–95%
of egos alive** across all densities — see
[docs/results.md](docs/results.md) for the per-density breakdown.

## Documentation

The `docs/` tree is the public-facing documentation:

| Page | What it covers |
| --- | --- |
| [docs/architecture.md](docs/architecture.md) | System layers: SUMO scenario → Gym env → agents → eval → live demo |
| [docs/rl_design.md](docs/rl_design.md) | Observation / action / reward design + algorithm walkthroughs |
| [docs/results.md](docs/results.md) | Final 20-seed eval matrix + plots + headline narrative |
| [docs/quickstart.md](docs/quickstart.md) | Install + run + train + re-eval, ~10 min from clone |

## Quickstart

```bash
git clone https://github.com/rzarka1298/Lane-Optimize.git
cd Lane-Optimize
make install                              # uv sync; ~3-5 min on a fresh machine
make test                                 # 176 tests, ~30 sec total

# In one terminal — backend:
uv run uvicorn backend.app.main:app --reload --port 8000

# In another terminal — frontend:
cd frontend && npm install && npm run dev
# → open http://localhost:5173
```

Apple Silicon is the primary supported platform; x86_64 Linux works
too. No separate SUMO install needed — the `eclipse-sumo` PyPI wheel
ships the full toolchain inside `.venv/bin/`.

Full instructions in [docs/quickstart.md](docs/quickstart.md).

## Repo layout

```
laneiq/                Main Python package
  env/                 Gymnasium env wrapping SUMO via TraCI
  agents/dqn/          From-scratch DQN (network, replay, agent, train)
  agents/baselines/    Three rule-based comparison agents
  agents/ppo_sb3.py    SB3 MaskablePPO wrapper (Policy protocol)
  eval/                Reproducible evaluation matrix
sumo_scenarios/        Hand-authored 3-lane highway scenario (XML)
backend/               FastAPI live-demo server (TraCI + WebSocket)
frontend/              Vite + React + TS dashboard
scripts/               CLI entrypoints (training, sweeping, plotting, recording)
checkpoints/final/     Canonical promoted checkpoints (5 agents shipped)
results/               Eval CSVs + replay JSONs
docs/                  Public-facing documentation (you are here)
```

## What's where

| File | Purpose |
| --- | --- |
| `Makefile` | Convenience targets: `install`, `test`, `lint`, `train-dqn`, `train-ppo`, `eval`, `sumo-gui-{low,medium,high}` |
| `pyproject.toml` | Python deps + extras (`sumo`, `rl`, `multiagent`, `dev`) |
| `frontend/package.json` | React + Vite + Tailwind + Recharts |
| `Lane-Optimize-PRD.pdf` | Original product requirements |
| `CLAUDE.md` | Project instructions used during development |

## Tests

```bash
make test                    # 163 fast tests; ~1 sec
uv run pytest                # all 176 tests including @sumo-marked slow tests; ~30 sec
make lint                    # ruff
cd frontend && npm run lint  # tsc --noEmit
```

Coverage spans every layer: SUMO runtime lifecycle, observation
builder, action handling, reward composer, env, mixed-density
wrapper, multi-agent env, baselines, DQN (network/replay/agent/
trainer), PPO wrapper, eval framework (metrics/runner/compare),
backend (12 endpoint + schema tests).

## Status

✅ Done: SUMO scenario, env, baselines, DQN-from-scratch, PPO, MAPPO,
eval matrix at 20 seeds, FastAPI backend with replay mode, React
dashboard, training-curve plots, lane-efficiency heatmap.

⏳ In progress: demo video, README hero GIF, polished resume bullets.

## License

MIT
