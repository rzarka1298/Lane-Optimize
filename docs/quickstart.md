# Quickstart

From `git clone` to a live dashboard in ~10 minutes on macOS or Linux.
Apple Silicon is the primary supported platform; x86_64 Linux works
too.

## Prerequisites

- **Python 3.11** (other versions untested; 3.12+ may work)
- **[`uv`](https://github.com/astral-sh/uv)** — install via
  `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Node.js 22+** (for the frontend; older versions may work too)
- **`make`** (for the convenience targets)

No separate SUMO install needed — the `eclipse-sumo` wheel ships the
full binary toolchain inside `.venv/bin/`.

## 1. Install Python deps

```bash
git clone https://github.com/rzarka1298/Lane-Optimize.git
cd Lane-Optimize
make install         # uv sync --extra sumo --extra rl --extra multiagent
```

This pulls in PyTorch, Stable-Baselines3, sb3-contrib, supersuit,
PettingZoo, FastAPI, eclipse-sumo, traci, and friends. ~3-5 min on
a fresh machine.

For development (tests + linting):

```bash
make install-dev     # adds pytest, ruff, etc.
```

## 2. Verify the install

Fast tests (no SUMO, ~1 sec):

```bash
make test
```

Full suite including SUMO integration (~30 sec):

```bash
uv run pytest        # 176 tests; 13 are slow (marked @sumo)
```

Lint:

```bash
make lint            # ruff
```

Visual check (opens SUMO GUI):

```bash
make sumo-gui-medium
```

## 3. Run the live demo

The demo has two processes — backend on port 8000, frontend on port
5173.

**Terminal 1 — backend:**

```bash
uv run uvicorn backend.app.main:app --reload --port 8000
```

You should see:

```
backend starting
INFO:     Uvicorn running on http://127.0.0.1:8000
```

**Terminal 2 — frontend:**

```bash
cd frontend
npm install          # first time only
npm run dev          # serves http://localhost:5173
```

Open **http://localhost:5173** in a browser.

## 4. Drive the demo

1. The agent dropdown is pre-selected to the canonical PPO
   (`ppo_v2_safety_heavy_150k` — labeled "PPO conservative").
2. Tick **"loop episodes"** if you want continuous playback.
3. Click **Start live**. After ~5 seconds of warmup the ego
   appears (gold rectangle, "EGO" label) and starts driving.
4. The DecisionTicker shows the AI's chosen action with a flash
   animation on every change. The cumulative-return chart climbs
   in real time.
5. Try the **quick-pick** buttons at the top: Conservative,
   Aggressive, Multi-agent — each loads a different trained
   policy with a tooltip explaining the trade-off.
6. Try **Play replay** with one of the pre-recorded JSONs
   (visible if `results/replays/` has any files) to see a
   deterministic playback that doesn't even need SUMO running.
7. Scroll to the bottom and expand **"How it learned"** and
   **"How it uses lanes"** for the training-curve and
   lane-occupancy plots.

## 5. (Optional) Re-train an agent

All canonical checkpoints ship in `checkpoints/final/`. To train a
fresh one:

```bash
# DQN from scratch (mixed density, 200k env steps, ~14 min on Apple Silicon)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_dqn.py \
    --density mixed \
    --total-steps 200000 \
    --seed 42 \
    --log-dir runs/dqn_fresh \
    --checkpoint-out checkpoints/dqn_fresh.pt

# PPO with custom reward weights (experiment with the racer demo)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_ppo.py \
    --density mixed \
    --total-steps 150000 \
    --seed 42 \
    --w-speed 5.0 --w-collision -10 --w-unsafe-gap -0.05 \
    --log-dir runs/ppo_racer \
    --checkpoint-out checkpoints/ppo_racer.zip
```

To use a fresh checkpoint in the live demo, copy it to
`checkpoints/final/`:

```bash
cp checkpoints/ppo_racer.zip checkpoints/final/ppo_racer_150k.zip
```

The backend re-walks `checkpoints/final/` on each `GET /runs` call,
so the dashboard's agent dropdown picks it up without a backend
restart.

## 6. (Optional) Re-run the eval matrix

```bash
# 20-seed final eval (5 agents × 3 densities × 20 seeds = 300 episodes, ~30 min)
uv run python -m laneiq.eval.compare \
    --agents random,greedy,safe_rule,dqn,ppo \
    --densities low,medium,high \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --dqn-checkpoint checkpoints/final/dqn_v3_mixed_200k.pt \
    --ppo-checkpoint checkpoints/final/ppo_v2_safety_heavy_150k.zip \
    --out results/final_eval_matrix.csv \
    --per-episode-out results/final_eval_episodes.csv
```

Then re-render the plots:

```bash
uv run python -m scripts.plot_pareto \
    --csv results/week4_reward_sweep.csv \
    --baselines-csv results/final_eval_matrix.csv \
    --out docs/images/pareto_safety_speed.png

uv run python -m scripts.plot_training_curves
uv run python -m scripts.plot_lane_efficiency \
    --in results/replays/*.json \
    --out docs/images/lane_efficiency.png
```

The `plot_lane_efficiency` script also mirrors the PNG to
`frontend/public/` so the dashboard tile updates automatically.

## Troubleshooting

**"sumo: command not found"** — you're not in the `uv` environment.
The SUMO binary is bundled inside the wheel; `uv run …` handles the
PATH setup. Don't try to install SUMO separately.

**"`cannot connect to TraCI`"** when running the eval/training:
there's a stale SUMO process holding the TraCI port. Cleanup:
`pkill -f sumo`. The next run will work fine.

**Frontend port 5173 already in use** — Vite is set to `strictPort:
true` (because the backend's CORS only allows port 5173). Free the
port: `lsof -i :5173` then `kill <pid>`.

**`/sim/start` returns 404 with "agent not found"** — the agent name
must match a file stem in `checkpoints/final/`. Check `GET /runs` to
see what the backend can see; the frontend's refresh button (↻ next
to the agent dropdown) re-fetches.

**MAPPO checkpoint won't load** — the `checkpoints/final/mappo_*.zip`
naming is load-bearing. The backend's policy loader detects the
`mappo_` prefix and uses plain `stable_baselines3.PPO.load` instead
of MaskablePPO (training-time vs eval-time loader mismatch — see the
backend services doc).
