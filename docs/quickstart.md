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

Full suite including SUMO integration (~1-2 min):

```bash
uv run pytest        # 219 tests; 29 are slow (marked @sumo)
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
   (`ppo_v12_safety_cd_150k`, labeled "PPO canonical"). Checkpoints
   trained before the side-lane observation fix are still listed,
   marked "(pre-fix)".
2. Tick **"loop episodes"** if you want continuous playback.
3. Click **Start live**. After ~5 seconds of warmup the ego
   appears (gold rectangle, "EGO" label) and starts driving.
4. The DecisionTicker shows the action the car actually took, with a
   flash animation on every change. A lane change SUMO refused as
   unsafe shows as "stay". The cumulative-return chart climbs in real
   time.
5. Try the **quick-pick** buttons at the top: Conservative
   (`ppo_v12`), Aggressive (the rule-based overtaker on the 10 km rush
   road), Multi-agent (`mappo_v3`), Gauntlet (`ppo_v10`). Each has a
   tooltip explaining the trade-off.
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
# DQN from scratch (mixed density, 200k env steps)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_dqn.py \
    --density mixed \
    --total-steps 200000 \
    --seed 42 \
    --log-dir runs/dqn_fresh \
    --checkpoint-out checkpoints/dqn_fresh.pt

# PPO — the canonical ppo_v12 recipe (safety weights + lane-change cost)
PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/train_ppo.py \
    --density mixed \
    --total-steps 150000 \
    --seed 42 \
    --w-collision -100 --w-unsafe-gap -1.0 --w-lane-cost -1.0 \
    --log-dir runs/ppo_fresh \
    --checkpoint-out checkpoints/ppo_fresh.zip

# Parameter-sharing MAPPO — the mappo_v3 recipe (4 egos, cooperative bonus)
uv run python scripts/train_mappo.py \
    --shared-speed-bonus 0.1 --w-lane-cost -1.0 \
    --checkpoint-out checkpoints/mappo_fresh.zip
```

Training runs under the env defaults: 1.5 s lane-change cooldown and
safety-checked lane changes (`ego_lane_change_mode=768`).
`--w-alive` takes a negative value to turn the alive bonus into a
per-step time penalty. We tried `-0.19` and it made PPO slower (see
[`rl_design.md`](rl_design.md)).

To use a fresh checkpoint in the live demo, copy it to
`checkpoints/final/`:

```bash
cp checkpoints/ppo_fresh.zip checkpoints/final/ppo_fresh_150k.zip
```

The backend re-walks `checkpoints/final/` on each `GET /runs` call,
so the dashboard's agent dropdown picks it up without a backend
restart.

## 6. (Optional) Re-run the eval matrix

```bash
# 20-seed eval (5 agents × 3 densities × 20 seeds = 300 episodes)
uv run python -m laneiq.eval.compare \
    --agents random,greedy,safe_rule,dqn,ppo \
    --densities low,medium,high \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --dqn-checkpoint checkpoints/final/dqn_v4_mixed_200k_obsfix.pt \
    --ppo-checkpoint checkpoints/final/ppo_v12_safety_cd_150k.zip \
    --out results/ctrl_eval_matrix_existing.csv \
    --per-episode-out results/ctrl_eval_matrix_existing_episodes.csv

# Gauntlet (PPO slot = the gauntlet specialist)
uv run python -m laneiq.eval.compare \
    --agents random,greedy,safe_rule,dqn,ppo --densities gauntlet \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --dqn-checkpoint checkpoints/final/dqn_v4_mixed_200k_obsfix.pt \
    --ppo-checkpoint checkpoints/final/ppo_v10_gauntlet_cd_300k.zip \
    --out results/ctrl_eval_gauntlet_existing.csv \
    --per-episode-out results/ctrl_eval_gauntlet_existing_episodes.csv

# MAPPO, per ego
uv run python scripts/eval_mappo.py \
    --checkpoint checkpoints/final/mappo_v3_coop_cd_300k.zip \
    --densities low,medium,high \
    --seeds 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --out results/ctrl_eval_mappo_v3_existing.csv
```

Rank drivers by `episode_length_s_mean` (finish time), not return: on a
fixed 1 km episode the return can't tell a fast driver from a slow one.
Add `--ego-lane-change-mode 0` to either eval CLI to reproduce the old
unchecked lane changes.

Then re-render the plots:

```bash
uv run python -m scripts.plot_pareto \
    --csv results/week4_reward_sweep.csv \
    --baselines-csv results/ctrl_eval_matrix_existing.csv \
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
