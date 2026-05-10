# LaneIQ

Reinforcement learning for real-time highway lane optimization in [SUMO](https://eclipse.dev/sumo/).

> 🚧 **Work in progress.** Built in vertical slices over 8 weeks; every week ends with something demoable end-to-end.

## What this is

A SUMO-based 3-lane highway simulation where an RL agent learns to make lane-change decisions
balancing **speed**, **safety**, and **stability** against three rule-based baselines.

Two algorithms compared:

- **DQN** — implemented from scratch in PyTorch (replay buffer, target network, ε-greedy).
- **PPO** — via [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3).

Plus **multi-agent parameter-sharing PPO** over PettingZoo for cooperative lane behavior at high
density.

A live React + FastAPI dashboard streams the simulation and the agent's decisions per tick.

## Status

| Component                    | Status |
| ---------------------------- | ------ |
| SUMO scenario (3-lane, 3 densities) | 🚧 Week 1 |
| `LaneIQEnv` (Gymnasium)      | ⏳     |
| Baselines (random/greedy/safe) | ⏳   |
| DQN (from scratch)           | ⏳     |
| PPO (SB3)                    | ⏳     |
| Multi-agent (parameter-sharing PPO) | ⏳ |
| FastAPI backend              | ⏳     |
| React dashboard              | ⏳     |
| Demo video                   | ⏳     |

## Quickstart

> ⚠️ Requires Python 3.11 and [uv](https://docs.astral.sh/uv/). No system SUMO install
> needed — the `eclipse-sumo` PyPI wheel bundles `sumo`, `sumo-gui`, `netconvert`,
> and `netedit` inside the venv.

```bash
uv sync --extra sumo --extra rl --extra multiagent --extra dev
make scenario      # regenerate SUMO net.xml (optional; committed)
make sumo-gui      # open the highway scenario in SUMO's GUI
```

## Layout

```
laneiq/             Main Python package
  env/              Gymnasium env wrapping SUMO via libsumo/traci
  agents/dqn/       From-scratch DQN
  agents/baselines/ Rule-based comparison agents
  agents/ppo_sb3.py SB3 PPO wrapper
  multi_agent/      Parameter-sharing MAPPO
  eval/             Reproducible evaluation matrix (agents × densities × seeds)
sumo_scenarios/     Hand-authored 3-lane highway (nod/edg/net + vTypes + routes)
backend/            FastAPI live-demo server (TraCI)
frontend/           Vite + React + TS dashboard
configs/            OmegaConf YAMLs (env / reward / agent variants)
```

## License

MIT
