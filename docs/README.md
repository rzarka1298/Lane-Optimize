# LaneIQ — documentation

End-to-end reinforcement learning for highway lane optimization in
[SUMO](https://eclipse.dev/sumo/), with a live React + FastAPI dashboard
that streams a trained agent's decisions in real time.

This `docs/` tree is the **public-facing** documentation — start here if
you've just cloned the repo or are reviewing the project. (Working /
internal notes live in `Project-Documentation/`, which is gitignored.)

## Table of contents

| Page | What it covers |
| --- | --- |
| **[architecture.md](architecture.md)** | System layers: SUMO scenario → Gym env → agents → evaluation → live demo. Where each piece of code lives and what it depends on. |
| **[rl_design.md](rl_design.md)** | The RL problem statement, observation/action spaces, reward function, and the algorithms (DQN-from-scratch, PPO via SB3, parameter-sharing MAPPO). |
| **[results.md](results.md)** | The numbers: 20-seed final eval matrix, v1-vs-v2 PPO comparison, MAPPO multi-agent results, embedded plots, and the headline narrative. |
| **[quickstart.md](quickstart.md)** | Install the project, run the test suite, drive the live demo locally. ~10 minutes from `git clone` to "watching cars on the dashboard." |

## Headline numbers

5 RL agents, 20 seeds × 3 densities = 300 episodes per agent. The
canonical PPO (`ppo_v2_safety_heavy_150k`) is the only single-agent
policy with **0% collisions across all densities** AND the highest
total return (693). It reaches that with **30% of the training budget**
of the alternative PPO variant — empirical evidence that reward shaping
outweighs training duration for safety-critical RL.

| Agent | Total return (sum across densities) | Worst-case collision rate |
| --- | --- | --- |
| `random` | 233 | 90% (high) |
| `greedy` | 509 | 45% (high) |
| `safe_rule` | 674 | 15% (high) |
| `dqn_v3_mixed_200k` | 627 | 25% (high) |
| **`ppo_v2_safety_heavy_150k`** | **693** | **0% everywhere** |

Multi-agent MAPPO over 4 concurrent egos in one scene: shared policy
keeps **80–95% of egos alive across all densities** — see
[results.md](results.md) for the per-density breakdown.

## Plots

Four canonical figures live in `docs/images/`:

- `pareto_safety_speed.png` — Week-4 reward-sweep Pareto frontier
- `mappo_vs_ppo.png` — multi-agent vs single-agent comparison
- `training_curves.png` — learning curves for all four canonical agents
- `lane_efficiency.png` — per-replay lane occupancy with ego trajectory

Each is referenced inline in the relevant doc page.
