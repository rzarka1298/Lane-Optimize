"""Eval-matrix CLI: agents x densities x seeds → CSV.

Run:
    uv run python -m laneiq.eval.compare \
        --agents random,greedy,safe_rule \
        --densities low,medium,high \
        --seeds 0,1,2,3,4 \
        --out results/single_agent_table.csv

For the "with DQN" runs in Week 3+, pass `--dqn-checkpoint <path.pt>` to
include the trained DQN agent. Without that flag, only the rule-based
baselines run.

The output CSV has one row per (agent, density) cell with mean + std
across the seed bank. A separate `--per-episode-out` flag writes the raw
per-episode rows to a second CSV for debugging.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections.abc import Iterable
from dataclasses import asdict
from pathlib import Path

from laneiq.agents.base import Policy
from laneiq.agents.baselines.greedy_agent import GreedyAgent
from laneiq.agents.baselines.random_agent import RandomAgent
from laneiq.agents.baselines.safe_rule_agent import SafeRuleAgent
from laneiq.eval.metrics import AggregatedMetrics, EpisodeMetrics, aggregate
from laneiq.eval.runner import run_episodes

# Map agent name → factory. Add new agents here when they land.
_AGENT_FACTORIES = {
    "random": lambda: RandomAgent(seed=0),
    "greedy": lambda: GreedyAgent(),
    "safe_rule": lambda: SafeRuleAgent(),
    # "dqn": handled specially in build_agent() below since it needs a checkpoint
}


def _parse_csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _parse_int_csv(value: str) -> list[int]:
    return [int(v.strip()) for v in value.split(",") if v.strip()]


def build_agent(name: str, *, dqn_checkpoint: Path | None = None) -> Policy:
    """Construct an agent by name. Raises if unknown."""
    if name == "dqn":
        if dqn_checkpoint is None:
            raise ValueError("--dqn-checkpoint required for agent 'dqn'")
        # Lazy import — torch is heavyweight.
        from laneiq.agents.dqn.dqn_agent import DQNAgent
        from laneiq.env.observations import OBS_DIM

        agent = DQNAgent(obs_dim=OBS_DIM, n_actions=3, device="cpu", seed=0)
        agent.load(dqn_checkpoint)
        return agent
    if name in _AGENT_FACTORIES:
        return _AGENT_FACTORIES[name]()
    raise ValueError(
        f"unknown agent {name!r}; valid: {[*sorted(_AGENT_FACTORIES), 'dqn']}",
    )


def write_aggregated_csv(rows: Iterable[AggregatedMetrics], out: Path) -> None:
    """One row per (agent, density) with mean ± std columns."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        raise ValueError("no rows to write")
    fieldnames = list(asdict(rows_list[0]).keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(asdict(row))


def write_per_episode_csv(rows: Iterable[EpisodeMetrics], out: Path) -> None:
    """One row per (agent, density, seed) — the raw episodes."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows_list = list(rows)
    if not rows_list:
        return
    fieldnames = list(asdict(rows_list[0]).keys())
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows_list:
            writer.writerow(asdict(row))


def run_matrix(
    *,
    agents: list[str],
    densities: list[str],
    seeds: list[int],
    max_steps: int = 1000,
    warmup_steps: int = 50,
    dqn_checkpoint: Path | None = None,
) -> tuple[list[AggregatedMetrics], list[EpisodeMetrics]]:
    """Run the full matrix. Returns (aggregated rows, raw per-episode rows)."""
    aggregated: list[AggregatedMetrics] = []
    per_episode: list[EpisodeMetrics] = []

    for agent_name in agents:
        agent = build_agent(agent_name, dqn_checkpoint=dqn_checkpoint)
        for density in densities:
            t0 = time.monotonic()
            episodes = run_episodes(
                agent,
                density=density,
                seeds=seeds,
                max_steps=max_steps,
                warmup_steps=warmup_steps,
            )
            elapsed = time.monotonic() - t0
            agg = aggregate(episodes)
            aggregated.append(agg)
            per_episode.extend(episodes)
            print(
                f"  {agent_name:<10} x {density:<6}  "
                f"speed={agg.mean_speed_mps_mean:5.2f} m/s  "
                f"return={agg.return_total_mean:7.2f}  "
                f"collisions={agg.collision_rate * 100:5.1f}%  "
                f"lc/min={agg.lane_changes_per_min_mean:5.2f}  "
                f"({len(seeds)} eps in {elapsed:5.1f}s)",
                flush=True,
            )

    return aggregated, per_episode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the eval matrix: agents x densities x seeds → CSV.",
    )
    parser.add_argument(
        "--agents",
        type=_parse_csv,
        default=["random", "greedy", "safe_rule"],
        help="comma-separated agent names (default: 3 baselines)",
    )
    parser.add_argument(
        "--densities",
        type=_parse_csv,
        default=["low", "medium", "high"],
        help="comma-separated densities (default: all 3)",
    )
    parser.add_argument(
        "--seeds",
        type=_parse_int_csv,
        default=[0, 1, 2, 3, 4],
        help="comma-separated seeds (default: 0,1,2,3,4 — dev-grade)",
    )
    parser.add_argument(
        "--max-steps", type=int, default=1000,
        help="env max_steps per episode",
    )
    parser.add_argument(
        "--warmup-steps", type=int, default=50,
        help="env warmup_steps before ego enters",
    )
    parser.add_argument(
        "--dqn-checkpoint", type=Path, default=None,
        help="path to a DQN checkpoint .pt file (required if 'dqn' in --agents)",
    )
    parser.add_argument(
        "--out", type=Path, default=Path("results/single_agent_table.csv"),
        help="output CSV path for aggregated results",
    )
    parser.add_argument(
        "--per-episode-out", type=Path, default=None,
        help="optional path for raw per-episode CSV (debugging)",
    )
    args = parser.parse_args(argv)

    print("\nLaneIQ eval matrix")
    print("=" * 70)
    print(f"  agents:    {args.agents}")
    print(f"  densities: {args.densities}")
    print(f"  seeds:     {args.seeds}")
    print(f"  out:       {args.out}")
    print("-" * 70)

    aggregated, per_episode = run_matrix(
        agents=args.agents,
        densities=args.densities,
        seeds=args.seeds,
        max_steps=args.max_steps,
        warmup_steps=args.warmup_steps,
        dqn_checkpoint=args.dqn_checkpoint,
    )

    write_aggregated_csv(aggregated, args.out)
    print(f"\n✓ wrote aggregated → {args.out}")
    if args.per_episode_out is not None:
        write_per_episode_csv(per_episode, args.per_episode_out)
        print(f"✓ wrote per-episode → {args.per_episode_out}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
