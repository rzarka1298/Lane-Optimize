"""Evaluate a trained MAPPO (parameter-sharing PPO) checkpoint.

Runs the shared policy in `LaneIQParallelEnv` across densities × seeds,
records per-agent metrics. Single big difference vs single-agent eval:
each episode produces N=4 rows (one per ego), not one.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/eval_mappo.py \\
        --checkpoint checkpoints/final/mappo_v1_300k.zip \\
        --densities low,medium,high \\
        --seeds 0,1,2,3,4 \\
        --out results/week5_mappo_eval.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


@dataclass(frozen=True)
class MAPPOAgentMetrics:
    """One row per (episode, ego). The MAPPO equivalent of EpisodeMetrics."""

    density: str
    seed: int
    ego_id: str            # ego_0 .. ego_{N-1}
    n_egos: int            # how many total in this episode
    return_total: float
    episode_length_steps: int
    collided: bool
    reached_end: bool
    mean_speed_mps: float
    lane_changes: int
    lane_changes_per_min: float
    near_miss_rate: float


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Eval a trained MAPPO checkpoint in LaneIQParallelEnv.",
    )
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--densities", type=str, default="low,medium,high")
    p.add_argument("--seeds", type=str, default="0,1,2,3,4")
    p.add_argument("--n-egos", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=1000)
    p.add_argument("--warmup-steps", type=int, default=50)
    p.add_argument("--out", type=Path, default=Path("results/week5_mappo_eval.csv"))
    return p


def run_one_episode(
    *, model, density: str, seed: int, n_egos: int,
    max_steps: int, warmup_steps: int,
) -> list[MAPPOAgentMetrics]:
    """Run one episode of `LaneIQParallelEnv` with the shared policy."""
    import math

    import traci

    from laneiq.env.actions import Action
    from laneiq.env.multi_agent_env import LaneIQParallelEnv

    env = LaneIQParallelEnv(
        density=density, n_egos=n_egos,
        max_steps=max_steps, warmup_steps=warmup_steps,
        seed=seed,
    )
    try:
        obs, _info = env.reset(seed=seed)

        # Per-agent accumulators.
        accs: dict[str, dict] = {
            aid: {
                "return": 0.0, "steps": 0,
                "speed_sum": 0.0, "lc": 0,
                "near_miss_steps": 0,
                "collided": False, "reached_end": False,
                "initial_pos": float(traci.vehicle.getLanePosition(aid)),
            }
            for aid in env.agents
        }

        terminated_global = False
        truncated_global = False
        while env.agents and not terminated_global and not truncated_global:
            actions: dict[str, int] = {}
            for aid in env.agents:
                a, _state = model.predict(obs[aid], deterministic=True)
                actions[aid] = int(a)

            obs_next, rewards, terms, truncs, infos = env.step(actions)

            for aid, r in rewards.items():
                acc = accs[aid]
                acc["return"] += float(r)
                acc["steps"] += 1
                if int(actions.get(aid, 0)) != int(Action.STAY):
                    acc["lc"] += 1
                # Raw speed + thw via TraCI snapshot (only if ego still present).
                if aid in traci.vehicle.getIDList():
                    speed = float(traci.vehicle.getSpeed(aid))
                    acc["speed_sum"] += speed
                    leader = traci.vehicle.getLeader(aid, 100.0)
                    if leader is not None and leader[0] != "":
                        gap = float(leader[1])
                        thw = gap / max(speed, 0.5)
                        if math.isfinite(thw) and thw < 1.5:
                            acc["near_miss_steps"] += 1
                if terms.get(aid, False):
                    reason = infos.get(aid, {}).get("ego_terminated_reason")
                    acc["collided"] = (reason == "collision")
                    acc["reached_end"] = (reason == "exited")

            obs = obs_next
            # End-of-episode: when all agents terminated/truncated AND none left.
            if all(terms.values()) or all(truncs.values()):
                terminated_global = True
            if not env.agents:
                break

        # Snapshot per-agent metrics.
        rows: list[MAPPOAgentMetrics] = []
        for aid, acc in accs.items():
            n = max(acc["steps"], 1)
            ep_len_s = acc["steps"] * 0.1
            mean_speed = acc["speed_sum"] / n
            lc_per_min = (acc["lc"] / ep_len_s) * 60.0 if ep_len_s > 0 else 0.0
            near_miss_rate = acc["near_miss_steps"] / n
            rows.append(MAPPOAgentMetrics(
                density=density, seed=seed, ego_id=aid, n_egos=n_egos,
                return_total=acc["return"],
                episode_length_steps=acc["steps"],
                collided=acc["collided"],
                reached_end=acc["reached_end"],
                mean_speed_mps=mean_speed,
                lane_changes=acc["lc"],
                lane_changes_per_min=lc_per_min,
                near_miss_rate=near_miss_rate,
            ))
        return rows
    finally:
        env.close()


def main(argv: list[str] | None = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    densities = [d.strip() for d in args.densities.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    from stable_baselines3 import PPO

    print(f"\nMAPPO eval: {args.checkpoint}")
    print(f"  densities = {densities}")
    print(f"  seeds     = {seeds}")
    print(f"  n_egos    = {args.n_egos}")
    print("-" * 70)

    model = PPO.load(args.checkpoint, device="cpu")

    all_rows: list[MAPPOAgentMetrics] = []
    for density in densities:
        for seed in seeds:
            t0 = time.monotonic()
            rows = run_one_episode(
                model=model, density=density, seed=seed,
                n_egos=args.n_egos,
                max_steps=args.max_steps, warmup_steps=args.warmup_steps,
            )
            elapsed = time.monotonic() - t0
            n_collisions = sum(1 for r in rows if r.collided)
            mean_ret = sum(r.return_total for r in rows) / max(len(rows), 1)
            mean_speed = sum(r.mean_speed_mps for r in rows) / max(len(rows), 1)
            print(
                f"  {density:<6} seed={seed}  "
                f"mean_ret={mean_ret:6.2f}  "
                f"collisions={n_collisions}/{len(rows)}  "
                f"speed={mean_speed:5.2f} m/s  "
                f"({elapsed:4.1f}s)",
                flush=True,
            )
            all_rows.extend(rows)

    # Write CSV.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(all_rows[0]).keys())
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in all_rows:
            writer.writerow(asdict(row))
    print(f"\n✓ wrote {len(all_rows)} rows → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
