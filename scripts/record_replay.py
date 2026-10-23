"""Record a replay JSON from one agent + env run.

Drives an agent through a single episode of LaneIQEnv and dumps every
tick in the same wire format the live demo streams (using
`backend.app.services.simulator.build_tick`, so payloads are
byte-identical). The backend can play these back via
`POST /sim/start_replay {replay}` — see backend/app/routes/sim.py.

Usage:

    uv run python -m scripts.record_replay \
        --agent ppo_v12_safety_cd_150k \
        --density medium --seed 42 \
        --out results/replays/ppo_medium_seed42.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from backend.app.services.policy_loader import resolve
from backend.app.services.simulator import build_tick
from laneiq.env.highway_env import LaneIQEnv


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", required=True,
                        help="checkpoint stem (e.g., ppo_v12_safety_cd_150k) "
                             "or baseline name (random/greedy/safe_rule)")
    parser.add_argument("--density",
                        choices=("low", "medium", "high", "gauntlet", "rush"),
                        default="medium")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--out", type=Path, required=True,
                        help="output JSON path")
    args = parser.parse_args(argv)

    print(f"loading agent={args.agent}")
    agent = resolve(args.agent)

    env = LaneIQEnv(
        density=args.density,
        max_steps=args.max_steps,
        warmup_steps=50,
        seed=args.seed,
        # Match the backend: the rush scenario auto-loops in the live
        # demo, so a recorded rush replay should too (otherwise the
        # replay ends after one lap and looks shorter than the live run).
        loop=(args.density == "rush"),
    )
    obs, _info = env.reset(seed=args.seed)

    ticks: list[dict] = []
    step = 0
    sim_time_s = 0.0
    terminated = truncated = False
    while not (terminated or truncated):
        mask = env.action_masks()
        action = agent.act(obs, action_mask=mask)
        obs, _reward, terminated, truncated, info = env.step(action)
        step += 1
        sim_time_s += 0.1
        tick = build_tick(
            step=step, sim_time_s=sim_time_s,
            action=int(action), mask=mask, obs=obs, info=info,
            terminated=terminated, truncated=truncated,
        )
        ticks.append(tick.model_dump(exclude_none=True))

    env.close()

    payload = {
        "agent": args.agent,
        "density": args.density,
        "seed": args.seed,
        "n_ticks": len(ticks),
        "terminated": bool(terminated),
        "truncated": bool(truncated),
        "ticks": ticks,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload))

    end_kind = "terminated" if terminated else "truncated"
    print(f"✓ {end_kind} after {step} steps; saved → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
