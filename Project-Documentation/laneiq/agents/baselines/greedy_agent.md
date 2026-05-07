# `laneiq/agents/baselines/greedy_agent.py`

## What

Greedy "highest-projected-speed" policy. For each reachable lane,
estimates what the ego's cruising speed would be if it were behind that
lane's leader; picks the lane with the highest projection. Ignores
safety entirely — no gap check, no follower check.

What this models: a myopic optimizer that only cares about going fast
right now. Expect: high mean speed, high collision rate. Useful as the
**speed-only** baseline that bounds the speed axis of the Pareto plot.

## Public API

```python
from laneiq.agents.baselines.greedy_agent import GreedyAgent

agent = GreedyAgent()                        # stateless; no constructor args
action = agent.act(observation)              # mask recovered from obs
action = agent.act(observation, action_mask=mask)
```

Plus two module-level helpers reused by `safe_rule_agent.py`:

```python
from laneiq.agents.baselines.greedy_agent import (
    _ego_lane_from_obs,    # obs → int lane in {0, 1, 2}
    _projected_speed,      # (obs, lane, ego_speed_norm) → float
)
```

## Algorithm

Per call:

1. Recover `ego_lane` from `obs[1]` (= lane / 2).
2. Compute `score[lane]` = projected speed if ego were in that lane:
   - If lane has no leader (`vc == -1`) → `1.0` (open lane = max).
   - Else → `ego_speed_norm + dv_to_leader`.
3. Compare `score[ego_lane]`, `score[ego_lane + 1]` (if left valid),
   `score[ego_lane - 1]` (if right valid).
4. Pick the highest. **Tie-break to STAY** (so an open current lane plus
   an open adjacent lane doesn't trigger thrashing).

## Design notes

- **Stateless.** `__init__` takes no args; `reset()` is a no-op.
- **Tie-break in favor of STAY** is load-bearing. Without it, when all
  three lanes are equally open (very common at low density), the policy
  oscillates between LEFT/STAY/RIGHT each step.
- **`_projected_speed` returns 1.0 for open lanes**, not `ego_speed_norm`.
  Without this, an open lane reads as "stays at current speed" — same
  as a perfectly-paced leader — and the agent never overtakes.
- **Helpers shared with `safe_rule_agent.py`.** The two agents have
  similar lane-comparison logic; pulling it out avoids duplication.
  Underscore-prefixed because it's intra-package; not for external import.

## Invariants & gotchas

- **Greedy ignores safety by design.** Don't add a gap check here — the
  whole point is to be the unsafe-but-fast bound on the Pareto plot.
  If you want safety, that's `safe_rule_agent.py`.
- **`_ego_lane_from_obs` clamps to [0, 2]** so a mid-lane-change reading
  of obs[1] doesn't produce out-of-range indices.
- **Validates obs shape** (raises `ValueError` if not (25,)). Catches
  observation-layout drift early.

## Tests

`tests/test_baselines.py`:

- `test_greedy_stays_when_all_lanes_open` — tie-break correctness
- `test_greedy_picks_left_when_left_lane_open_and_current_blocked`
- `test_greedy_picks_right_when_only_right_is_better` — exhaustively blocked
- `test_greedy_respects_mask_blocking_left`
- `test_greedy_does_not_change_when_current_lane_is_open`
- `test_greedy_validates_obs_shape`
- Plus `test_baseline_runs_one_episode[greedy]` (slow integration)

## Last updated

2026-05-07 · `e4f5fa6` (initial implementation)
