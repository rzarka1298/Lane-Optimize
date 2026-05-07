# `laneiq/agents/baselines/safe_rule_agent.py`

## What

Conservative rule-based policy: change lanes only when the target is
**both** clear (front + rear gaps above thresholds) **and** strictly
faster than the current lane by a margin. Otherwise STAY.

What this models: a cautious driver who prioritizes not-crashing over
going-fast. Expect: low collision rate, modest mean speed. Useful as the
**safety-only** baseline that bounds the safety axis of the Pareto plot.

## Public API

```python
from laneiq.agents.baselines.safe_rule_agent import SafeRuleAgent

agent = SafeRuleAgent(
    clear_front_gap_norm=0.30,    # ≥ 30 m to leader in target lane
    clear_rear_gap_norm=0.20,     # ≥ 20 m to follower in target lane
    faster_threshold_norm=0.05,   # ≥ ~1.8 m/s faster
)
action = agent.act(observation)
action = agent.act(observation, action_mask=mask)
```

## Algorithm

Per call:

1. Default decision = STAY.
2. Try **left** first (faster traffic tends to be left in our setup):
   - If left is reachable AND `lane_is_clear(left)` AND
     `is_faster(left)` → return CHANGE_LEFT.
3. Try **right**:
   - If right is reachable AND `lane_is_clear(right)` AND
     `is_faster(right)` → return CHANGE_RIGHT.
4. Otherwise return STAY.

Where:

- `lane_is_clear(lane)` = `front_gap[lane] ≥ clear_front_gap_norm`
  AND `rear_gap[lane] ≥ clear_rear_gap_norm`.
- `is_faster(lane)` = `projected_speed(lane) > current_speed + faster_threshold_norm`.

`projected_speed` and `_ego_lane_from_obs` are imported from
`greedy_agent.py` — same helpers, same logic.

## Why these thresholds

Defaults chosen for "instructor-grade caution" at the v_max=36.11 m/s
scale:

| Param | Default | Real-world | Why |
| --- | --- | --- | --- |
| `clear_front_gap_norm` | 0.30 | 30 m | ~1 sec leader gap at v_max |
| `clear_rear_gap_norm` | 0.20 | 20 m | ~0.5 sec follower gap at v_max |
| `faster_threshold_norm` | 0.05 | ~1.8 m/s | Don't change for a marginal gain |

These are constructor args; per-eval overrides land in
`configs/agent/safe_rule.yaml` when that file exists (Week 4-ish).

## Design notes

- **Stateless.** No constructor state beyond the three thresholds;
  `reset()` is a no-op.
- **Tries left first.** In our 3-lane scenario, slow vehicles default to
  lane 0 and aggressive vehicles default to higher lanes; left is more
  likely to find faster traffic. This is a heuristic ordering, not
  load-bearing for correctness — could try right first; expected to make
  small differences in eval numbers.
- **Reuses `_projected_speed` from greedy_agent.** The "what speed would
  I do in lane X?" calculation is identical; one source of truth.
- **`is_faster` uses strict `>` with margin.** Without the margin, the
  policy thrashes between two equally-fast lanes; with `>=` and zero
  margin, ties cause oscillation.

## Invariants & gotchas

- **Front gap check uses obs[front_offset]; rear uses obs[front_offset + 3].**
  Layout from `../../env/observations.md`. If you change the obs layout
  THIS file MUST be updated alongside.
- **The "current speed" in the comparison is the *projected* speed of
  the ego's lane**, not the raw `obs[0]`. So if there's a slow leader
  ahead, the comparison correctly weighs "I'm about to slow down to
  match this leader" vs "the other lane is open."
- **`is_faster` ignores the lane-cost from `rewards.py`.** It doesn't
  know about the `-0.05` per-step cost of changing. A formally optimal
  rule would compare projected reward over an N-step horizon; we keep
  it simple. This is part of why DQN/PPO should beat safe-rule on
  total reward.

## Tests

`tests/test_baselines.py`:

- `test_safe_stays_when_no_clear_faster_option` — default stay
- `test_safe_changes_left_when_left_clear_and_faster`
- `test_safe_does_not_change_left_when_left_front_too_close`
- `test_safe_does_not_change_left_when_left_rear_too_close`
- `test_safe_picks_right_when_only_right_is_safe_and_faster`
- `test_safe_stays_when_no_target_clear_and_faster` — both blocked
- `test_safe_respects_mask_blocking_left`
- `test_safe_validates_obs_shape`
- Plus `test_baseline_runs_one_episode[safe_rule]` (slow integration)

## Last updated

2026-05-07 · pending commit (initial implementation)
