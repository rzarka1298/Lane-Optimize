# `laneiq/agents/baselines/` — three rule-based agents

## Purpose

The PRD's required baselines: random, greedy, and safe-rule. They turn
abstract reward numbers into a story — "DQN gets +5 reward" only matters
when juxtaposed with random's +1, greedy's +3, safe-rule's +4. Without
baselines the project reads as a toy.

## Status

✅ **Complete.** All three agents implement the `Policy` protocol from
`../base.py`; integration-tested against `LaneIQEnv`; reproducible via
seeding for `RandomAgent`.

## Files in this folder

- **[`random_agent.md`](random_agent.md)** — `RandomAgent`. Uniform over
  valid actions. The floor.
- **[`greedy_agent.md`](greedy_agent.md)** — `GreedyAgent`. Picks lane
  with highest projected speed. Bounds the speed axis of the Pareto plot.
- **[`safe_rule_agent.md`](safe_rule_agent.md)** — `SafeRuleAgent`.
  Changes only when target is clear *and* faster. Bounds the safety axis.

## Why these three

The Week-4 hero plot is the **safety-vs-speed Pareto frontier**: x-axis
collision rate, y-axis mean speed, one point per agent × density. The
three baselines are calibrated to occupy distinct corners of that plot:

| Agent | Behavior | Expected position on Pareto plot |
| --- | --- | --- |
| `RandomAgent` | uniform actions | bottom-right (slow + crashy) |
| `GreedyAgent` | speed-only optimization | top-right (fast + crashy) |
| `SafeRuleAgent` | safety-first | bottom-left (slow + safe) |
| `DQN` (Task #12) | learned trade-off | **top-left target** (fast + safe) |
| `PPO` (Week 4) | learned trade-off | **top-left target** (fast + safe) |

If the trained agents land in the top-left, the project demonstrates that
RL learns the Pareto frontier rather than picking one extreme. If they
land elsewhere, the per-component reward plot tells us which weight to
tune.

## Folder-wide invariants

- **All three agents are pure functions of `(observation, action_mask)`** —
  no internal state beyond `RandomAgent`'s RNG. Tests can assert outputs
  given hand-crafted observations without spinning up SUMO.
- **All three respect explicit `action_mask`** when provided. Falling
  back to `valid_actions_mask(obs)` when omitted.
- **Tie-break in favor of STAY** (greedy and safe-rule). Prevents
  thrashing when multiple lanes look equally attractive.
- **Greedy and safe-rule share helpers** (`_ego_lane_from_obs`,
  `_projected_speed`) defined in `greedy_agent.py`. If you change the
  observation layout, those helpers MUST be updated to match — the
  obs-builder doc (`../../env/observations.md`) and these helpers are
  load-bearing in pair.

## Tests

`tests/test_baselines.py` — 26 tests after parametrize expansion:

| Section | Count | What |
| --- | --- | --- |
| Policy protocol | 3 (parametrized) | each baseline `isinstance(agent, Policy)` |
| RandomAgent | 6 | range, mask, lane-edge mask, uniformity, seeding |
| GreedyAgent | 6 | tie-break, picks open lane, mask, shape validation |
| SafeRuleAgent | 6 | stays default, changes when clear+faster, gap thresholds, mask |
| Integration (slow + sumo) | 3 (parametrized) | each baseline drives one episode end-to-end |

Run them:

```bash
uv run pytest tests/test_baselines.py -v -m "not slow"   # 23 fast
uv run pytest tests/test_baselines.py -v -m "slow"       # 3 SUMO tests
```

## Last updated

2026-05-07 · `e4f5fa6` (initial implementation; 26 tests green)
