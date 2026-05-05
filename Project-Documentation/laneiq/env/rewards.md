# `laneiq/env/rewards.py`

## What

Pure function `compute_reward(prev, action, nxt, *, config) -> (scalar, components)`
implementing the v1 6-component reward shape from the master plan. No
TraCI, no SUMO — just a function over a small `RewardState` slice the env
extracts in `step()`.

The deeper "why this shape, why these weights" (and the recruiter-grade
narrative around reward shaping) lives in
[`../../concepts/02-reward-shaping.md`](../../concepts/02-reward-shaping.md).

## Public API

```python
from laneiq.env.rewards import (
    RewardConfig,                # frozen dataclass; weights + thresholds
    DEFAULT_REWARD_CONFIG,       # singleton, equal to RewardConfig()
    RewardState,                 # frozen dataclass; ego state slice
    COMPONENT_KEYS,              # ordered tuple of the 6 component names
    compute_reward,              # (prev, action, nxt, *, config) -> (float, dict)
    reward_state_at_rest,        # convenience builder for placeholder states
)
```

### `RewardConfig`

Frozen dataclass. Defaults match the v1 plan; mutate per-experiment via
`dataclasses.replace(cfg, w_speed=...)`.

| Field | Default | What it does |
| --- | --- | --- |
| `w_speed` | `+1.0` | Reward for cruising at the lane speed limit |
| `w_lane_cost` | `-0.05` | Per-step cost of any non-STAY action |
| `w_jerk` | `-0.2` | Penalty proportional to `|Δaccel| / a_max` |
| `w_unsafe_gap` | `-0.5` | Penalty when time-headway < `unsafe_thw_s` |
| `w_collision` | `-50.0` | Terminal penalty (dominant) |
| `w_alive` | `+0.01` | Per-step bonus (anti-suicide) |
| `v_max` | `36.11` | Speed-norm denominator (m/s; lane speed limit) |
| `unsafe_thw_s` | `1.5` | THW threshold (s) below which gap is "unsafe" |
| `a_max_norm` | `4.5` | Jerk denominator (m/s²; ego decel) |

### `RewardState`

Frozen dataclass. Fields are in **raw units** (no normalization), extracted
by `LaneIQEnv.step()` directly from the TraCI snapshot:

| Field | Type | Source |
| --- | --- | --- |
| `speed` | `float` (m/s) | `vehicle.getSpeed(ego)` |
| `accel` | `float` (m/s²) | `vehicle.getAcceleration(ego)` |
| `front_gap` | `float` (m) | `vehicle.getLeader(ego)` gap, or `math.inf` |
| `front_thw` | `float` (s) | `front_gap / max(speed, eps)`, or `math.inf` |
| `collided` | `bool` | `ego in simulation.getCollidingVehiclesIDList()` |
| `lane_idx` | `int` | `vehicle.getLaneIndex(ego)` |

### `compute_reward(prev, action, nxt, *, config) -> (float, dict)`

Returns the scalar reward plus a per-component dict keyed by
`COMPONENT_KEYS`. The dict sums exactly to the scalar (no rounding).

The components dict is what the env logs to TensorBoard as
`reward/components/<key>` per step. **This is the most useful diagnostic**
when a policy converges to weird behavior — the per-component plot
identifies which term is being gamed.

## Reward formula

```
speed_term       = w_speed       · min(speed / v_max, 1.0)
lane_cost_term   = w_lane_cost   · I[action != STAY]
jerk_term        = w_jerk        · |Δaccel| / a_max_norm
unsafe_gap_term  = w_unsafe_gap  · I[front_thw < unsafe_thw_s]
collision_term   = w_collision   · I[collided]
alive_term       = w_alive

total = sum of all six
```

## Design notes

- **Reward operates on raw units.** Normalizing inside the reward would
  couple it to obs-builder constants. Keeping `RewardState` raw means
  reward edits don't ripple into obs code.
- **Returns components dict by design.** A scalar-only return loses the
  most useful debugging signal. TensorBoard plots per-component reward
  curves cost ~0 extra training time and tell you exactly which term is
  driving (or breaking) learning.
- **Speed term is *clipped*, not clamped.** `min(speed / v_max, 1.0)` is
  one-sided clipping — high values cap, low values stay raw. We don't
  reward going backward (which is impossible in SUMO anyway, but defensive).
- **Lane-cost is per-step, not per-distinct-change.** Even if the policy
  thrashes left-right-left, each action incurs the same `-0.05`. This is
  intentional — small enough that a justified change still pays for itself
  (speed gain >> 0.05), large enough that mindless thrashing converges to
  STAY.
- **Collision is `-50`, not `-100` or `-10`.** Calibrated so a single
  crash erases ~50 seconds of perfect cruising (`w_speed * 1.0 * 50 = 50`).
  Bigger penalty makes the policy too risk-averse (refuses any lane change);
  smaller and it learns to "occasionally crash for speed."
- **Alive bonus is small (0.01).** Just enough to make survival > suicide
  when other components are net-negative for a stretch. Without it, a
  policy stuck in heavy traffic might learn that an early collision
  shortens the episode and saves accumulated penalty.
- **Module imports `Action` from `actions.py`.** No circular dep risk —
  `actions.py` doesn't import from `rewards.py`. The dependency direction
  is rewards → actions → (nothing in env), reverse of the data flow.

## Invariants & gotchas

- **`prev` and `nxt` use raw units.** If you start passing in normalized
  values the speed and jerk terms silently produce garbage. The env's
  `step()` is the single place that constructs `RewardState` from TraCI;
  keep it that way.
- **First-step jerk handling is the env's responsibility.** `compute_reward`
  always computes `|nxt.accel - prev.accel|`. On the first step pass
  `prev = nxt` to get `jerk = 0`. The `reward_state_at_rest()` helper is
  a *neutral* starting state (all zeros + inf gaps), not a "free pass" —
  using it as `prev` produces non-zero jerk if the ego has any acceleration.
- **`v_max` must match the obs-builder constant.** If you tune `v_max`
  here, also tune the speed-normalization constant in `observations.py`.
  The two SHOULD be derived from one source-of-truth eventually; for now
  they're cross-referenced by docs and the test
  `test_reward_config_defaults_match_plan` documents the v1 value.
- **Collision is logged but doesn't terminate.** The env owns termination.
  `compute_reward` just emits a -50 contribution when `nxt.collided` is True.
  The env's `step()` returns `terminated=True` for that step, and the
  trainer treats the next state as terminal.
- **Defensive division guards.** `v_max=0` and `a_max_norm=0` short-circuit
  to 0 instead of dividing. Tests pin this. Means a misconfigured YAML
  produces a degraded reward, not a crashing run.

## Tests

`tests/test_rewards.py` (31 tests, all fast — no SUMO needed):

| Section | Count | Sample asserts |
| --- | --- | --- |
| Config + module invariants | 4 | defaults match plan; components sum to total |
| Speed | 4 | zero at rest, +1.0 at v_max, clipped above, uses `nxt` |
| Lane cost (parametrized) | 6 | STAY=0, LEFT/RIGHT=-0.05, raw int accepted |
| Jerk | 4 | zero when Δa=0; proportional; absolute value; first-step |
| Unsafe gap | 4 | fires below threshold, exact threshold safe, inf safe |
| Collision | 3 | -50 on collision; total dominated by -50 |
| Alive | 1 | always +0.01 regardless of state |
| Custom config | 3 | overrides; defensive division guards |
| Compound scenarios | 2 | smooth cruising > 0; lane change net-positive when gap exists |

Run them:

```bash
uv run pytest tests/test_rewards.py -v   # ~0.16 s (no SUMO)
```

## Last updated

2026-05-05 · `ee53ff5` (initial implementation; 31 tests green; lint clean)
