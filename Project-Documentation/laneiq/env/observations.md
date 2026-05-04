# `laneiq/env/observations.py`

## What

Translates the live TraCI state of the ego vehicle into a fixed-length
**25-dim float32 vector clipped to [-1, 1]** — the Markov state the RL
policy sees. Pure function for tests and ad-hoc scripts; stateful wrapper
for the env's hot loop.

The deeper "why this shape, why these features" lives in
[`../../concepts/01-sumo-traffic-model.md`](../../concepts/01-sumo-traffic-model.md);
this doc is the per-file reference: schema, public API, design notes,
file-specific gotchas, tests.

## Public API

```python
from laneiq.env.observations import (
    OBS_DIM,                 # 25
    OBS_NAMES,               # tuple[str, ...] of length 25
    OBSERVATION_SPEC,        # ObservationSpec singleton (gym.spaces.Box, dim, names)
    DEFAULT_OBSERVATION_CONFIG,
    ObservationConfig,       # frozen dataclass — tunable constants
    ObservationSpec,         # frozen dataclass — Box + names + dim
    build_observation,       # pure function
    ego_present,             # bool helper
    ObservationBuilder,      # stateful wrapper
)
```

### `build_observation(traci_conn, ego_id, *, config, v_max_lane) -> np.ndarray`

Pure. Reads TraCI for `ego_id`, returns shape-(25,) float32 array. Raises
`KeyError` if `ego_id` is not in the current vehicle list — callers
(`LaneIQEnv.reset/step`) must guarantee presence.

`traci_conn` is duck-typed. In production it's the `traci` module; in tests
it's a `Mock` that responds to the calls listed below.

### `class ObservationBuilder`

Stateful wrapper that the env holds across episodes. Identical wire format
to `build_observation`; this is purely a perf-amortization layer.

```python
builder = ObservationBuilder(traci_conn, config=...)
builder.on_episode_start(ego_id)   # caches v_max_lane
obs = builder.build(ego_id)        # called per env.step()
builder.on_episode_end()           # clears cache
```

The `on_vehicle_seen` hook is a no-op today; it exists so Week 3 profiling
(Task #5) can flip on TraCI subscriptions without touching the env code.

### `class ObservationConfig` (frozen)

| Field | Default | Purpose |
| --- | --- | --- |
| `max_gap_m` | `100.0` | Search radius for `getLeader`/`getNeighbors`; gap normalizer |
| `max_density_count` | `20` | Vehicles in window → density 1.0 |
| `accel_norm` | `4.5` | Ego decel; sets dim[2] scale |
| `thw_clip_s` | `5.0` | Cap on time-headway; >5s reads as "safe" |
| `speed_eps` | `0.5` | Avoid div-by-zero in THW when ego stopped |
| `edge_id` | `"e1"` | For `lane.getMaxSpeed` lookup |
| `lane_width_m` | `3.2` | SUMO's default lane width |

## Observation layout (25 dims)

| Idx | Name | Source / formula | Missing sentinel |
| --- | --- | --- | --- |
| 0 | `ego_speed` | `getSpeed / v_max_lane` | n/a |
| 1 | `ego_lane_idx` | `getLaneIndex / 2.0` | clamp |
| 2 | `ego_accel` | `getAcceleration / 4.5` | 0.0 |
| 3 | `ego_lat_pos` | `getLateralLanePosition / (lane_width/2)` | 0.0 |
| 4–21 | 6× `(gap, dv, vclass)` | per-lane front/rear neighbors | `(1.0, 0.0, -1.0)` |
| 22 | `local_density` | `count_within_window / 20` | 0.0 |
| 23 | `lane_validity_packed` | `(left·2 + right) / 3` | always defined |
| 24 | `ego_thw_to_leader` | `min(gap/v, 5.0) / 5.0` | 1.0 (no leader) |

Neighbor slot indexing: `idx = 4 + lane*6 + slot*3`, where `lane ∈ {0,1,2}`
and `slot ∈ {0=front, 1=rear}`.

vClass scalar bucket: `{aggressive=0, normal=1, slow=2, ego=3}` →
`(id-1)/2 ∈ {-0.5, 0.0, 0.5, 1.0}`. Missing → `-1.0`.

Lane validity packed: `0` = none, `1/3` = right only, `2/3` = left only,
`1.0` = both. Recovered in `actions.py` via `pack = round(obs[23] * 3);
left = (pack >> 1) & 1; right = pack & 1`.

## TraCI calls per `build_observation`

| Call | Count per step (worst case) |
| --- | --- |
| `vehicle.getIDList()` | 2 (once for `ego_present`, once for density) |
| `vehicle.getSpeed(id)` | 1 + ≤6 (ego + per-neighbor) |
| `vehicle.getLaneIndex(ego)` | 1 |
| `vehicle.getAcceleration(ego)` | 1 |
| `vehicle.getLateralLanePosition(ego)` | 1 |
| `vehicle.getLanePosition(id)` | 1 + ~20 (ego + density window) |
| `vehicle.getTypeID(id)` | ≤6 (per-neighbor) |
| `vehicle.getLeader(ego, max_gap)` | 1 |
| `vehicle.getFollower(ego, max_gap)` | 1 |
| `vehicle.getNeighbors(ego, mode)` | 4 |
| `lane.getMaxSpeed(...)` | 1 (cached by `ObservationBuilder`) |

~25–35 calls per step in the cold path. `ObservationBuilder` will drop
this to ~7 once TraCI subscriptions are turned on (Week 3 profiling
decides).

### `getNeighbors` mode bits

- bit0 = left(1) / right(0)
- bit1 = leader(1) / follower(0)

So: `0b011 = 3` left-leader · `0b001 = 1` left-follower ·
`0b010 = 2` right-leader · `0b000 = 0` right-follower.

## Design notes

- **Speed-norm = lane speed limit (read once)**, not a hardcoded literal.
  Avoids the silent coupling warned about in `../../sumo_scenarios/network.md`
  (change the lane speed → obs auto-rescales).
- **vClass scalar bucket > one-hot**. 6 neighbors × 3 dims of one-hot would
  burn 18 dims for an ordinal signal a 2-layer MLP rebuilds for free.
- **Time-headway only for the ego's own leader**, not per-neighbor. THW
  matters for the ego's next-step survival; for non-leader neighbors,
  `(gap, dv)` already encodes everything THW would. Per-neighbor THW would
  6× the feature for ~zero policy benefit.
- **Lane validity bit-packed** (not two separate floats). Saves a dim,
  reconstruction in `actions.py` is one multiply.
- **Stateless function + stateful builder.** Tests stay trivial (mock,
  call, assert); the env still gets the per-episode caching win. Same
  wire format means the two can be swapped freely.
- **No `import traci` at module top.** Caller passes the connection in.
  Lets unit tests mock; lets `import laneiq.env.observations` succeed in
  a fresh clone before `uv sync --extra sumo`.

## Invariants & gotchas

- **The 25-dim layout is load-bearing.** `OBS_NAMES` and the index
  arithmetic in `build_observation` must stay in lockstep. The
  `assert len(OBS_NAMES) == OBS_DIM` at module import catches the most
  common slip.
- **Side neighbors fill into their actual lane index, not relative to ego.**
  A left-leader when ego is in lane 1 fills the lane-2 front slot. The
  policy reads "no left lane" via the lane-validity flag (idx 23), not
  via the missing sentinel.
- **`getLeader` returns `None` OR `("", -1.0)`** depending on SUMO version.
  We handle both. Same for `getFollower`.
- **`getNeighbors` returns a tuple of `(vehID, gap)` with positive gap**
  for both leaders and followers (gap is unsigned distance). The
  `ego_pos ± gap` math in the mock factory mirrors this.
- **Race window between `getNeighbors` and `getSpeed`/`getTypeID`.** A
  background vehicle can exit between the two calls. `_safe_neighbor_attrs`
  catches the resulting TraCI exception and the slot falls back to the
  missing sentinel for that step. Acceptable — it's a one-tick blip.
- **`v_max_lane = 0` defensive fallback.** If `lane.getMaxSpeed` returns 0
  or raises, we fall back to `_DEFAULT_V_MAX_LANE = 36.11` (matches the
  scenario default). Never crash an episode over a TraCI hiccup.
- **`ego_lane` clamped to [0, 2]** because SUMO can return -1 mid-lane-change
  in some edge cases.

## Tests

`tests/test_observations.py` (17 tests after pytest parametrize expansion):

| Test | Asserts |
| --- | --- |
| `test_observation_spec_shape_and_bounds` | `Box(-1, 1, (25,), float32)`; `len(OBS_NAMES) == 25` |
| `test_build_observation_returns_correct_dtype_shape` | shape (25,), dtype float32, all finite |
| `test_ego_state_normalization` | dims [0..3] reflect the normalized values exactly |
| `test_neighbor_missing_sentinel` | every empty slot reads `(1.0, 0.0, -1.0)` |
| `test_neighbor_filled_slot_same_lane_front` | leader at (gap=50, dv=-5, aggressive) → dims [10..12] |
| `test_neighbor_filled_slot_left_leader` | left-leader fills the lane-2 front slot when ego in lane 1 |
| `test_lane_validity_packed_value` (×3 params) | lane 0 → 2/3, lane 1 → 1.0, lane 2 → 1/3 |
| `test_local_density_clipped` | 30 in-window vehicles → idx 22 == 1.0 (not 1.5) |
| `test_thw_normalization_and_clip` (×3 params) | covers clipped, in-range, and no-leader cases |
| `test_clip_range_invariant` | adversarial inputs still yield [-1, 1] dims |
| `test_missing_ego_raises_keyerror` | `KeyError` if ego not in vehicle list |
| `test_observation_builder_caches_v_max_lane` | `on_episode_start` caches; subsequent `build()` calls don't re-query |
| `test_observation_against_running_sumo` (`@slow @sumo`) | end-to-end against real SUMO |

Run them:

```bash
uv run pytest tests/test_observations.py -v -m "not slow"  # 16 fast
uv run pytest tests/test_observations.py -v -m "slow"      # 1 SUMO test
uv run pytest tests/test_observations.py -v                # all
```

## Last updated

2026-05-03 · `0477f0d` (initial implementation; 17 tests green; lint clean)
