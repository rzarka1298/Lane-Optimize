# `laneiq/env/actions.py`

## What

Three discrete actions plus the glue to apply them, mask invalid ones, and
disable SUMO's internal lane-change override on the ego. This is the
opposite end of the policy loop from `observations.py`: observation comes
in → policy chooses an action → this module turns it into a TraCI call.

## Public API

```python
from laneiq.env.actions import (
    Action,                       # IntEnum: STAY=0, CHANGE_LEFT=1, CHANGE_RIGHT=2
    N_ACTIONS,                    # 3
    ACTION_SPACE,                 # gym.spaces.Discrete(3)
    ACTION_NAMES,                 # ("stay", "change_left", "change_right")
    disable_sumo_lane_change,     # called once per episode in env.reset()
    apply_action,                 # called per env.step()
    valid_actions_mask,           # for the env / DQN action filter
    mask_invalid_q_values,        # for the DQN's argmax path
)
```

### `Action` enum

```python
class Action(IntEnum):
    STAY = 0
    CHANGE_LEFT = 1   # increases lane_idx (0 → 1, 1 → 2)
    CHANGE_RIGHT = 2  # decreases lane_idx (2 → 1, 1 → 0)
```

Lane indexing follows SUMO: lane 0 is rightmost, lane 2 is leftmost. So
"CHANGE_LEFT" raises the integer index. Counterintuitive at first; consistent
with every other reference in the codebase.

### `disable_sumo_lane_change(traci_conn, ego_id)`

Calls `traci.vehicle.setLaneChangeMode(ego_id, 0)`. Mode `0` clears every
internal SUMO lane-change motivation AND every safety override:

| Bits  | Cleared motivation        |
| ----- | ------------------------- |
| 0–1   | strategic (route)         |
| 2–3   | cooperative               |
| 4–5   | speed-gain                |
| 6–7   | keep-right                |
| 8–9   | sublane                   |
| 10–11 | TraCI-requested override  |

After this call, SUMO will execute exactly the lane changes the policy
requests via `apply_action()`, including unsafe ones. **The ego CAN crash
from a bad policy decision.** That's intentional — the collision penalty
in `rewards.py` is what teaches the policy not to.

This is one of the load-bearing details a recruiter notices. See
`../../concepts/01-sumo-traffic-model.md` for the full rationale.

### `apply_action(traci_conn, ego_id, action, *, duration_s=1.0)`

| Action          | Effect                                               |
| --------------- | ---------------------------------------------------- |
| `STAY`          | No-op. Ego continues under Krauss (longitudinal) control. |
| `CHANGE_LEFT`   | `traci.vehicle.changeLane(ego, current+1, duration_s)` |
| `CHANGE_RIGHT`  | `traci.vehicle.changeLane(ego, current-1, duration_s)` |

If the target lane doesn't exist (e.g., `CHANGE_LEFT` while in lane 2),
the function is a **silent no-op**. The action mask should have prevented
this; the silent-no-op is defense in depth, not the primary guard.

`duration_s` is the time SUMO uses to derive lateral acceleration for the
maneuver. `1.0 s` ≈ a real driver's lane change. Lower = more aggressive.

### `valid_actions_mask(observation: np.ndarray) -> np.ndarray`

Returns a shape-(3,) bool array. `STAY` is always valid; `CHANGE_LEFT`
and `CHANGE_RIGHT` are recovered from `observation[23]` (the packed
lane-validity scalar set by `build_observation`).

Decode logic:

```python
pack = round(observation[23] * 3)        # ∈ {0, 1, 2, 3}
left_ok  = bool((pack >> 1) & 1)
right_ok = bool(pack & 1)
return np.array([True, left_ok, right_ok], dtype=bool)
```

Round-then-clamp absorbs the small float drift `np.clip` can leave
behind in the observation builder.

### `mask_invalid_q_values(q_values, mask) -> np.ndarray`

Returns Q-values with positions where `mask is False` set to `-np.inf`.
Does NOT mutate the input. Used by DQN's ε-greedy / argmax path so an
invalid action is never chosen.

For PPO: use `MaskablePPO` from `sb3-contrib` instead — pass the mask
into the env's `info` dict and let the trainer handle it.

## Design notes

- **`STAY` is always valid.** Even at edge boundaries, the policy can
  choose to do nothing; the env doesn't need to handle a "no valid
  action" case.
- **Silent no-op on invalid lane targets.** Crashing on a bad action
  would propagate every action-mask bug into a fatal error. The
  combination of (mask in DQN/PPO) + (silent no-op in apply_action) is
  belt-and-suspenders.
- **`apply_action` reads `getLaneIndex` per call.** Could plumb the
  cached lane idx from the observation, but the freshness matters: between
  obs and action the ego may have moved laterally a tiny amount, and we
  always want to base the target on the *current* lane. The TraCI
  round-trip is one call — cheap.
- **Lane bounds hardcoded to [0, 2]** for the 3-lane scenario. If we ever
  ship a multi-edge network with variable lane counts, lift these into
  `ObservationConfig` and read them from `traci.edge.getLaneNumber(...)`
  at episode start.
- **`mask_invalid_q_values` returns float64** even if input is float32.
  `-np.inf` is fine in either; float64 keeps argmax stable on degenerate
  inputs (all Qs equal, etc.) without forcing a cast in callers.

## Invariants & gotchas

- **Calling order in `LaneIQEnv.reset`** matters:
  1. `traci.start(...)`
  2. warmup steps to populate background traffic
  3. `traci.vehicle.add(ego_id, ...)`
  4. one `simulationStep()` so the ego appears in `getIDList`
  5. **`disable_sumo_lane_change(traci, ego_id)`** ← here, not before
  6. first `build_observation`
  Calling `setLaneChangeMode` before the ego exists raises a TraCI error.
- **`apply_action` accepts a raw int.** This is convenience for the env
  passing through whatever the policy returns. `Action(int(action))` will
  raise `ValueError` if the int is outside {0, 1, 2}.
- **The action mask depends on the obs encoding** of `obs[23]`. If the
  `lane_validity_packed` formula in `observations.py` changes,
  `valid_actions_mask` MUST change to match — and ideally the formula
  should be a shared constant. For now they're cross-referenced in docs.
- **`changeLane` is asynchronous in SUMO**. After issuing
  `changeLane(ego, target, 1.0)`, the lane index will not change for the
  next ~10 sim-steps while the lateral move plays out. The integration
  test confirms a CHANGE_LEFT from lane 1 lands in lane 2 within ~40
  sim-steps; the env's per-step loop should treat in-progress moves as
  "still ongoing" via the `getLaneIndex` value.

## Tests

`tests/test_actions.py` (23 tests after parametrize expansion):

| Test | What it asserts |
| --- | --- |
| `test_action_enum_values` | enum values + `N_ACTIONS` + `ACTION_NAMES` |
| `test_action_space_is_discrete_3` | `gym.spaces.Discrete(3)` |
| `test_disable_sumo_lane_change_calls_set_mode_zero` | calls `setLaneChangeMode(ego, 0)` once, no other calls |
| `test_apply_action_stay_is_noop` | STAY makes zero TraCI calls |
| `test_apply_action_valid_transitions` (×4) | (lane, action) → expected `changeLane(target, dur)` |
| `test_apply_action_accepts_raw_int` | passing `1` works as `Action.CHANGE_LEFT` |
| `test_apply_action_custom_duration` | `duration_s` propagates through |
| `test_apply_action_invalid_target_is_silent_noop` (×2) | edge-of-network actions don't call `changeLane` |
| `test_valid_actions_mask_values` (×4) | packed values map to the expected bool triples |
| `test_valid_actions_mask_rejects_wrong_shape` | non-(25,) input raises `ValueError` |
| `test_valid_actions_mask_robust_to_float_drift` | ±1e-5 of canonical packed value still decodes correctly |
| `test_mask_invalid_q_values_basic` | -inf at masked positions, real Qs elsewhere |
| `test_mask_invalid_q_values_does_not_mutate_input` | input array untouched |
| `test_mask_invalid_q_values_argmax_safety` | argmax never picks an invalid action |
| `test_mask_invalid_q_values_shape_mismatch_raises` | mismatched shapes raise loudly |
| `test_apply_action_changes_lane_in_running_sumo` (`@slow @sumo`) | end-to-end: SUMO + ego + CHANGE_LEFT → ego at lane 2 within 4 s |

Run them:

```bash
uv run pytest tests/test_actions.py -v -m "not slow"  # 22 fast
uv run pytest tests/test_actions.py -v -m "slow"      # 1 SUMO test
```

## Last updated

2026-05-04 · `b4d7fd1` (initial implementation; 23 tests green; lint clean)
