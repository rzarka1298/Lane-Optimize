# `laneiq/env/highway_env.py`

## What

`LaneIQEnv` — the integrated `gymnasium.Env` that stitches `sumo_runtime`,
`observations`, `actions`, and `rewards` into a single class the RL trainer
can drive directly. This is the file every agent in the project (DQN, PPO,
MAPPO, baselines) consumes.

Passes `gymnasium.utils.env_checker.check_env` — verified by a slow test.

## Public API

```python
from laneiq.env.highway_env import LaneIQEnv

env = LaneIQEnv(
    density="medium",          # "low" | "medium" | "high"
    max_steps=1000,
    warmup_steps=50,
    gui=False,
    ego_depart_lane=1,
    ego_depart_speed=20.0,
    seed=42,
    obs_config=...,            # ObservationConfig override (optional)
    reward_config=...,         # RewardConfig override (optional)
)

obs, info = env.reset(seed=42)            # restart SUMO + warmup + add ego
mask = env.action_masks()                 # for MaskablePPO / DQN argmax
obs, reward, terminated, truncated, info = env.step(action)
env.close()                               # idempotent
```

### `info` dict per step

| Key | Type | Meaning |
| --- | --- | --- |
| `reward_components` | `dict[str, float]` | Per-term contribution; sums to `reward`. Logged to TensorBoard as `reward/components/<key>`. |
| `step_count` | `int` | Steps elapsed in the current episode. |
| `ego_terminated_reason` | `str` | Only on terminal steps: `"collision"` or `"exited"`. |

### Constructor parameters

| Param | Default | Meaning |
| --- | --- | --- |
| `scenario_path` | `DEFAULT_SUMOCFG` | Path to a `.sumocfg`. Override for alternate scenarios. |
| `density` | `"medium"` | Picks the route file (`routes_<density>.rou.xml`). |
| `max_steps` | `1000` | After this many `step()` calls, `truncated=True` (= 100 sim seconds at 10 Hz). |
| `warmup_steps` | `50` | Steps to advance before adding the ego (= 5 sim seconds; populates background traffic). |
| `gui` | `False` | If True, launches `sumo-gui` instead of headless. Slow; for debugging. |
| `ego_depart_lane` | `1` | Lane index where ego spawns (0 = right, 2 = left). |
| `ego_depart_speed` | `20.0` | m/s. Below `v_max` so first action has both directions free. |
| `seed` | `None` | Default seed. Per-episode `reset(seed=...)` overrides. |
| `obs_config` | `DEFAULT_OBSERVATION_CONFIG` | Frozen `ObservationConfig`. |
| `reward_config` | `DEFAULT_REWARD_CONFIG` | Frozen `RewardConfig`. |

## Episode lifecycle

```
__init__:
  validate config; do NOT start SUMO

reset(seed):
  close prior SUMO connection (guarded; idempotent)
  traci.start(build_argv(seed=seed, route_file=...density))
  for _ in range(warmup_steps): traci.simulationStep()
  traci.vehicle.add("ego_0", route="r_through", typeID="ego", ...)
  traci.simulationStep()                  # ego appears in getIDList
  disable_sumo_lane_change(traci, "ego_0") # policy is sole LC decider
  builder = ObservationBuilder(traci); builder.on_episode_start("ego_0")
  obs = builder.build("ego_0")
  prev_reward_state = extract from TraCI snapshot
  return (obs, {})

step(action):
  apply_action(traci, "ego_0", action)
  traci.simulationStep()
  if "ego_0" in getCollidingVehiclesIDList():
    terminate with collided=True
  elif "ego_0" not in getIDList():
    terminate (off-end of road) with collided=False
  else:
    nxt = extract reward state; reward, components = compute_reward(prev, action, nxt)
    obs = builder.build("ego_0")
    truncated = step_count >= max_steps
    return (obs, reward, False, truncated, {reward_components, step_count})
```

## Three terminal conditions

| Condition | `terminated` | `truncated` | Notes |
| --- | --- | --- | --- |
| Ego in `getCollidingVehiclesIDList()` | True | False | Collision penalty fires; `info["ego_terminated_reason"] = "collision"` |
| Ego no longer in `getIDList()` (no collision) | True | False | Reached end of 1 km road; `"ego_terminated_reason" = "exited"` |
| `step_count >= max_steps` | False | True | Time-limit only; world continues |

The `terminated` vs `truncated` split matters because RL trainers treat
them differently: `terminated` → no future bootstrap (Q_target = r);
`truncated` → bootstrap (Q_target = r + γ·Q(s')). Mixing them up is a
classic subtle RL bug.

## Design notes

- **`__init__` does not start SUMO.** SUMO startup costs ~150 ms; we defer
  it to `reset()` so constructing an env in tests, eval scripts, or vector
  wrappers is free. The `_sumo_running` flag tracks state.
- **TraCI label `"laneiq_env"`** — distinct from `sumo_session`'s `"laneiq"`
  so a process running both (e.g., a test that uses `sumo_session` AND
  constructs an env) doesn't collide.
- **Single-instance constraint.** TraCI's connection state is module-level;
  two `LaneIQEnv` instances in the same process will fight over it. For
  parallel rollouts, use SB3's `SubprocVecEnv` (one SUMO per subprocess).
- **`apply_action` wrapped in try/except.** TraCI raises if the ego was
  removed between `getIDList` checks and `changeLane`. The post-step
  ego-disappeared check handles the actual termination cleanly; the
  try/except just prevents a noisy stack trace.
- **Reward state slice extracted from raw TraCI**, not from the obs.
  Per the rewards-design contract, reward operates on raw units — this
  keeps it decoupled from any future obs-builder edits.
- **`action_masks()` exposed for MaskablePPO.** SB3-contrib looks for this
  exact method name. Returns the validity mask from the cached last obs;
  raises if called before `reset()`.

## Invariants & gotchas

- **`reset()` calls `super().reset(seed=seed)`** to feed the seed into
  Gymnasium's RNG infrastructure. Even though we don't use `self.np_random`
  (SUMO owns the randomness), the env_checker requires this call.
- **`render()` returns `None`** with `"render_modes": []`. Real
  visualization is the FastAPI/React dashboard (Week 5+). The env_checker
  is told to skip the render check via `skip_render_check=True`.
- **First step's `prev`** is the post-warmup ego state, not a placeholder.
  Means the very first jerk term may be non-zero if the ego accelerated
  during the spawn step. This is correct behavior.
- **Closing twice is safe.** `_close_sumo_safely` checks the running flag
  and uses `contextlib.suppress` around `traci.close`.
- **Multiple resets work.** Each `reset()` tears down the prior connection
  before opening a new one. Tested with `test_multiple_resets_work`.
- **`getCollidingVehiclesIDList`** is the authoritative collision signal.
  Don't try to detect collisions from gap=0 in the observation — SUMO can
  resolve a "collision" via teleport in some configs (we disable that
  with `time-to-teleport=-1`, but trust the explicit API anyway).

## Tests

`tests/test_highway_env.py` (15 tests):

| Section | Test | Speed |
| --- | --- | --- |
| Construction | `test_construction_does_not_start_sumo` | fast |
| Construction | `test_observation_and_action_spaces` | fast |
| Construction | `test_invalid_density_raises` | fast |
| Lifecycle | `test_step_before_reset_raises` | fast |
| Lifecycle | `test_action_masks_before_reset_raises` | fast |
| Lifecycle | `test_close_idempotent_before_reset` | fast |
| Reset | `test_reset_returns_obs_and_info` | slow + sumo |
| Step | `test_step_returns_5_tuple_with_in_range_obs` | slow + sumo |
| Step | `test_step_info_has_reward_components` | slow + sumo |
| Step | `test_episode_truncates_at_max_steps` | slow + sumo |
| Step | `test_action_masks_after_reset_is_valid` | slow + sumo |
| Lifecycle | `test_multiple_resets_work` | slow + sumo |
| Determinism | `test_seeding_is_reproducible` | slow + sumo |
| Lifecycle | `test_close_after_reset_idempotent` | slow + sumo |
| **Contract** | **`test_env_passes_gymnasium_check_env`** | slow + sumo |

Last test runs `gymnasium.utils.env_checker.check_env(env)` — the
gold-standard contract validator. It calls `reset/step` in unusual
orders, checks observation/action types, verifies the 5-tuple shape, etc.
Passing this is the proof that `LaneIQEnv` is a well-formed Gymnasium env.

Run them:

```bash
uv run pytest tests/test_highway_env.py -v -m "not slow"   # 6 fast (~0.1 s)
uv run pytest tests/test_highway_env.py -v -m "slow"       # 9 slow (~20 s)
```

## Last updated

2026-05-05 · `7c6a6e9` (initial implementation; 15 tests green; lint clean;
env_checker passes; 93-test full suite, 24.5 s)
