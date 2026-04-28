# Vehicle types — `vtypes.add.xml`

## What

Defines four vehicle "personalities" used in the simulation, tuning both the
**Krauss car-following model** (longitudinal) and the **LC2013 lane-changing
model** (lateral). Three are background traffic (`aggressive`, `normal`,
`slow`); one is the RL-controlled ego (`ego`).

The deep theoretical context for *why* these parameters produce the behavior
they do lives in `Project-Documentation/concepts/01-sumo-traffic-model.md`.
This doc is the per-file reference: schema, design, invariants.

## Schema

```xml
<vType id="..." vClass="passenger"
       maxSpeed="..." accel="..." decel="..." emergencyDecel="..."
       sigma="..." tau="..." minGap="..." length="..."
       lcStrategic="..." lcCooperative="..." lcSpeedGain="..."
       lcKeepRight="..." lcAssertive="..." lcImpatience="..."
       color="R,G,B" guiShape="..."/>
```

## The four profiles

| Param | aggressive | normal | slow | ego |
| --- | --- | --- | --- | --- |
| `maxSpeed` (m/s) | 40.0 | 33.33 | 25.0 | 40.0 |
| `accel` | 3.0 | 2.5 | 1.5 | 2.5 |
| `decel` | 5.0 | 4.5 | 4.0 | 4.5 |
| `emergencyDecel` | 9.0 | 9.0 | 7.0 | 9.0 |
| `sigma` | 0.1 | 0.5 | 0.3 | 0.0 |
| `tau` | 0.8 | 1.2 | 1.5 | 1.0 |
| `minGap` | 1.5 | 2.5 | 3.0 | 2.0 |
| `length` | 4.5 | 4.5 | 5.5 | 4.5 |
| `lcCooperative` | 0.2 | 1.0 | 1.0 | (overridden) |
| `lcSpeedGain` | 2.0 | 1.0 | 0.3 | (overridden) |
| `lcKeepRight` | 0.0 | 1.0 | 2.0 | (overridden) |
| `lcAssertive` | 1.5 | 1.0 | 0.5 | (overridden) |
| `color` (RGB) | 220,40,40 | 240,240,240 | 40,80,200 | 255,215,0 |
| `guiShape` | sedan | passenger | hatchback | sedan |

## Design notes

### Two layers of behavior

Each vType tunes both layers independently:

- **Krauss (longitudinal):** `accel`, `decel`, `tau`, `sigma`, `minGap`.
  Determines speed/braking given the leader.
- **LC2013 (lateral):** `lcCooperative`, `lcSpeedGain`, `lcKeepRight`,
  `lcAssertive`, `lcImpatience`. Determines when to change lanes.

The orthogonality matters: an aggressive driver isn't faster *because* they
change lanes more — they're tuned to do both independently, which composes
into emergent overtaking.

### Why these specific numbers

- **`aggressive`** (`tau=0.8`, `sigma=0.1`, `lcSpeedGain=2.0`,
  `lcCooperative=0.2`): tailgater who chases speed and rarely yields. The
  hardest opponent for the RL ego — won't make space.
- **`normal`** (`sigma=0.5`): the SUMO default, with realistic dawdling.
  High `sigma` is what produces phantom traffic jams at high density. Without
  it, traffic flow looks artificially smooth.
- **`slow`** (`maxSpeed=25`, `lcKeepRight=2.0`, `length=5.5`): a "rolling
  wall" the ego must overtake. Sticks to lane 0. Slightly longer body (5.5 m)
  for a visual "this is a slower vehicle, maybe a small truck" cue.
- **`ego`** (`sigma=0.0`): zero longitudinal noise. Two reasons: (1) the
  RL agent should see deterministic ego dynamics for credit assignment;
  (2) episodes need to be reproducible given a seed. Note `tau=1.0` and
  `decel=4.5` are middle-of-road — we're not optimizing the longitudinal
  model, just the lane-change strategy.

### Why the ego's LC params don't matter

The env code calls `traci.vehicle.setLaneChangeMode(ego, 0)` after spawning
the ego, which **disables every internal lane-change motivation**. The
`lc*` params on the `ego` vType are never consulted at runtime; the policy
issues `traci.vehicle.changeLane(...)` directly.

This is intentional and load-bearing: we want a policy that genuinely owns
the lane-change decision, including the freedom to make unsafe ones (and
pay the reward penalty). See `concepts/01-sumo-traffic-model.md` for the
full rationale.

### Color choices

Distinct hues so the GUI is readable at a glance:

- aggressive = red (matches "danger")
- normal = white (neutral)
- slow = blue (calm)
- ego = gold (impossible to lose visually — important for the demo video)

## Invariants & gotchas

- **The `id` strings are referenced by name** in routes (`type="normal"`)
  and by the Gym env (`traci.vehicle.add(typeID="ego", ...)`). Renaming an
  id requires a sweep across `routes_*.rou.xml` and the env code.
- **`emergencyDecel ≥ decel`** must hold or SUMO warns and clamps. We set
  `emergencyDecel=9.0` for cars (close to ~1g deceleration); `slow` drops
  to 7.0 to model worse braking on a slower/heavier vehicle.
- **`vClass="passenger"` for all four.** Don't change to `truck` /
  `bus` without checking SUMO's `permissions` defaults on the network —
  some lane permissions exclude trucks. Our network has no permissions set,
  so this is moot today, but it's a future trip-wire.
- **Don't move ego into the route file.** The ego is added at runtime by
  the env via `traci.vehicle.add(...)`, after the simulation starts. If
  it's also in the route file, you'll get a duplicate-ID error.
- **`speedFactor` is implicit (default 1.0).** If you set it < 1.0, vehicles
  will drive *below* the lane speed limit by that factor; baselines and
  the observation normalization assume `speedFactor=1.0`.

## How to verify

The vTypes don't have a standalone test (their effect is observed via
SUMO behavior, which is integration-tested via `tests/test_sumo_runtime.py`
and visually in `make sumo-gui-medium`).

Manual check that all four vTypes are loaded:

```bash
uv run python -c "
import traci
from laneiq.env.sumo_runtime import sumo_session

with sumo_session(seed=42, end_time=120):
    for _ in range(50):
        traci.simulationStep()
    for vtype in ['aggressive', 'normal', 'slow', 'ego']:
        try:
            mx = traci.vehicletype.getMaxSpeed(vtype)
            print(f'{vtype:11s} maxSpeed={mx:.1f} m/s')
        except traci.exceptions.TraCIException as e:
            print(f'{vtype:11s} MISSING: {e}')
"
```

Expected: all four printed with maxSpeeds matching the table above.

## Last updated

2026-04-26 · `494e850` (initial four-profile design)
