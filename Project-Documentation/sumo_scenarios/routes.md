# Routes — `routes_low.rou.xml` · `routes_medium.rou.xml` · `routes_high.rou.xml`

## What

Three route files, one per density level. Each spawns background traffic
on edge `e1` using **Poisson-distributed** inter-arrival times. The vType
*mix* is held constant across densities (60% normal · 25% aggressive ·
15% slow); only the per-vType arrival *rates* change.

The ego (RL-controlled) vehicle is **not** in any route file — it's added
at runtime by the Gym env via `traci.vehicle.add(...)`.

## Schema

```xml
<routes>
    <route id="r_through" edges="e1"/>

    <flow id="f_normal" type="normal" route="r_through"
          begin="0" end="3600" period="exp(λ)"
          departLane="best" departSpeed="desired"/>

    <flow id="f_aggressive" type="aggressive" route="r_through" .../>
    <flow id="f_slow"       type="slow"       route="r_through" .../>
</routes>
```

## Density rates

`period="exp(λ)"` produces an exponentially distributed inter-arrival time
with rate λ vehicles/second (i.e., a Poisson process).

| Density | normal λ | aggressive λ | slow λ | Total | Mean inter-arrival | Steady-state cars on 1 km |
| --- | --- | --- | --- | --- | --- | --- |
| **low** | 0.15 | 0.06 | 0.04 | 0.25/s | ~4.0 s | ~7–8 |
| **medium** | 0.40 | 0.15 | 0.10 | 0.65/s | ~1.5 s | ~20 |
| **high** | 0.85 | 0.40 | 0.20 | 1.45/s | ~0.7 s | ~45 |

Steady-state estimates use Little's Law: `cars ≈ rate × mean_traversal_time`,
with mean traversal ~30 s at autobahn speeds.

## Design notes

### Why Poisson (not periodic)

Real traffic has variable spacing. Periodic insertions (`period="N"`) would
produce a metronome-like pattern that's both unrealistic *and* exploitable
by the RL agent ("a car always shows up every 1.5 s in the right lane").
The exponential `period="exp(λ)"` matches what a real highway looks like:
clusters and gaps.

### Why the same mix across densities

Holding the vType ratio constant separates the **density effect** (how many
cars) from the **personality effect** (which kinds). When the eval matrix
shows the agent doing worse at high density vs low, we know it's the
density (more constraints) and not "more aggressive cars per minute"
(which would also vary if mix changed with rate).

### Why slow vehicles depart in lane 0

`departLane="0"` for `f_slow` (instead of `"best"`). Mirrors the real-world
fact that slow drivers self-select into the right lane. If we let `"best"`
pick for them, they'd often spawn in the middle and immediately need to
move right — a transient dynamic that confuses the RL agent in the first
~5 seconds.

`departLane="best"` for normal and aggressive is fine: SUMO picks based on
local density, which is the realistic behavior.

### Why `departSpeed="desired"`

`"desired"` = `vType.maxSpeed × speedFactor`, capped at the lane limit. So
new vehicles enter at their preferred cruising speed.

Alternatives considered:
- `"max"`: ignores the lane speed limit for vehicles whose `maxSpeed` exceeds
  it (aggressive at 40 vs lane 36.11). Produces unrealistic insertions.
- `"random"`: floods the road with stalled vehicles right at the entry node.
  The new vehicle spawns at random in [0, max]; small samples at congestion
  cause cascading insertion failures.
- `"speedLimit"`: caps at lane limit. Acceptable, but `"desired"` is more
  faithful to vType personality.

### Why one route definition (`r_through`)

There's only one path through a single-edge network — start → end. All flows
share the route. If we add merges/exits later, this becomes a list of routes
with weighted selection.

### Why `begin="0" end="3600"`

3600 s = 1 hour of simulation. Vastly more than any individual episode
(`max_steps=1000` in the planned env config = 100 s of sim time). The flow
keeps producing traffic for as long as the env wants to run.

## Invariants & gotchas

- **Ego is NOT in route files.** Adding an `<vehicle id="ego" .../>` here
  collides with `traci.vehicle.add(vehID="ego", ...)` in the env code.
  Keep ego out of the routes; the env owns its insertion.
- **`type="..."` references the vTypes by id.** Renaming a vType in
  `vtypes.add.xml` requires updating all three route files.
- **The route id `r_through`** is referenced by every flow. Rename it
  consistently if changed.
- **Density is set by the env, not by editing files at runtime.** The Gym
  env passes `--route-files routes_<density>.rou.xml` via
  `build_argv(route_file=...)`; per-episode density switching does not
  require modifying these files.
- **Aggregate insertions per second ≠ steady-state cars.** Some vehicles
  are still in transit; expect `inserted` count >> `running` count after
  steady state.

## How to verify

Run a 60-second headless sim per density and dump statistics:

```bash
cd sumo_scenarios/highway_3lane
for d in low medium high; do
  uv run sumo -c highway.sumocfg \
    --route-files "routes_${d}.rou.xml" \
    --end 60 \
    --statistic-output "/tmp/stats_${d}.xml" \
    --no-step-log true
  uv run python -c "
import xml.etree.ElementTree as ET
r = ET.parse('/tmp/stats_${d}.xml').getroot()
v = r.find('vehicles')
print('${d}: inserted=', v.attrib['inserted'], 'running=', v.attrib['running'])
"
done
```

Expected ranges (after 60 s, still warming up):

| Density | inserted | running |
| --- | --- | --- |
| low | 8–18 | 1–3 |
| medium | 25–45 | 12–25 |
| high | 60–95 | 30–50 |

## Last updated

2026-04-26 · `494e850` (initial Poisson flows at 0.25 / 0.65 / 1.45 veh/s)
