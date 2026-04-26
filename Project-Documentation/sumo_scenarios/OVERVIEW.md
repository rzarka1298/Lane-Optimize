# `sumo_scenarios/` — hand-authored SUMO scenarios

## Purpose

Holds all SUMO scenario files (network, vehicle types, route flows, top-level
config) in plain XML. This is the *only* part of the project that's
SUMO-the-tool-specific; everything Python is downstream of these files. They
are version-controlled and meant to be read as a unit by anyone evaluating
the project — they're the spec for the simulated world the RL agent learns in.

## Status

✅ **`highway_3lane/` complete.** Headless smoke test passes for all three
density variants (insertion + running counts match the planned Poisson rates).
Ready for visual sanity-check via `make sumo-gui-medium` (Task #4).

## Key files

```
sumo_scenarios/highway_3lane/
├── build.sh                  # regenerates highway.net.xml via netconvert
├── highway.nod.xml           # 2 nodes — start (0,0), end (1000,0)
├── highway.edg.xml           # 1 edge, 3 lanes, 36.11 m/s (130 km/h) speed limit
├── highway.net.xml           # GENERATED — committed for cold-clone repro
├── vtypes.add.xml            # 4 vehicle profiles: aggressive, normal, slow, ego
├── routes_low.rou.xml        # ~7-8 background vehicles on the road
├── routes_medium.rou.xml     # ~20 background vehicles
├── routes_high.rou.xml       # ~45 background vehicles (near capacity)
└── highway.sumocfg           # default config — picks routes_medium by default
```

## Public API

These files are consumed by:

- **TraCI / `laneiq.env.highway_env`** (Week 2+) — starts SUMO via
  `traci.start([sumo_binary, "-c", "<scenario>/highway.sumocfg", ...])`,
  optionally overriding `--route-files` to switch density at episode reset.
- **`make sumo-gui-{low,medium,high}`** — visual debug shortcuts (Makefile).
- **`make scenario`** → `build.sh` — regenerates `highway.net.xml` from
  the plain-XML sources whenever the network topology changes.

The `vType id="ego"` profile is referenced by name when the RL env spawns
the ego vehicle via `traci.vehicle.add(vehID="ego", typeID="ego", ...)`.

## Invariants & gotchas

- **Lane indexing.** SUMO numbers lanes from the right: `lane=0` is the rightmost
  (slow) lane, `lane=2` is the leftmost (fast) lane. The observation builder
  must respect this; `lcKeepRight` increases with lane index closer to 0.
- **`build.sh` regenerates `highway.net.xml`.** That file is committed for
  cold-clone reproducibility (so a fresh clone can `uv run sumo-gui` without
  needing `netconvert`), but it's also marked `GENERATED` — never hand-edit.
  Edit `highway.nod.xml` / `highway.edg.xml` and re-run `make scenario`.
- **`--` is illegal inside XML comments.** SUMO's parser is strict about this
  (we hit it during scaffold). Don't write `--route-files` or similar in the
  comment header of any `.xml` here. The `<!-- -->` delimiters themselves are
  fine; only `--` *inside* the comment body breaks parsing.
- **`departSpeed="desired"`** lets vehicles enter at their cruising preference
  (vType `maxSpeed × speedFactor`, capped at lane limit). Don't use `"max"` —
  for the `aggressive` vType (`maxSpeed=40`) it ignores the lane's 36.11 limit
  and produces unrealistic insertions. Don't use `"random"` — it floods the
  road with stalled vehicles right at the entry node.
- **Slow vehicles depart in `lane="0"` only.** Realistic — slow drivers stay
  right. Other vTypes use `departLane="best"` so SUMO chooses based on local
  density.
- **`collision.action="warn"`** + **`collision.mingap-factor="0"`** in the
  sumocfg are *load-bearing* for RL: we want collisions detected by TraCI
  (`getCollidingVehiclesIDList`) at true zero-overlap, not handled by SUMO
  via teleport. If you change either, the env's collision detection breaks.
- **`time-to-teleport="-1"`** disables the "stuck vehicle" teleport. Without
  this, in high-density scenarios SUMO would silently teleport stuck cars
  past obstacles, including potentially the ego, masking real failures.
- **Step length = 0.1s (10 Hz).** Shared with the live demo's WebSocket
  stream rate; if you change it, the WS tick rate in `backend/` must match.

## Visual debugging in `sumo-gui`

Standard recipe to eyeball the scenario:

```bash
cd /Users/rugvedzarkar/Lane-Optimize        # repo root
make sumo-gui-medium                         # or sumo-gui-low / sumo-gui-high
```

The GUI opens with the simulation paused at t=0. Then:

1. **Set the delay slider.** Top toolbar — find the "Delay (ms)" field. Default
   is `0` which runs as fast as the GUI can render (~30s of sim per real
   second). Set to `100` ms to roughly real-time, or `50` ms to watch at 2×.
2. **Start.** Press the green ▶ "Run" button (top-left). Press ⏸ to pause.
3. **Zoom / pan.** Mouse wheel zooms; right-click + drag pans. There's a
   "Locate" button in the toolbar to recenter.
4. **What to look for.**
   - Aggressive (red) vehicles overtake on either side, rarely yield.
   - Slow (blue) vehicles stick to lane 0 (rightmost).
   - At high density, ~40 vehicles on screen with natural stop-and-go waves.
   - **The bottom-right warning panel should stay quiet.** Any "collision"
     or "teleport" warnings during normal flow indicate a tuning issue.
5. **Inspect a vehicle.** Right-click any vehicle → "Show Parameter" opens a
   live panel of its state (speed, lane, accel, leader gap). Useful for
   sanity-checking that vTypes are behaving as designed.

If `make sumo-gui-medium` errors out: confirm the venv is synced
(`uv sync --extra sumo`) and that `uv run sumo --version` prints `1.26.0`.

## Density rationale

| Density | Total Poisson rate | Mean inter-arrival | Steady-state cars on 1 km road |
| --- | --- | --- | --- |
| low    | 0.25 veh/s | ~4 s   | ~7-8  |
| medium | 0.65 veh/s | ~1.5 s | ~20   |
| high   | 1.45 veh/s | ~0.7 s | ~45   |

Mix is constant across densities (60% normal · 25% aggressive · 15% slow); only
the rates change. This keeps the *behavioral* distribution of opponents
identical so we cleanly separate the effect of density from the effect of
traffic personality.

## Last updated

2026-04-26 · `2499dd8` (visual-debugging recipe added);
prior `494e850` (scenario authored, headless smoke test passes)
