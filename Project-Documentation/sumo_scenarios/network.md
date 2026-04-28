# Network — `highway.nod.xml` + `highway.edg.xml` + `highway.net.xml`

## What

The geometric backbone of the simulation: 2 nodes 1 km apart connected by a
single 3-lane edge. Hand-authored as plain XML (`.nod.xml`, `.edg.xml`)
and compiled to SUMO's runtime format (`.net.xml`) via `netconvert`.

## Files

- **`highway.nod.xml`** — node positions. 2 nodes total: `start (0,0)` and
  `end (1000,0)`, both `type="dead_end"`. ~10 lines.
- **`highway.edg.xml`** — edge definitions. 1 edge `e1` from `start` to `end`,
  `numLanes="3"`, `speed="36.11"` (m/s ≈ 130 km/h, autobahn-like).
- **`highway.net.xml`** — **GENERATED** by `netconvert`. Committed for
  cold-clone reproducibility (a fresh clone needs zero SUMO tooling beyond
  the wheel). Never hand-edit; regenerate via `make scenario`.

## Schema

```xml
<!-- highway.nod.xml -->
<nodes>
    <node id="start" x="0.0"    y="0.0" type="dead_end"/>
    <node id="end"   x="1000.0" y="0.0" type="dead_end"/>
</nodes>

<!-- highway.edg.xml -->
<edges>
    <edge id="e1" from="start" to="end" numLanes="3" speed="36.11"/>
</edges>
```

## Design notes

### Why dead-end nodes

`type="dead_end"` tells SUMO not to insert traffic-light or junction logic.
For a straight highway with no intersections, this gives clean entry/exit
behavior — vehicles spawn at `start`, despawn at `end`, no junction
artifacts.

### Why a single edge, not a network

The simplest topology that exercises lane-change RL. A multi-edge network
(merges, exits) would let the `lcStrategic` motivation kick in — necessary
for some scenarios but adds complexity unrelated to the core RL question
("when should I change lanes?"). Single edge → `lcStrategic` never fires →
the dynamics you see are pure speed-gain ↔ keep-right ↔ cooperative
tension.

### Why 1 km

Long enough that the ego experiences a full episode (ego travels for 30+ s
at highway speeds), short enough that an episode terminates naturally
(reaching `end`) within a `max_steps` budget. At 36.11 m/s × 30 s = 1083 m,
so ~28 s typical ego traversal.

### Why 36.11 m/s (= 130 km/h)

Autobahn-like, consistent with German traffic literature. Also avoids the
edge case of `vType.maxSpeed > lane.speed` for the `aggressive` profile
(maxSpeed=40), where SUMO caps vType max at the lane limit and behavior
becomes implicit.

### Lane indexing convention

SUMO numbers lanes from the right:

```
lane 2 (leftmost / fast)        ←→ "left" in CHANGE_LEFT
lane 1 (middle)
lane 0 (rightmost / slow)       ←→ "right" in CHANGE_RIGHT
```

This matters everywhere — observation builder, action applier, baselines,
visualization. **`CHANGE_LEFT` increases `lane_index` by 1; `CHANGE_RIGHT`
decreases it.**

## Invariants & gotchas

- **`highway.net.xml` is generated, not hand-authored.** Edit `.nod.xml` /
  `.edg.xml` and run `make scenario` (which calls `build.sh`). Direct edits
  to `.net.xml` will be lost on the next regeneration.
- **Don't add intermediate nodes** without considering downstream effects.
  Even a single intermediate junction triggers SUMO's junction logic and
  may insert micro-deceleration that confounds RL training.
- **Speed is in m/s, not km/h.** Plain XML uses SI throughout. 36.11 m/s
  ≈ 130 km/h ≈ 80 mph. If you change it, update the speed-normalization
  constant in `observations.py` to match.
- **The XML schema is referenced via `xsi:noNamespaceSchemaLocation`** —
  helpful for IDE validation but not required at runtime.

## How to verify

After editing `.nod.xml` or `.edg.xml`:

```bash
make scenario             # regenerates highway.net.xml; warnings printed inline
make sumo-gui-medium      # eyeball the new geometry
```

Or programmatically:

```bash
uv run python -c "
import sumolib
net = sumolib.net.readNet('sumo_scenarios/highway_3lane/highway.net.xml')
edge = net.getEdge('e1')
print(f'edge {edge.getID()}: {edge.getLength():.0f} m, {edge.getLaneNumber()} lanes')
print(f'speed limit: {edge.getSpeed():.2f} m/s')
"
```

## Last updated

2026-04-26 · `494e850` (initial 1 km, 3-lane, 36.11 m/s straight)
