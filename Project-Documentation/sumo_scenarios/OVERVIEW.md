# `sumo_scenarios/` — hand-authored SUMO scenarios

## Purpose

Holds all SUMO scenario files (network, vehicle types, route flows, top-level
config) as plain XML. This is the *only* part of the project that's
SUMO-the-tool-specific; every Python module is downstream of these files.
They are version-controlled and meant to be readable as a unit by anyone
evaluating the project — they're the spec for the simulated world the RL
agent learns in.

## Status

✅ **`highway_3lane/` complete.** Headless smoke test passes for all three
density variants; visual sanity check (Task #4) passed.

## Files in this folder

The detailed per-component docs live alongside this OVERVIEW.md:

- **[`network.md`](network.md)** — `highway.{nod,edg,net}.xml`. The 1 km
  3-lane straight, lane-indexing convention, and why the network is shaped
  this way.
- **[`vtypes.md`](vtypes.md)** — `vtypes.add.xml`. The four vehicle profiles
  (aggressive, normal, slow, ego) and their Krauss + LC2013 parameters.
- **[`routes.md`](routes.md)** — `routes_{low,medium,high}.rou.xml`. The
  three Poisson flow density variants and why the mix is held constant.
- **[`config.md`](config.md)** — `highway.sumocfg` + `build.sh`. The collision
  policy, step length, route-file override pattern, and the network-rebuild
  script.

For the deeper *theoretical* context (Krauss formula, LC2013 motivations,
why each vType param produces the personality it does), see
[`../concepts/01-sumo-traffic-model.md`](../concepts/01-sumo-traffic-model.md).

## Public API

These files are consumed by:

- **TraCI / `laneiq.env.sumo_runtime`** — starts SUMO via the locked argv
  built around `highway.sumocfg`, optionally overriding `--route-files`
  to switch density per episode.
- **`make sumo-gui-{low,medium,high}`** — visual-debug Makefile shortcuts
  (recipe in `config.md`).
- **`make scenario`** → `build.sh` — regenerates `highway.net.xml` after
  edits to the plain-XML sources.

The `vType id="ego"` profile is referenced by name when the env spawns the
ego: `traci.vehicle.add(vehID="ego", typeID="ego", ...)`.

## Folder-wide invariants

- **Lane indexing.** SUMO numbers lanes from the right: `lane=0` is the
  rightmost (slow) lane, `lane=2` is the leftmost (fast) lane. The
  observation builder, action applier, and baselines all rely on this.
- **`'--' is illegal inside XML comment bodies.** SUMO's parser is strict.
  We hit this during scaffold and codified the rule in `config.md`.
- **`highway.net.xml` is generated, not hand-authored.** Committed for
  cold-clone repro; regenerated via `make scenario`. Never hand-edit.
- **The collision/teleport policy is load-bearing for RL** and is set in
  *both* `highway.sumocfg` and `laneiq/env/sumo_runtime._LOCKED_FLAGS`.
  These two must stay in sync — see `config.md` for which wins (CLI does)
  and why we keep the cfg matched anyway.

## Last updated

2026-04-26 · `2499dd8` (visual-debug recipe was last addition); per-file
docs split out from this OVERVIEW.md pending commit.
