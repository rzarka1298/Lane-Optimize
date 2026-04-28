# Config — `highway.sumocfg` + `build.sh`

## What

- **`highway.sumocfg`** — top-level SUMO configuration. Pulls together net,
  routes, and vTypes; sets the simulation-time and processing flags that
  matter for RL.
- **`build.sh`** — idempotent regenerator. Runs `netconvert` against the
  plain-XML sources (`highway.nod.xml` + `highway.edg.xml`) to produce
  `highway.net.xml`.

## `highway.sumocfg` — what it sets

```xml
<configuration>
    <input>
        <net-file value="highway.net.xml"/>
        <route-files value="routes_medium.rou.xml"/>
        <additional-files value="vtypes.add.xml"/>
    </input>

    <time>
        <begin value="0"/>
        <end value="3600"/>
        <step-length value="0.1"/>
    </time>

    <processing>
        <ignore-route-errors value="true"/>
        <collision.action value="warn"/>
        <collision.mingap-factor value="0"/>
        <time-to-teleport value="-1"/>
    </processing>

    <report>
        <verbose value="false"/>
        <no-step-log value="true"/>
        <no-warnings value="false"/>
    </report>
</configuration>
```

## Design notes

### Why `step-length="0.1"` (10 Hz sim)

- 10 Hz is a sane decision rate for a lane-change policy; realistic
  driver decisions happen on the order of 100 ms anyway.
- It matches the planned WebSocket stream rate for the live demo
  (Week 5+) — one tick = one frame.
- 100 sim-steps = 10 sim-seconds, and `max_steps=1000` in the env config
  gives 100-second episodes — long enough to traverse the 1 km road, short
  enough that DQN replay-buffer episodes don't blow up.

### Collision policy — load-bearing for RL

Three collision-related settings work together:

| Setting | Value | Why |
| --- | --- | --- |
| `collision.action` | `warn` | Don't auto-teleport-or-remove on collision; let TraCI detect it via `getCollidingVehiclesIDList`. RL needs to *see* collisions to penalize them. |
| `collision.mingap-factor` | `0` | Register collisions at true zero overlap, not at a generous safety margin. We *want* the RL ego's bad decisions to count as crashes, not "near misses." |
| `time-to-teleport` | `-1` | Disable the "stuck vehicle" teleport entirely. Without this, in high-density scenarios SUMO would silently teleport stuck cars (potentially the ego!) past obstacles, masking real failures. |

**Don't change one without understanding all three.** A common
mis-configuration is leaving `time-to-teleport` at its default (300 s)
which silently teleports the ego past collisions, making it appear as
though the policy never crashes.

### Why the routes file in the cfg is `routes_medium`

It's just a default. The Gym env always overrides via
`--route-files routes_<density>.rou.xml` from `build_argv()`. The default
exists so `make sumo-gui` (no env, just visualizing) opens at a useful
density.

### Why `ignore-route-errors="true"`

Defensive: if a vehicle has a route validation issue (e.g., a future
multi-edge network adds an unreachable edge), SUMO will warn instead of
crashing the whole sim. For RL training, we want graceful degradation, not
training runs killed by a single misconfigured flow.

### Why `verbose="false"` + `no-step-log="true"`

RL training prints progress at episode boundaries; SUMO's per-step logging
would drown that out. The TensorBoard dashboard is the source of training
progress; SUMO's stdout stays minimal.

`no-warnings="false"` (i.e., warnings ARE shown) — we want collision
warnings and route validation warnings to surface in training logs so we
can spot bugs.

## `build.sh` — what it does

```bash
#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
uv run netconvert \
    --node-files highway.nod.xml \
    --edge-files highway.edg.xml \
    --output-file highway.net.xml \
    --no-turnarounds true
```

- `set -euo pipefail`: fail on any error, undefined variable, or pipe
  failure. Standard hardened-bash header.
- `cd "$(dirname "$0")"`: makes the script callable from any cwd (the
  Makefile target uses this).
- `--no-turnarounds true`: don't auto-generate U-turn connections at the
  dead-end nodes. We don't want them on a one-way highway.

Invoke from anywhere via `make scenario` (preferred) or directly:

```bash
bash sumo_scenarios/highway_3lane/build.sh
```

## Invariants & gotchas

- **CLI flags override `.sumocfg` values.** When `build_argv()` in
  `sumo_runtime.py` passes `--collision.action warn`, it overrides whatever
  the cfg says. So the cfg and the locked CLI flags should match —
  diverging them silently is a bug magnet.
- **The cfg is a default for `make sumo-gui` and ad-hoc CLI usage.** The
  authoritative settings during training are in
  `laneiq/env/sumo_runtime._LOCKED_FLAGS`.
- **`build.sh` requires `uv` on PATH.** It calls `uv run netconvert`. CI
  workflows that run it must `uv` install first.
- **`highway.net.xml` is committed.** Re-running `build.sh` only changes
  the file if the inputs (`.nod.xml` / `.edg.xml`) actually changed. If you
  see an unexpected diff after `make scenario`, check whether netconvert
  re-ordered attributes in the output; commit the new version if so.
- **`'--' is illegal in XML comment bodies.** SUMO's parser is strict.
  Don't write `--route-files` (or any other `--flag-name`) inside a
  `<!-- -->` block in `.sumocfg`. Use prose instead.

## How to verify

Visual check (best):

```bash
make sumo-gui-medium
```

Validate the cfg parses:

```bash
cd sumo_scenarios/highway_3lane
uv run sumo -c highway.sumocfg --no-step-log true --end 5
# should print version banner + "Quitting." with no errors
```

Regenerate the network:

```bash
make scenario
# outputs "Success." and writes highway.net.xml
```

## Last updated

2026-04-26 · `2499dd8` (visual-debug recipe doc updated; config unchanged
since `494e850`)
