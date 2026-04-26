# Concept 01 — SUMO's two-layer traffic model

## Why this matters for RL

Your RL agent doesn't learn to drive in a vacuum. It learns *against* the
behavior of the surrounding traffic — and that traffic is itself running a
microscopic simulation, not a random-noise generator. Understanding that
simulation tells you:

1. **What patterns the agent can exploit** (e.g., a hesitating cooperative
   driver makes a lane-change feasible; an aggressive driver does not).
2. **What's controllable in your reward signal** (you can make the world
   harder by tuning vType params, not by changing the RL algorithm).
3. **What you can confidently put in interview answers** ("here's exactly
   how the simulator works, and here's why my hyperparameters are sane").

## The two layers

SUMO models each vehicle's behavior in two independent layers, evaluated every
sim step (0.1 s):

```
                ┌──────────────────────────┐
                │  Lane-change model       │  → output: target lane
                │  (LC2013 / Erdmann 2014) │
                └──────────────┬───────────┘
                               │ then once lane is set...
                               ▼
                ┌──────────────────────────┐
                │  Car-following model     │  → output: speed/accel
                │  (Krauss, default)       │
                └──────────────────────────┘
```

The **lane-change** layer decides *whether* to change lanes. The
**car-following** layer decides *how fast* to go given the leader in the
current lane. They're orthogonal — vType parameters tune them separately.

## Layer 1: Krauss car-following

The Krauss (1998) model picks a safe speed `v_safe` such that even if the
leader brakes maximally, this vehicle can stop without colliding. Then it
applies stochastic dawdling — drivers don't perfectly hit the safe speed.

Formally, at each step:

```
v_safe(t) = v_lead(t) + (gap - v_lead(t)·τ) / ((v + v_lead) / (2·b) + τ)
v_des(t)  = min(v_safe, v + accel·dt, v_max)
v(t+dt)   = max(0, v_des - sigma·accel·dt·rand())
```

where:
- `τ` (`tau`) — desired headway in seconds. Bigger = larger gap.
- `b` (`decel`) — comfortable deceleration.
- `accel`, `decel` — comfortable accel / decel.
- `v_max` (`maxSpeed`) — vType's preferred max.
- `sigma` — driver imperfection ∈ [0, 1]. Subtracts up to `sigma·accel·dt`
  random m/s from the speed each step, simulating dawdling/inattention.

The two parameters that matter most for **traffic feel**:

- High `sigma` (~0.5) → realistic stop-and-go waves at high density. This is
  the parameter that creates the "phantom traffic jam" effect even when no
  one brakes hard.
- High `tau` (~1.5) → cautious driver, big gaps. Pushes followers further back
  and *reduces* downstream throughput.

These are why we set `sigma=0.5` for `normal` vehicles — without it, traffic
flow looks artificially smooth.

## Layer 2: LC2013 lane-changing

Erdmann's LC2013 model (the SUMO default, ~2014) computes a per-step
lane-change incentive as a weighted sum of four motivations. If the incentive
exceeds a threshold *and* the change is safe (gap to new leader/follower is
sufficient given Krauss safety), the vehicle changes.

Motivations (weighted by named params):

| Motivation | Triggered when… | Param | Default |
| --- | --- | --- | --- |
| **Strategic** | Must change to follow the route | `lcStrategic` | 1.0 |
| **Cooperative** | Adjacent vehicle wants to change → I move out of their way | `lcCooperative` | 1.0 |
| **Speed gain** | Adjacent lane is faster | `lcSpeedGain` | 1.0 |
| **Keep right** | Country requires returning right when possible | `lcKeepRight` | 1.0 |

Plus modifiers:

- `lcAssertive` — how small a gap I'll accept before squeezing in. >1.0 =
  more aggressive gap-fitting.
- `lcImpatience` — increases over time when I want to change but can't,
  eventually forcing a lower safety threshold.

For a single-edge highway like ours, `lcStrategic` never fires (no route
choice). The dynamics you actually see come from the speed-gain ↔ keep-right
tension, mediated by cooperative yielding.

### Why our three vTypes feel distinct

| | aggressive | normal | slow |
| --- | --- | --- | --- |
| `lcCooperative` | 0.2 | 1.0 | 1.0 |
| `lcSpeedGain` | 2.0 | 1.0 | 0.3 |
| `lcKeepRight` | 0.0 | 1.0 | 2.0 |
| `lcAssertive` | 1.5 | 1.0 | 0.5 |

- **Aggressive** rarely yields (`Cooperative=0.2`), eagerly chases speed
  (`SpeedGain=2.0`), never returns right (`KeepRight=0.0`), and squeezes
  into tight gaps (`Assertive=1.5`). This produces a driver that overtakes
  on either side and rarely lets the ego in.
- **Normal** is balanced — the SUMO defaults.
- **Slow** mostly sits in the right lane (`KeepRight=2.0`), rarely chases
  speed (`SpeedGain=0.3`), and won't squeeze (`Assertive=0.5`). They become
  a "rolling wall" the ego must overtake.

This heterogeneity is what makes the RL problem *interesting* — a uniform
fleet would let the agent overfit to a single behavior.

## What the ego inherits

The `ego` vType inherits Krauss for longitudinal control (good — we don't
want the RL agent to also learn to brake; that's a solved problem and would
explode the action space). But its lane-change behavior is fully overridden:

```python
traci.vehicle.setLaneChangeMode(ego_id, 0)  # 0b000000000000
```

This disables every internal LC motivation including safety overrides. The
*only* lane-change signal SUMO accepts for the ego is
`traci.vehicle.changeLane(ego, target_lane, duration)`. **This is by design** —
we want a policy that genuinely owns the lane-change decision, including the
freedom to make unsafe ones (and pay the reward penalty for them). Document
this prominently in the README; it's the kind of "this person understands
the simulator" detail that earns interview points.

## How this feeds back into RL design

A few concrete consequences for downstream design:

- **Reward shaping** — collisions are *possible* because ego safety overrides
  are off. The collision penalty (`w_collision = -50`) must be heavy enough
  to dominate any speed gain from reckless lane changes.
- **Observation features** — knowing that aggressive vehicles tailgate
  (`tau=0.8`) means the "gap to rear neighbor" feature has a different
  meaning per-vType. The observation builder includes a `vClass-bucket-id`
  feature per neighbor partly for this reason — let the policy implicitly
  learn "tailgater behind = different lane-change risk."
- **Reward shaping for jerk** — Krauss + `sigma=0.5` already produces some
  natural acceleration noise on the ego (when the leader dawdles). Don't
  over-penalize jerk or you train a policy that's afraid to brake when needed.

## Sources

- Krauss, S. (1998). *Microscopic modeling of traffic flow: Investigation of
  collision-free vehicle dynamics.* PhD thesis, DLR.
- Erdmann, J. (2014). *SUMO's lane-changing model.* In *Modeling Mobility
  with Open Data* (pp. 105–123). Springer.
  [PDF on DLR e-lib](https://elib.dlr.de/102254/1/Springer-SUMOs_Lane_changing_model.pdf)
- SUMO docs — [Car-following models](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#car-following_models)
  · [Lane-changing models](https://sumo.dlr.de/docs/Definition_of_Vehicles%2C_Vehicle_Types%2C_and_Routes.html#lane-changing_models)
  · [TraCI changeLane](https://sumo.dlr.de/docs/TraCI/Change_Vehicle_State.html#change_lane_0x13)

## Last updated

2026-04-26 · `494e850` (concept note authored alongside Task #3)
