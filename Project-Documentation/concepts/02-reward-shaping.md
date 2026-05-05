# Concept 02 — Reward shaping

## Why this matters

Reward shaping is the highest-leverage decision in an RL project. Get it
wrong and the agent learns the wrong thing — fast, persistently, and
sometimes in ways that look correct on a metric dashboard. The literature
calls this **reward hacking**: the policy optimizes exactly what you
*wrote*, not what you *meant*.

For LaneIQ, the reward function is also where the project's recruiter
narrative lives:

- The **Pareto sweep** in Week 4 (vary one weight at a time, plot
  safety vs speed) is the "I understand reward shaping" hero artifact.
- The **per-component reward dict** logged to TensorBoard is the
  "this person can debug RL" diagnostic.
- The **deliberate `-50` collision penalty + lane-change cost** is what
  separates this project from "I plugged numbers into an env and trained"
  on a recruiter's mental scorecard.

This note explains the v1 reward in `laneiq/env/rewards.py` and why each
piece was chosen — the kind of explanation you'd give in an interview.

## The v1 reward

```
r =  w_speed       · min(speed / v_max, 1)        # +1.0   drive fast
   + w_lane_cost   · I[action != STAY]            # -0.05  don't thrash
   + w_jerk        · |Δaccel| / a_max             # -0.2   drive smoothly
   + w_unsafe_gap  · I[front_thw < 1.5s]          # -0.5   keep your distance
   + w_collision   · I[collided]                  # -50.0  don't crash (terminal)
   + w_alive       · 1                            # +0.01  small dense signal
```

Six components, three positive forces (speed, alive, implicitly the lack
of collision), three negative forces. Each one earns its place by avoiding
a *specific failure mode* of simpler shapes.

## The classic failure modes — and how each component prevents one

### 1. Sparse reward → never learns

> "Reward = +1 if you survive 1000 steps, 0 otherwise."

Most rollouts get 0; the policy never sees a useful gradient. RL is dead
in the water.

**Our prevention:** `w_speed · (v / v_max)` is a per-step dense reward.
Even a freshly-initialized random policy gets immediate signal — drive
faster, get more reward. `w_alive` adds a tiny constant per step so the
policy also sees "exists = good."

### 2. Speed-only → suicide cruiser

> "Reward = current speed. That's it."

The policy learns to floor it, ignoring leaders. Crashes 5 seconds in,
every episode. Looks great on the speed metric.

**Our prevention:** `w_collision = -50` is calibrated so a crash erases
~50 seconds of perfect cruising (`w_speed · 1.0 · 50 = 50`). The
expected return of "drive fast, crash often" is strictly worse than
"drive moderately, never crash."

### 3. Tiny collision penalty → "occasional crash for speed"

> "Reward = speed - 5 · I[collided]"

Policy learns that the speed gain over 200 steps is +160, while the
crash penalty is only -5. So crashing is *positive expected value*. Trains
to crash deliberately.

**Our prevention:** the -50 vs +1 ratio is large enough that no amount of
speed makes a crash worth it. We checked the math during weight selection;
the test `test_total_dominated_by_collision` pins the invariant.

### 4. No lane-cost → thrashing

> No `w_lane_cost` term.

Policy learns that constantly checking lanes (LEFT, then RIGHT, then LEFT,
then RIGHT, ...) gets it the highest expected speed because it's always
opportunistically maximizing. Looks like a panicked driver. Burns CPU.
Annoying to watch in the demo.

**Our prevention:** `w_lane_cost = -0.05` per non-STAY action. Small
enough that a justified change still pays off (a lane change buys ~5 m/s
sustained = +0.14 reward over the next many steps), large enough that
mindless thrashing converges to STAY.

### 5. No jerk penalty → motorbike driver

> No `w_jerk` term.

Policy uses max acceleration and max braking constantly, because that's
the locally-optimal way to maximize speed in changing traffic. Feels
like driving with a sociopath. Shows up in the demo as "the AI flickers."

**Our prevention:** `w_jerk = -0.2 · |Δa| / a_max`. Smooths the policy's
acceleration profile without preventing actual responses to threats. Tuned
small relative to the speed reward so emergency braking is still
preferable to a collision.

### 6. No unsafe-gap signal → "wait until collision"

> Only the binary `collided` signal teaches safety.

The policy learns that 1m gap is fine until the moment of collision. There's
no continuous gradient between "safe" and "crashed." Convergence is slow
because the only safety teaching signal is rare and binary.

**Our prevention:** `w_unsafe_gap = -0.5 · I[front_thw < 1.5s]`. A continuous
"you're tailgating" signal that fires before the collision. The policy
gets penalized every step it spends in the danger zone, learning the safe
distance much faster than it would from collision feedback alone.

### 7. Sparse alive → "self-terminate when negative"

> No `w_alive` term, all other components net-negative for a stretch.

Imagine the policy stuck in heavy traffic with no overtaking opportunities
for 30 seconds. Net reward is mildly negative (no speed gain, plus the
occasional unsafe-gap or jerk hit). The policy learns: *if I crash now,
the episode ends and I stop accumulating negative reward*. Suicide is
locally optimal.

**Our prevention:** `w_alive = +0.01` per step. Small enough that it
doesn't dominate any other term, large enough that survival > suicide
even when other components are net-negative. The tiny `0.01` is a
deliberately small number — bigger and the policy learns "exist forever,
don't bother going fast."

## How the weights were chosen

This is the interview question. The honest answer is **calibration**, not
theory:

1. **Pick the dominant terms.** Speed is the main *positive* force; collision
   is the main *negative* terminal. Set them so the ratio matches your
   risk preference. Ours: 1 unit/step of speed vs 50 units of collision
   = "trade 50 seconds of perfect cruising for one crash" = "absolutely not."

2. **Set lane-cost and jerk small relative to speed.** They're regularizers,
   not primary objectives. Rule of thumb: their per-step magnitude should
   be 5–20% of `w_speed · 1.0 = 1.0`. Ours: 0.05 and 0.2. Small enough
   not to fight speed; big enough to shape behavior.

3. **Set unsafe-gap between alive and collision.** It's a continuous proxy
   for "imminent collision risk." Ours: 0.5 — about 10× the lane-cost
   (because ramming is worse than thrashing) but 100× smaller than a
   crash itself.

4. **Alive last.** Just enough to break ties in favor of survival. 0.01.

In Week 4, the **Pareto sweep** systematically varies these to produce
the hero plot: x-axis = collision rate, y-axis = mean speed, one point
per weight setting. The sweep file is `configs/reward/*.yaml`; we'll
ship 5–6 settings (default, safety-heavy, speed-heavy, etc.) and the
plot lives at `docs/results.md`.

## Common pitfalls we've avoided

- **Reward hacking via division.** "Reward = speed / (1 + collisions)"
  looks safe but is not. Policy learns to game the denominator. We use
  *additive* terms with explicit signs.

- **Implicit reward via observation manipulation.** Adding `time_since_last_change`
  to the obs and reward = "smooth driving" implicitly. Bad: the policy
  learns to satisfy the obs feature, not the underlying goal. We keep
  reward in `rewards.py` and obs in `observations.py`.

- **Per-episode bonuses.** "+100 for reaching the end of the highway."
  Sparse, hard to credit-assign across 1000 steps. We keep all rewards
  per-step and dense.

- **Logging only the scalar.** Returning `(scalar, components)` is the
  diagnostic that tells us *which term* is being gamed. When the policy
  starts thrashing, the `lane_cost` curve spikes; when it tailgates,
  `unsafe_gap` does. Without the dict you'd see "reward dropped" and
  guess.

## What the policy actually learns

By design, the optimal policy under v1 should:

1. Maintain near-`v_max` cruising speed.
2. Change lanes only when the projected speed gain over the next ~10
   seconds exceeds the `0.05` cost.
3. Keep at least 1.5 s headway when possible.
4. Brake smoothly to avoid jerk, but emergency-brake when needed (the
   collision penalty dominates the jerk penalty).
5. Never deliberately crash, even when stuck in slow traffic.

When you see the Week-3 trained DQN, these are the behaviors to verify.
If you see the policy thrashing, tailgating, or never overtaking,
that's the per-component plot's job to explain *why*.

## Sources

- Sutton & Barto, *Reinforcement Learning: An Introduction*, 2nd ed.,
  Ch. 17.4 ("Designing Reward Signals").
- Krakovna et al., *Specification gaming examples in AI*. DeepMind, 2020.
  ([blog post](https://deepmind.com/blog/article/Specification-gaming-the-flip-side-of-AI-ingenuity))
- Ng, Harada, Russell, *Policy Invariance Under Reward Transformations:
  Theory and Application to Reward Shaping*, ICML 1999. ([paper](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf))
- Skalse et al., *Defining and Characterizing Reward Hacking*, NeurIPS 2022.
  ([arXiv:2209.13085](https://arxiv.org/abs/2209.13085))

## Last updated

2026-05-05 · pending commit (concept note authored alongside Task #9)
