# RL design

The technical core. What the AI sees, what it can do, what it's
optimizing, and how three different algorithm families approach the
same problem.

## Problem statement

> Given the current state of a 3-lane highway populated with
> background traffic, choose one of three discrete actions every
> 100 ms (`STAY`, `CHANGE_LEFT`, `CHANGE_RIGHT`) such that, over
> a 100-second episode, the ego maximizes a 6-component reward
> trading off speed, safety, and stability.

Episode terminates when the ego (a) collides, (b) reaches the end
of the 1 km road, or (c) hits the 1000-step truncation. The agent
sees only its own observation — no privileged info about other
vehicles' intents.

## Observation space — 25 dimensions

`laneiq/env/observations.py:build_observation` produces a
`Box(25,)` vector per step from a TraCI snapshot (`OBS_NAMES` lists
every slot).

| Indices | Component | Meaning |
| --- | --- | --- |
| 0–3 | Ego self-state | speed / lane limit, lane idx (0/1/2 → 0/0.5/1), accel, lateral offset |
| 4–21 | Neighbor slots | for each of 3 lanes × {front, rear}: gap / 100 m, Δspeed / v_max, vType bucket (−1 = no car) |
| 22 | Local density | vehicles within ±100 m, saturating at 20 |
| 23 | Lane-validity packed scalar | `(left_ok·2 + right_ok) / 3` — "can I change left / right *right now*" |
| 24 | Time-headway to leader | normalized, capped at 5 s |

All values are clipped to `[-1, 1]`. Normalization constants live in
`ObservationConfig` so changing them is a one-place edit.

**The bug that hid behind good numbers.** Side-lane slots come from
TraCI's `getNeighbors(ego, mode)`, where bit 0 selects left (0) or
right (1). The original table read that bit backwards: in the middle
lane the left and right slots were swapped, and from an outer lane the
real adjacent lane was never queried, so it always looked empty. Every
agent trained before the fix learned on mirrored input, and the "safe"
policies were mostly ones that had learned never to change lanes. The
unit-test mock had been written from the same wrong table, so it agreed
with the bug. The fix came with a test that places cars in known lanes
in real SUMO and checks which slots they land in. Lesson: build mocks
from the external system's spec, never from the code under test.

**Why 25 not larger:** the trained PPO converges in 150k steps with
this much. Larger obs = bigger network = more compute, no signal that
extra signal would help. The lane-validity packed scalar (slot 23) is
the cleverest dimension: it carries the action mask in one float, for
both DQN (`mask_invalid_q_values`) and MaskablePPO
(`env.action_masks()`). It also carries the lane-change cooldown (see
below), so the policy can *see* when a change isn't available.

## Action space — 3 discrete

`laneiq/env/actions.py`:

| Action | Effect |
| --- | --- |
| `STAY` (0) | No-op; ego stays in current lane (Krauss model handles longitudinal) |
| `CHANGE_LEFT` (1) | `traci.vehicle.changeLane(ego, current+1, 1.0s)` |
| `CHANGE_RIGHT` (2) | `traci.vehicle.changeLane(ego, current-1, 1.0s)` |

Lane indexing is SUMO's: 0 = rightmost (slow), 2 = leftmost (fast).
So `CHANGE_LEFT` raises the lane index. Counterintuitive but
consistent throughout the codebase.

**Why discrete (not continuous lateral control):** the demo question
is "when should I change lanes?", not "what trajectory should I
follow?". Continuous control would also force a much bigger
observation space (visible road geometry) and a different
algorithmic approach (DDPG / SAC / continuous PPO).

**Action masking:** `valid_actions_mask(obs)` reads slot 23 and
returns `[True, left_ok, right_ok]`. `STAY` is always valid — the
agent can choose to do nothing even at lane boundaries.

**Lane-change cooldown** (`LaneChangeCooldown`, 15 steps = 1.5 s).
After an executed lane change the env forces `STAY` for 1.5 s and
clears slot 23, so the mask blocks changes. Unmasked learners (MAPPO)
still see why a request won't execute. Without the cooldown, agents that
could see the side lanes dithered left and right. `gate()`
picks what to submit, the env steps SUMO, then `confirm(changed)` starts
the cooldown only if the ego's lane index actually moved. The reward
still charges the *requested* action, so spamming requests costs.

**Who decides, and who checks.** The env turns off all of SUMO's own
lane-change motivations on the ego, so every change is the policy's
decision. SUMO then *checks* each request against the
`ego_lane_change_mode` setting (`laneiq/env/actions.py`), which is
measured against real SUMO with a car parked right beside the ego:

| Mode | Request into an occupied lane |
| --- | --- |
| `0` (pre-Week-10b default) | executed unchecked → collision |
| `768` (**default**) | refused; the ego stays put, no speed adaptation |
| `1024` | executed → collision (a sublane-model bit, *not* a safety check) |

768 works like a car's lane-change assist: the policy still chooses
*when* and *where* to change, and SUMO won't let it cut into another
car. Every collision in the earlier evals was a change into a car
directly alongside under mode 0. `info["executed_action"]` reports what
SUMO actually did (a refused change counts as `STAY`), and that's what
the eval harness and the live ticker use.

The ego's vType also pins `speedFactor="1" speedDev="0"`. Without it
SUMO sampled a speed factor per vehicle, so the ego's top speed varied
by seed (up to 50 m/s on a 36.11 m/s road).

## Reward — 6 components

`laneiq/env/rewards.py:compute_reward` returns
`(total, components)`. Components sum exactly to total (no rounding).

| Component | Default weight | Sign | Triggers |
| --- | --- | --- | --- |
| `speed` | `+1.0` | + | `w_speed × min(speed/v_max, 1.0)` per step |
| `lane_cost` | `-0.05` | − | `w_lane_cost` per non-`STAY` action (requested) |
| `jerk` | `-0.2` | − | `w_jerk × |Δaccel| / a_max_norm` |
| `unsafe_gap` | `-0.5` | − | `w_unsafe_gap` if `front_thw < 1.5s` |
| `collision` | `-50.0` | − | terminal; dominant magnitude |
| `alive` | `+0.01` | + | dense per-step bonus (anti-suicide) |

The canonical PPO uses the `safety_heavy` weights (`w_collision=-100`,
`w_unsafe_gap=-1.0`) with `w_lane_cost=-1.0`. A lane change used to cost
5% of one step's speed reward, which is effectively free. The value
came from a −0.05 / −0.3 / −1.0 sweep.

**The pitfall: speed reward on a fixed distance.** Episodes end at the
1 km mark, and the speed term is paid per step. Summed over an episode it
is about `distance / (v_max · dt)`, the same for every driver below the
limit. Going faster just ends the episode sooner, and the alive bonus
pays for every extra second. So **episode return is flat to negative in
speed** and is not a measure of driving quality here. Evals rank drivers
by **finish time**.

**A fix that didn't work.** Turning `alive` into a per-step time penalty
(`w_alive=-0.19`, so finishing 1 s sooner earns +2.0) and retraining made
PPO *slower*, not faster. The likely cause: with the safety weights,
merging in front of traffic triggers `unsafe_gap` (−1 per step under
1.5 s headway), which costs more than the time a pass saves. Those
checkpoints weren't promoted. The next attempt should rebalance
`unsafe_gap` against a time or progress term.

See [`results.md`](results.md) for the numbers.

## Algorithm 1 — DQN from scratch

`laneiq/agents/dqn/`. PyTorch implementation including:

- **Q-network** — 25 → 128 → 128 → 3 MLP, ReLU activations, 17k parameters
- **Replay buffer** — uniform sampling from a 100k-transition deque (pre-allocated arrays for speed)
- **Target network** — hard-copied from the online net every 1000 env steps for Bellman-target stability
- **ε-greedy exploration** — linear decay from 1.0 → 0.05 over the first 50k steps
- **Double DQN** — target uses `Q(s', argmax_a Q(s', a))` from the online net, value from the target net (mitigates Q-overestimation)
- **Huber loss** — see "Tribal knowledge" below

Training: `scripts/train_dqn.py`. CartPole-v1 is the gate test — the
exact same code trained on `gym.make("CartPole-v1")` reaches solved
(195+ over 100 episodes) within ~50k steps. That gate has to stay
green before any LaneIQ-specific tweak ships.

### Tribal knowledge — Huber loss

Mid-Week-3, DQN on LaneIQ plateaued at mean-100 return ≈ 175 with
60% high-density collisions. Investigation:

- `train/loss`: mean **290**, max **2387** (should be O(1))
- `train/grad_norm`: mean **655**, max **4466** (clipped at 10 every step)
- `train/mean_q`: saturates at 320–360 (max episode return is ~268, so Q-values overestimating by ~40%)

Root cause: MSE loss on `(Q - target)²` where the target sometimes
includes a `-50` collision reward → squared TD error explodes →
gradient norm 1000+ → constant clipping at 10 → effective LR is ~1%
of intended.

Fix: switch to Huber (smooth L1). The linear-for-large-errors
regime tolerates the `-50` outlier without exploding gradients.
After Huber:

- `train/loss`: mean **0.31**, max **2.44** (935× lower)
- `train/grad_norm`: mean **2.32** (282× lower)
- `train/mean_q`: saturates at 57 (sensible — well below max return 268)

Important nuance: Huber loss **hurt** CartPole (mean100 dropped 268
→ 155). CartPole has no reward outliers (`+1/step`); MSE's quadratic
push converges faster there. Made `huber_loss: bool = True` a config
option; the CartPole gate uses MSE.

**Generalizable lesson:** Huber is the right default for any RL
problem with sparse high-magnitude rewards; MSE is the right default
for bounded continuous rewards.

## Algorithm 2 — PPO via SB3

`laneiq/agents/ppo_sb3.py:PPOAgent` wraps `sb3_contrib.MaskablePPO`
to implement the same `Policy` protocol the eval harness expects.

Hyperparameters (`PPOConfig` in the same file):

- `lr=3e-4` (SB3 default; PPO is sensitive to higher rates)
- `n_steps=2048` (rollout buffer per update)
- `batch_size=64`, `n_epochs=10`
- `gamma=0.99`, `gae_lambda=0.95`, `clip_range=0.2`
- `ent_coef=0.01` (entropy bonus encourages exploration)
- `vf_coef=0.5`, `max_grad_norm=0.5`
- `net_arch=(128, 128)` (matches DQN for fair comparison)

**Why MaskablePPO not vanilla:** action masking via
`env.action_masks()` zero-probabilities invalid actions before
sampling. Without this, the policy occasionally proposes
`CHANGE_RIGHT` when in lane 0 — which the env silently no-ops, but
wastes exploration.

**Why a library and not from-scratch:** PPO has too many subtle
implementation details (clipping at the right place, GAE bootstrap
on truncation, value-function clipping) for a single-file reimpl
to match SB3's reference within a project budget. The DQN code is
the "I implemented an algorithm" portfolio piece; PPO is the "I
can pick the right tool" piece.

## Algorithm 3 — Parameter-sharing MAPPO

`scripts/train_mappo.py` trains a single shared policy across N=4
concurrent egos in one SUMO scene. Pipeline:

1. `LaneIQParallelEnv` (PettingZoo `ParallelEnv`) — N egos, each
   getting their own per-agent observation / action / reward.
2. `supersuit.black_death_v3(par_env)` — keeps terminated agents in
   the API (zero observations, no-op actions accepted) so vec-env
   wrappers don't choke on changing agent counts.
3. `pettingzoo_env_to_vec_env_v1` + `concat_vec_envs_v1` —
   flatten the multi-agent env into an SB3-compatible VecEnv.
4. **Plain `stable_baselines3.PPO`** (NOT MaskablePPO) — supersuit
   doesn't propagate action masks through the multi-agent wrapping.
   The env turns an invalid or cooling-down request into `STAY`, and
   slot 23 shows each ego when a change isn't available.

The shared policy network has the same shape as single-agent PPO
(25 → 128 → 128 → 3), so the same checkpoint can drive a single
ego in `LaneIQEnv` (which is what the live demo does — see
`backend/app/services/policy_loader._MAPPOEvalAgent`).

## Eval methodology

The 5 single-agent drivers are evaluated on a `5 × 3 × 20 = 300`
matrix (agents × densities × seeds), plus the gauntlet at 20 seeds.
Each (agent, density, seed) cell runs one episode end-to-end with the
same `EpisodeMetrics` collector, so there's no agent-specific logic in
the metrics layer. Lane changes are counted from
`info["executed_action"]`, i.e. changes SUMO actually made.

Aggregated CSV columns: mean ± std per metric, plus collision rate
and reached-end rate. Per-episode CSV available for "why is one
number weird?" investigations.

**Primary metric: finish time per km.** Under mode 768 every driver is
collision-free, and return can't see speed (above), so the comparison
is who gets through traffic fastest. Collisions stay in the table as a
sanity check.

**For safety-critical RL, never trust a 0% rate at <10 seeds.** A true
15% Bernoulli has a 44% chance of looking like 0% in 5 trials. A
5-seed eval hid exactly that (15% high-density collisions in an early
PPO) until the 20-seed re-run.

See [`results.md`](results.md) for the actual numbers.
