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
`Box(25,)` vector per step. Pure function over a TraCI snapshot
(no hidden state).

| Indices | Component | Meaning |
| --- | --- | --- |
| 0–4 | Ego self-state | speed (norm.), accel (norm.), lane idx (0/1/2 → 0/0.5/1), lateral offset, headway to leader |
| 5–14 | Per-lane neighbors | for each of 3 lanes: leader gap + leader speed + follower gap + follower speed (zeros if no leader/follower) |
| 15–22 | Lane-by-lane density / mean speed | aggregate over a 100m window |
| 23 | Lane-validity packed scalar | encodes `(left_ok, right_ok)` as `pack ∈ {0, 1, 2, 3}` then `pack/3` |
| 24 | Time-to-end | normalized fraction of road left |

All values are clipped to `[-1, 1]` or `[0, 1]` per the relevant
SUMO physics. Normalization constants live in
`laneiq/env/observations.py` so changing them is a one-place edit.

**Why 25 not larger:** the trained PPO converges in 150k steps with
this much. Larger obs = bigger network = more compute, no signal that
extra signal would help. The lane-validity packed scalar (slot 23) is
the cleverest dimension — it encodes the action mask the policy needs
in a single float so the obs vector stays compact for both DQN
(which uses `mask_invalid_q_values`) and MaskablePPO (which reads
`env.action_masks()`).

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

The env disables SUMO's internal lane-change motivations on the
ego (`traci.vehicle.setLaneChangeMode(ego, 0)`), so every lane
change is the policy's decision. **The ego CAN crash from a bad
policy choice.** That's intentional — the collision reward teaches
it not to.

## Reward — 6 components

`laneiq/env/rewards.py:compute_reward` returns
`(total, components)`. Components sum exactly to total (no rounding).

| Component | Default weight | Sign | Triggers |
| --- | --- | --- | --- |
| `speed` | `+1.0` | + | `w_speed × min(speed/v_max, 1.0)` per step |
| `lane_cost` | `-0.05` | − | `w_lane_cost` per non-`STAY` action |
| `jerk` | `-0.2` | − | `w_jerk × |Δaccel| / a_max_norm` |
| `unsafe_gap` | `-0.5` | − | `w_unsafe_gap` if `front_thw < 1.0s` |
| `collision` | `-50.0` | − | terminal; dominant magnitude |
| `alive` | `+0.1` | + | dense per-step bonus (anti-suicide) |

The Week-4 sweep tested four reward configurations. The big finding
(Week 7, after re-running at 20 seeds): **reward shaping > training
duration**. The `safety_heavy` variant
(`w_collision=-100`, `w_unsafe_gap=-1.0`) reached genuine universal
0% collisions in 150k training steps; the default-weight PPO needed
500k AND still missed 15% of high-density episodes that the 5-seed
Week-4 eval failed to surface.

See [`results.md`](results.md) for the v1-vs-v2 comparison numbers.

## Algorithm 1 — DQN from scratch

`laneiq/agents/dqn/`. PyTorch implementation including:

- **Q-network** — 25 → 128 → 128 → 3 MLP, ReLU activations, 17k parameters
- **Replay buffer** — uniform sampling from a 100k-transition deque (pre-allocated arrays for speed)
- **Target network** — soft-updated every 1000 env steps for Bellman-target stability
- **ε-greedy exploration** — linear decay from 1.0 → 0.05 over the first 10k steps
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
   The env's silent-no-op behavior on invalid lane targets handles
   it instead.

The shared policy network has the same shape as single-agent PPO
(25 → 128 → 128 → 3), so the same checkpoint can drive a single
ego in `LaneIQEnv` (which is what the live demo does — see
`backend/app/services/policy_loader._MAPPOEvalAgent`).

## Eval methodology

The 5 canonical agents are evaluated on a `5 × 3 × 20 = 300`
matrix (agents × densities × seeds). Each (agent, density, seed)
cell runs one episode end-to-end with the same `EpisodeMetrics`
collector — no agent-specific logic in the metrics layer.

Aggregated CSV columns: mean ± std per metric, plus collision rate
and reached-end rate. Per-episode CSV available for "why is one
number weird?" investigations.

The 20-seed bump (vs Week-4's 5 seeds) revealed a 15% collision
rate in `ppo_v1_mixed_500k` at high density that 5-seed sampling
masked as 0%. **For safety-critical RL, never trust a 0% rate at
<10 seeds** — a true 15% Bernoulli has a 44% chance of looking like
0% in 5 trials. Promote to 20 seeds before any "this is universally
safe" claim.

See [`results.md`](results.md) for the actual numbers.
