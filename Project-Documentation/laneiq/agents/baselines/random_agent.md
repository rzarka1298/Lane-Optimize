# `laneiq/agents/baselines/random_agent.py`

## What

Uniform-random policy. The PRD's lowest-bar baseline. If the trained DQN
or PPO doesn't beat this on both mean speed AND collision rate, something
is broken end-to-end — random is the floor.

## Public API

```python
from laneiq.agents.baselines.random_agent import RandomAgent

agent = RandomAgent(seed=42)
agent.reset(seed=99)                       # optional re-seed
action = agent.act(observation)            # uses obs[23] for mask
action = agent.act(observation, action_mask=mask)  # explicit mask
```

## Behavior

- `act(obs)` — derives the validity mask from `obs[23]` via
  `valid_actions_mask(obs)`, samples uniformly over valid actions.
- `act(obs, action_mask=mask)` — uses the explicit mask instead.
- `reset(seed=N)` — re-seeds the internal RNG; without `seed=`, no-op.

The RNG is `numpy.random.Generator` from `np.random.default_rng(seed)`.
Seeding is reproducible: same `seed` → same sequence of actions for the
same sequence of observations.

## Design notes

- **Internal RNG**, not `numpy.random.*` global state. Keeps random
  agents in different evals from polluting each other.
- **STAY fallback** if the mask is all-False (pathological; the obs
  builder guarantees STAY is valid, but defensive).
- **Avoids `np.random.choice` in the all-valid case** — `integers(0, 3)`
  is faster when no filtering is needed.

## Invariants & gotchas

- **Same seed → same actions for same observations.** Tested via
  `test_random_seeding_reproducible` and `test_random_reset_with_seed_reproduces_sequence`.
- **Distribution is approximately uniform** across valid actions over
  many calls. Tested via `test_random_distribution_roughly_uniform`
  (~3000 samples; counts within 200 of 1000).
- **Mask handling matches the env's `action_masks()` semantics.** If you
  change the obs layout in `../../env/observations.md`, the mask recovery
  here adapts automatically (it goes through `valid_actions_mask`).

## Last updated

2026-05-07 · pending commit (initial implementation)
