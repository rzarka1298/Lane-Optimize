# Project-Documentation — working docs index

This tree holds **working documentation that mirrors the code**. It is *not*
recruiter-facing — that lives in `../docs/`. The contract for what goes
here, what each doc must contain, and the rule about reading-before-editing
all live in [`../CLAUDE.md`](../CLAUDE.md).

## Structure

```
Project-Documentation/
├── README.md                       ← you are here (the index)
├── setup.md                        ← cross-cutting: dev env, SUMO install, full
│                                     test/verification walkthrough, MPS notes
├── concepts/                       ← cross-cutting learning notes
│   └── 01-sumo-traffic-model.md    ← Krauss + LC2013 deep-dive
├── laneiq/                         ← mirrors laneiq/ Python package
│   ├── OVERVIEW.md
│   └── env/
│       ├── OVERVIEW.md             ← folder summary
│       └── sumo_runtime.md         ← per-file detail
├── sumo_scenarios/                 ← mirrors sumo_scenarios/
│   ├── OVERVIEW.md                 ← folder summary
│   ├── network.md                  ← nodes + edges + generated net
│   ├── vtypes.md                   ← vehicle profiles
│   ├── routes.md                   ← Poisson flows × 3 densities
│   └── config.md                   ← sumocfg + build.sh
├── backend/                        ← (Week 5)
└── frontend/                       ← (Week 6)
```

## Three doc shapes

1. **`OVERVIEW.md`** — one per code folder. The "lobby." Lists the per-file
   docs in that folder, gives a folder-wide status, and codifies invariants
   that span multiple files. Keep ≤120 lines.
2. **Per-file `<filename>.md`** — one per non-trivial code file (or per
   logical component for tightly-coupled XML configs). Detailed: schema,
   public API, design notes, file-specific gotchas, tests. Keep ≤150 lines.
3. **Top-level cross-cutting docs** (e.g., `setup.md`) — for concerns that
   don't belong to any single code folder: dev environment, deployment,
   training reproducibility. Live alongside this README, not inside a
   code-mirrored folder.

`OVERVIEW.md` and per-file docs only exist for folders/files with
non-trivial code. Empty scaffolding folders don't get docs until they hold
real code — empty "TBD" docs are noise.

Plus **concept notes** in `concepts/` — cross-cutting, learning-flavored
explainers for conceptual milestones (DQN math, PPO + GAE, multi-agent
parameter sharing, etc.). Numbered (`01-...`, `02-...`) for stable
ordering; reference each one by number from per-file docs.

## What is NOT maintained here

- **ADRs** (decision records) — locked decisions live in `CLAUDE.md`; new
  forks are resolved in chat.
- **Activity logs / changelogs** — git log and commit messages serve this
  role.
- **Recruiter-facing content** — that's `../docs/` (architecture diagrams,
  results tables, polished RL design write-up, demo video).

## Index of current docs

### Top-level / cross-cutting

| Path | Status | Last touched |
| --- | --- | --- |
| [`setup.md`](setup.md) | ✅ Dev env + install + full verification walkthrough | `d9d8601` |

### Concept notes

| Path | Status | Last touched |
| --- | --- | --- |
| [`concepts/01-sumo-traffic-model.md`](concepts/01-sumo-traffic-model.md) | ✅ Krauss + LC2013 explainer; vType design rationale | `494e850` |
| [`concepts/02-reward-shaping.md`](concepts/02-reward-shaping.md) | ✅ 6-component reward design; failure modes; Pareto sweep narrative | `ee53ff5` |

### `laneiq/` (Python package)

| Path | Status | Last touched |
| --- | --- | --- |
| [`laneiq/OVERVIEW.md`](laneiq/OVERVIEW.md) | ✅ Scaffold-only state recorded | `be30837` |
| [`laneiq/env/OVERVIEW.md`](laneiq/env/OVERVIEW.md) | ✅ `sumo_runtime.py` landed; rest pending Week 2 | `d9d8601` |
| [`laneiq/env/sumo_runtime.md`](laneiq/env/sumo_runtime.md) | ✅ Process lifecycle helpers documented | `d9d8601` |
| [`laneiq/env/observations.md`](laneiq/env/observations.md) | ✅ 25-dim obs vector; `build_observation` + `ObservationBuilder` | `0477f0d` |
| [`laneiq/env/actions.md`](laneiq/env/actions.md) | ✅ `Action` enum + `apply_action` + `valid_actions_mask` + masking | `b4d7fd1` |
| [`laneiq/env/rewards.md`](laneiq/env/rewards.md) | ✅ `RewardConfig` + `compute_reward` (scalar + per-component dict) | `ee53ff5` |
| [`laneiq/env/highway_env.md`](laneiq/env/highway_env.md) | ✅ `LaneIQEnv(gymnasium.Env)` — passes `env_checker` | `7c6a6e9` |
| `laneiq/agents/dqn/` | ⏳ Tasks #12–13 |  — |
| `laneiq/agents/baselines/` | ⏳ Task #11 |  — |
| `laneiq/eval/` | ⏳ Week 3 |  — |
| `laneiq/multi_agent/` | ⏳ Week 5 |  — |

### `sumo_scenarios/`

| Path | Status | Last touched |
| --- | --- | --- |
| [`sumo_scenarios/OVERVIEW.md`](sumo_scenarios/OVERVIEW.md) | ✅ Trimmed to summary; per-file docs split out | `d9d8601` |
| [`sumo_scenarios/network.md`](sumo_scenarios/network.md) | ✅ Nodes + edges + net | `d9d8601` |
| [`sumo_scenarios/vtypes.md`](sumo_scenarios/vtypes.md) | ✅ Four vehicle profiles | `d9d8601` |
| [`sumo_scenarios/routes.md`](sumo_scenarios/routes.md) | ✅ Poisson flows × 3 densities | `d9d8601` |
| [`sumo_scenarios/config.md`](sumo_scenarios/config.md) | ✅ sumocfg + build.sh | `d9d8601` |

### Other

| Path | Status | Last touched |
| --- | --- | --- |
| `backend/` | ⏳ Week 5 |  — |
| `frontend/` | ⏳ Week 6 |  — |
