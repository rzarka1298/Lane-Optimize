# Project-Documentation — working docs index

This tree holds **working documentation that mirrors the code**. It is *not*
recruiter-facing — that lives in `../docs/`. The contract for what goes here,
what an OVERVIEW.md must contain, and the rule about reading-before-editing all
live in [`../CLAUDE.md`](../CLAUDE.md).

## Structure

```
Project-Documentation/
├── README.md                     ← you are here (the index)
├── setup.md                      ← cross-cutting: dev environment, SUMO install, MPS notes
├── concepts/                     ← cross-cutting learning notes (DQN math, PPO, etc.)
│   └── README.md                 ← concept notes index (created when first note lands)
├── laneiq/                       ← mirrors laneiq/ Python package
│   ├── OVERVIEW.md
│   ├── env/OVERVIEW.md
│   ├── agents/OVERVIEW.md
│   ├── agents/dqn/OVERVIEW.md
│   ├── agents/baselines/OVERVIEW.md
│   ├── eval/OVERVIEW.md
│   ├── viz/OVERVIEW.md
│   ├── multi_agent/OVERVIEW.md
│   └── utils/OVERVIEW.md
├── sumo_scenarios/OVERVIEW.md    ← mirrors sumo_scenarios/
├── backend/OVERVIEW.md           ← mirrors backend/
└── frontend/OVERVIEW.md          ← mirrors frontend/
```

Two doc shapes live here:

- **Mirrored OVERVIEW.md** — one per code folder, kept in 1:1 sync with the
  code structure. Read before editing the matching folder.
- **Top-level cross-cutting docs** (e.g., `setup.md`) — for concerns that
  don't belong to any single code folder: dev environment, training
  reproducibility, deployment, etc. Read when their topic is in scope.

OVERVIEW.md files **only exist for folders whose code is non-trivial**. Empty
scaffolding folders don't get an OVERVIEW.md until they hold real code — empty
"TBD" docs are noise.

## Doc types maintained here

Per the user's choice during setup:

- **Component overviews** (one OVERVIEW.md per code folder). Required.
- **Concept notes** (cross-cutting, learning-flavored explainers in `concepts/`).
  Created as conceptual milestones come up — DQN math, PPO clipping objective +
  GAE, parameter sharing, reward shaping intuition, observation design, etc.

Not maintained here (by user choice):

- ADRs (decision records) — locked decisions live in `CLAUDE.md`; new forks are
  resolved in chat with the user.
- Activity logs / changelogs — git log + commit messages serve this role.

## OVERVIEW.md content contract

See `../CLAUDE.md` for the canonical version. In short, every OVERVIEW.md
contains: Purpose · Status · Key files · Public API · Invariants & gotchas ·
Last-updated date + commit SHA prefix.

Keep them 50–200 lines. Writing for a future Claude session that reads only
this doc + the code; they should understand what's there and what not to break.

## Index of current docs

| Path | Status | Last updated |
| --- | --- | --- |
| [`setup.md`](setup.md) | ✅ Dev env + SUMO install + libsumo decision | `2a5f8ff` |
| [`laneiq/OVERVIEW.md`](laneiq/OVERVIEW.md) | ✅ Scaffold-only state recorded | `be30837` |
| [`concepts/01-sumo-traffic-model.md`](concepts/01-sumo-traffic-model.md) | ✅ Krauss + LC2013 explainer; vType design rationale | `494e850` |
| [`sumo_scenarios/OVERVIEW.md`](sumo_scenarios/OVERVIEW.md) | ✅ 3-lane scenario + sumo-gui visual-debug recipe | `2499dd8` |
| `laneiq/env/OVERVIEW.md` | ⏳ Week 2 (LaneIQEnv lands) | — |
| `laneiq/agents/dqn/OVERVIEW.md` | ⏳ Week 2-3 (DQN from scratch) | — |
| `laneiq/agents/baselines/OVERVIEW.md` | ⏳ Week 2 | — |
| `laneiq/eval/OVERVIEW.md` | ⏳ Week 3 (eval matrix) | — |
| `laneiq/multi_agent/OVERVIEW.md` | ⏳ Week 5 | — |
| `backend/OVERVIEW.md` | ⏳ Week 5 | — |
| `frontend/OVERVIEW.md` | ⏳ Week 6 | — |
