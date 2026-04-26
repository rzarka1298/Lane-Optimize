# Project-Documentation — working docs index

This tree holds **working documentation that mirrors the code**. It is *not*
recruiter-facing — that lives in `../docs/`. The contract for what goes here,
what an OVERVIEW.md must contain, and the rule about reading-before-editing all
live in [`../CLAUDE.md`](../CLAUDE.md).

## Structure

```
Project-Documentation/
├── README.md                     ← you are here
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

| Path | Status |
| --- | --- |
| `laneiq/OVERVIEW.md` | ✅ Scaffold-only state recorded |
| `concepts/` | empty (first concept note lands when DQN starts in Week 2) |
| `sumo_scenarios/OVERVIEW.md` | ⏳ created during Task #3 (next) |
| `backend/OVERVIEW.md` | ⏳ Week 5 |
| `frontend/OVERVIEW.md` | ⏳ Week 6 |
