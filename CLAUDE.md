# LaneIQ — instructions for Claude

This file is auto-loaded into Claude's context on every session in this repo.
**Read it fully each session. Then follow the workflow below.**

---

## Project in one paragraph

LaneIQ is a portfolio-grade RL system: a Gymnasium-wrapped SUMO highway simulator
+ a from-scratch PyTorch DQN + a Stable-Baselines3 PPO + parameter-sharing multi-agent
PPO, with rule-based baselines, a reproducible eval matrix, and a live FastAPI/React
dashboard. It is being built in 8 weeks (~20 hrs/week) by a CS student targeting
big-tech ML and SWE internships. The complete plan lives at
[`~/.claude/plans/i-have-attached-a-humble-balloon.md`](~/.claude/plans/i-have-attached-a-humble-balloon.md);
the original PRD is at [`Lane-Optimize-PRD.pdf`](Lane-Optimize-PRD.pdf).

## Locked-in technical decisions (do not relitigate without asking)

- **Simulator:** SUMO via `traci` everywhere (training + live demo). `libsumo` is intentionally
  *not* used: no macOS arm64 wheel ships on PyPI as of 1.26. Week 3 profiling will decide
  whether the ~8× speed-up justifies a from-source libsumo build. Until then, single runtime
  path keeps the env code simple.
- **DQN:** from scratch in PyTorch. **PPO:** via Stable-Baselines3.
- **Multi-agent:** parameter-sharing PPO over a `pettingzoo.ParallelEnv` (committed stretch goal).
- **Compute:** Apple Silicon, MPS backend. No CUDA assumptions.
- **Stack:** Python 3.11 + uv, FastAPI backend, Vite + React + TypeScript frontend, Canvas2D, TensorBoard.

## Working mode (durable)

- **Pair-program style.** Before writing a non-trivial file, give a 2–4 sentence preface:
  what we're about to build and why. After writing, point to the lines that matter.
  The user reads every diff. Don't dump 500 lines without narration.
- **Explain conceptual milestones** as they come up: DQN math, observation design,
  reward shaping, PPO clipping + GAE, multi-agent parameter sharing, WS protocol
  design, libsumo vs TraCI trade-offs.
- **Don't push to GitHub.** Local commits only until the user says otherwise.
- **No `Co-Authored-By: Claude` (or any AI-attribution trailer) in commit messages or
  PR bodies.** Author commits as if they were the user's own. Drop the default
  Claude Code trailer template entirely.

---

## 🚦 The Project-Documentation gate (HARD RULE)

> **Note for fresh clones / contributors:** `Project-Documentation/` is
> intentionally **gitignored** — it lives only on the original author's
> machine. If the folder doesn't exist in your working tree, the gate
> below does not apply; recruiter-facing polish lives in `docs/` instead.

`Project-Documentation/` mirrors the code structure (`Project-Documentation/laneiq/env/`
↔ `laneiq/env/`, etc.) and holds working docs that **must stay in sync with the
code at all times**. It is separate from `docs/` (which is recruiter-facing).

**Before** editing or creating code in any folder, you **must**:

1. Read `Project-Documentation/README.md` (the index — quick, always).
2. Read the `OVERVIEW.md` for any folder you are about to touch, plus any
   sibling/parent OVERVIEW.md whose invariants your change might affect.
3. Read concept notes in `Project-Documentation/concepts/` only if relevant to
   the change (e.g., touching DQN → read the DQN concept note).

**After** the code change, in the same response:

4. If the change invalidates anything in the OVERVIEW.md you read — **update it**.
   This is not optional. Stale docs are worse than no docs.
5. If the folder you changed has no OVERVIEW.md yet and the code is now non-trivial
   (more than scaffolding), **create one**.

**Exceptions** (no doc-read needed):
- Trivial edits: typos, one-line bugfix in code that has no OVERVIEW.md yet.
- Files outside `laneiq/`, `backend/`, `frontend/`, `sumo_scenarios/` (e.g., `.gitignore`,
  `Makefile`, `pyproject.toml`) — these don't have OVERVIEW.md gates.

**Rule of thumb:** if a future Claude sessions reads only the OVERVIEW.md and the code,
they should understand what's there and *what not to break*. Write for that reader.

---

## Doc shapes inside `Project-Documentation/`

Two kinds of mirrored docs per code folder:

### 1. `OVERVIEW.md` — folder summary

One per code folder. The "lobby" for that folder. Contains:

1. **Purpose** — one paragraph: what this folder is for.
2. **Status** — current implementation state (scaffold / partial / complete).
3. **Files in this folder** — bulleted list of the per-file docs that live
   alongside this OVERVIEW.md, with one-line summaries. Acts as the index.
4. **Public API** — what other parts of the codebase import from here.
5. **Invariants & gotchas** — folder-wide "don't change X without Y" rules.
   Per-file gotchas live in the per-file doc.
6. **Last updated** — ISO date + git SHA prefix of the most recent commit
   that touched anything in this folder.

Keep OVERVIEW.md ≤120 lines. It's an index, not a textbook.

### 2. Per-file `<filename>.md` — detailed component doc

One per non-trivial code file (or per logical component for grouped XML
configs — e.g., `routes.md` covers all three `routes_*.rou.xml`). Contains:

1. **What** — one paragraph: what this file/component does.
2. **Public API / Schema** — the exported functions, classes, or XML
   elements with their signatures or attribute lists.
3. **Design notes** — why it's shaped this way; alternatives considered and
   rejected; any non-obvious trade-offs. This is where deep content lives.
4. **Invariants & gotchas** — file-specific rules. Things that would surprise
   a new reader.
5. **Tests / how to verify** — what tests cover this and how to run them.
6. **Last updated** — ISO date + git SHA prefix.

Keep per-file docs ≤150 lines. If one balloons past that, it probably
should be split into a concept note in `Project-Documentation/concepts/`
plus a leaner per-file doc.

### When to skip

- Trivial files (single-line `__init__.py`, `.gitkeep`, generated outputs
  like `highway.net.xml`) don't get their own doc — mention them in the
  parent OVERVIEW.md "Files in this folder" list and that's enough.
- Cross-cutting concerns that don't belong to any single file (dev
  environment, deployment, training reproducibility) live as **top-level
  docs** alongside `setup.md`, not buried in a code-folder mirror.

---

## Workflow checklist (use on every code task)

- [ ] Read this file (you should have already on session start).
- [ ] Read `Project-Documentation/README.md`.
- [ ] Read OVERVIEW.md for every folder you intend to touch.
- [ ] Pause and narrate: "Here's what I'm about to build and why" (pair-program style).
- [ ] Make the code change.
- [ ] Update or create OVERVIEW.md in the same response.
- [ ] Run lints/tests if the change is non-trivial.
- [ ] Commit only if the user explicitly asks.

---

## Quick reference

| Path | What |
| --- | --- |
| `laneiq/` | Main Python package |
| `sumo_scenarios/` | Hand-authored SUMO XML (nod/edg/net + vTypes + routes) |
| `backend/` | FastAPI live-demo server |
| `frontend/` | Vite + React + TypeScript dashboard |
| `configs/` | OmegaConf YAMLs (env / reward / agent variants) |
| `scripts/` | Top-level CLI entrypoints (`train_dqn.py`, etc.) |
| `tests/` | Pytest suite |
| `docs/` | **Recruiter-facing.** Polished architecture / results / RL-design write-ups. Linked from README. |
| `Project-Documentation/` | **Working docs.** Read before editing; update after. |
| `Lane-Optimize-PRD.pdf` | Original product requirements |
