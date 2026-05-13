"""Discover pre-recorded replay JSON files under `results/replays/`.

Replays are produced by `scripts/record_replay.py` and played back by
`SimulatorService.start_replay`. The wire format inside each JSON
matches `SimTick` exactly (one entry per recorded tick), with
top-level metadata (agent, density, seed, n_ticks) that this loader
exposes to the frontend via `GET /replays`.
"""

from __future__ import annotations

import json
from pathlib import Path

from backend.app.schemas import ReplayListEntry

_REPO_ROOT: Path = Path(__file__).resolve().parents[3]
REPLAY_DIR: Path = _REPO_ROOT / "results" / "replays"


def list_replays() -> list[ReplayListEntry]:
    """Enumerate replay JSON files. Tolerates a missing dir."""
    out: list[ReplayListEntry] = []
    if not REPLAY_DIR.is_dir():
        return out
    for p in sorted(REPLAY_DIR.glob("*.json")):
        try:
            payload = json.loads(p.read_text())
            out.append(ReplayListEntry(
                name=p.stem,
                agent=str(payload.get("agent", "?")),
                density=str(payload.get("density", "?")),
                n_ticks=int(payload.get("n_ticks", 0)),
                size_bytes=p.stat().st_size,
            ))
        except (OSError, ValueError, KeyError):
            # Skip malformed files rather than 500 the whole endpoint.
            continue
    return out


def resolve_replay(name: str) -> Path:
    """Map a replay name to its file path. Raises FileNotFoundError."""
    candidate = REPLAY_DIR / f"{name}.json"
    if not candidate.is_file():
        raise FileNotFoundError(
            f"no replay matching {name!r} under {REPLAY_DIR}",
        )
    return candidate
