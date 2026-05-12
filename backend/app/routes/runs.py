"""GET /runs — list available agents/checkpoints for the frontend's picker."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import RunListEntry
from backend.app.services.policy_loader import list_runs

router = APIRouter(tags=["runs"])


@router.get("/runs", response_model=list[RunListEntry])
def get_runs() -> list[RunListEntry]:
    """Enumerate checkpoints under `checkpoints/final/`.

    The three builtin baselines (random, greedy, safe_rule) are NOT in
    this list — they don't have files. The frontend hardcodes them
    in the agent picker.
    """
    return list_runs()
