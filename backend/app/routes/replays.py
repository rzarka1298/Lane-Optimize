"""GET /replays — list pre-recorded episodes for the frontend's picker."""

from __future__ import annotations

from fastapi import APIRouter

from backend.app.schemas import ReplayListEntry
from backend.app.services.replay_loader import list_replays

router = APIRouter(tags=["replays"])


@router.get("/replays", response_model=list[ReplayListEntry])
def get_replays() -> list[ReplayListEntry]:
    """Enumerate replays under `results/replays/`. Empty list if none recorded."""
    return list_replays()
