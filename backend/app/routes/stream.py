"""WebSocket /sim/stream — pushes per-tick SimTick frames at ~10 Hz."""

from __future__ import annotations

import asyncio
import logging
from contextlib import suppress

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.app.schemas import StreamFrame
from backend.app.services.simulator import SIM_SERVICE

router = APIRouter(tags=["stream"])
_log = logging.getLogger(__name__)


@router.websocket("/sim/stream")
async def sim_stream(ws: WebSocket) -> None:
    """Stream live sim ticks. Client connects AFTER /sim/start.

    Per-tick payload: see `SimTick` in schemas.py. Plus 'started' /
    'stopped' / 'error' control frames bracketing the episode.
    """
    await ws.accept()
    if not SIM_SERVICE.is_running:
        await ws.send_json(StreamFrame(
            type="error", message="no sim running; POST /sim/start first",
        ).model_dump(mode="json"))
        await ws.close(code=4004)
        return

    queue = SIM_SERVICE.subscribe()
    try:
        while True:
            frame: StreamFrame = await queue.get()
            await ws.send_json(frame.model_dump(mode="json", exclude_none=True))
            # End-of-episode control frames close the stream.
            if frame.type in ("stopped", "error"):
                await ws.close()
                return
    except WebSocketDisconnect:
        _log.info("websocket disconnected")
    except asyncio.CancelledError:
        raise
    except Exception:
        _log.exception("error in sim stream")
        with suppress(Exception):
            await ws.close(code=4500)
    finally:
        SIM_SERVICE.unsubscribe(queue)
