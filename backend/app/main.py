"""LaneIQ live-demo FastAPI app.

Run locally:
    uv run uvicorn backend.app.main:app --reload --port 8000

Then the React frontend at frontend/src/api/* connects to:
    POST   http://localhost:8000/sim/start
    POST   http://localhost:8000/sim/stop
    GET    http://localhost:8000/sim/status
    GET    http://localhost:8000/runs
    WS     ws://localhost:8000/sim/stream
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.routes import runs, sim, stream
from backend.app.services.simulator import SIM_SERVICE

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
_log = logging.getLogger("laneiq.backend")


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Process-lifetime hooks. Cleanly stop any running sim on shutdown."""
    _log.info("backend starting")
    try:
        yield
    finally:
        _log.info("backend shutting down; stopping any running sim")
        await SIM_SERVICE.stop()


app = FastAPI(
    title="LaneIQ live demo",
    version="0.1.0",
    description=(
        "Backend for the LaneIQ React dashboard. Streams a live SUMO "
        "simulation driven by a trained RL agent."
    ),
    lifespan=_lifespan,
)

# CORS — frontend runs on a different port (vite default 5173).
# For local dev allow everything; production deployment would tighten this.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes.
app.include_router(sim.router)
app.include_router(stream.router)
app.include_router(runs.router)


@app.get("/healthz", tags=["health"])
def healthz() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}
