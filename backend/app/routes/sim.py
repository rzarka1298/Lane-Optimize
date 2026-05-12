"""HTTP control endpoints for the live sim: start, stop, status."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.schemas import SimStatusResponse, StartSimRequest, StartSimResponse
from backend.app.services.policy_loader import resolve
from backend.app.services.simulator import SIM_SERVICE

router = APIRouter(prefix="/sim", tags=["sim"])


@router.post("/start", response_model=StartSimResponse)
async def start_sim(req: StartSimRequest) -> StartSimResponse:
    """Start a new sim. Conflict (409) if one is already running."""
    if SIM_SERVICE.is_running:
        raise HTTPException(
            status_code=409,
            detail="a sim is already running; POST /sim/stop first",
        )
    try:
        policy = resolve(req.agent)
    except (FileNotFoundError, ValueError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    run_id = await SIM_SERVICE.start(
        agent=policy,
        agent_name=req.agent,
        density=req.density,
        seed=req.seed,
        max_steps=req.max_steps,
    )
    return StartSimResponse(
        run_id=run_id, agent=req.agent, density=req.density, seed=req.seed,
    )


@router.post("/stop")
async def stop_sim() -> dict[str, str]:
    """Stop the currently-running sim. Idempotent."""
    await SIM_SERVICE.stop()
    return {"status": "stopped"}


@router.get("/status", response_model=SimStatusResponse)
async def get_status() -> SimStatusResponse:
    """Current backend sim state."""
    return SimStatusResponse(**SIM_SERVICE.status())
