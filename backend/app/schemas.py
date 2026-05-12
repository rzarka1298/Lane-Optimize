"""Wire-format pydantic models shared by every endpoint.

These define the JSON contract the React frontend depends on. Adding a
field is fine (frontend ignores unknown); removing or renaming a field
is a breaking change — update both ends together.

Tight by design: ~2KB per `SimTick` at 30 vehicles. No protobuf, no
compression — JSON over WebSocket is fine for our 10 Hz rate.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# HTTP request/response models
# ---------------------------------------------------------------------------


class StartSimRequest(BaseModel):
    """POST /sim/start body."""

    agent: str = Field(
        description="agent name — must match a checkpoint or builtin baseline",
        examples=["ppo", "dqn", "random", "greedy", "safe_rule"],
    )
    density: Literal["low", "medium", "high"] = "medium"
    seed: int | None = Field(default=None, description="reproducibility seed")
    max_steps: int = Field(default=1000, ge=10, le=10_000)


class StartSimResponse(BaseModel):
    run_id: str
    agent: str
    density: str
    seed: int | None


class SimStatusResponse(BaseModel):
    """Current backend sim state — for the frontend's status indicator."""

    running: bool
    run_id: str | None = None
    agent: str | None = None
    density: str | None = None
    seed: int | None = None
    step: int = 0
    sim_time_s: float = 0.0


class RunListEntry(BaseModel):
    """One row of GET /runs — a checkpoint the user can pick."""

    name: str          # e.g. "ppo_v1_mixed_500k"
    algorithm: str     # "ppo" / "dqn"
    path: str          # absolute path on backend filesystem
    size_bytes: int


# ---------------------------------------------------------------------------
# WebSocket /sim/stream payloads
# ---------------------------------------------------------------------------


class Vehicle(BaseModel):
    """One vehicle's state at a single tick."""

    id: str
    x: float                          # m (longitudinal along edge)
    y: float                          # m (lateral; lane center + lat offset)
    speed: float                      # m/s
    lane: int                         # 0 (right) .. 2 (left)
    is_ego: bool
    vtype: Literal["aggressive", "normal", "slow", "ego"]


class Decision(BaseModel):
    """Decision-explanation tag attached to ticks where ego acted."""

    action: Literal["stay", "change_left", "change_right"]
    explanation: str                  # template-filled human-readable string
    q_values: tuple[float, float, float] | None = None  # DQN only


class SimTick(BaseModel):
    """One frame of the simulation, pushed over `WS /sim/stream`."""

    t: float                          # sim time in seconds
    step: int                         # env step count
    ego_id: str
    vehicles: list[Vehicle]
    decision: Decision | None = None
    terminated: bool = False
    truncated: bool = False
    info: dict[str, float] = Field(default_factory=dict)  # reward components, etc.


class StreamFrame(BaseModel):
    """Top-level WS message envelope.

    Frontend parses `type` first to route to handlers. Discriminator
    pattern lets us add new frame types (e.g. 'metrics', 'error') without
    breaking older clients.
    """

    type: Literal["tick", "started", "stopped", "error"]
    tick: SimTick | None = None
    message: str | None = None
