"""Backend skeleton tests — exercise endpoints without spinning up SUMO.

The simulator service starts an env on POST /sim/start, which DOES start
SUMO. Those tests are marked @slow @sumo. The fast tests verify the
endpoint surface, schema validation, and error paths only.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.policy_loader import _CHECKPOINT_DIR, list_runs


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


# ---------------------------------------------------------------------------
# Fast: pure endpoint / schema tests
# ---------------------------------------------------------------------------


def test_healthz(client: TestClient) -> None:
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_status_when_no_sim_running(client: TestClient) -> None:
    r = client.get("/sim/status")
    assert r.status_code == 200
    data = r.json()
    assert data["running"] is False
    assert data["step"] == 0


def test_runs_endpoint_returns_list(client: TestClient) -> None:
    r = client.get("/runs")
    assert r.status_code == 200
    runs = r.json()
    assert isinstance(runs, list)
    # We've checkpoint'd PPO + DQN canonical artifacts; expect at least one.
    if _CHECKPOINT_DIR.is_dir():
        expected_count = len(list_runs())
        assert len(runs) == expected_count


def test_runs_each_entry_has_required_fields(client: TestClient) -> None:
    r = client.get("/runs")
    for entry in r.json():
        assert "name" in entry
        assert "algorithm" in entry
        assert entry["algorithm"] in ("dqn", "ppo")
        assert "path" in entry
        assert entry["size_bytes"] > 0


def test_start_sim_rejects_unknown_agent(client: TestClient) -> None:
    r = client.post("/sim/start", json={
        "agent": "nonexistent_agent_xyz",
        "density": "medium",
        "seed": 0,
    })
    assert r.status_code == 404


def test_start_sim_rejects_invalid_density(client: TestClient) -> None:
    r = client.post("/sim/start", json={
        "agent": "random",
        "density": "extreme",  # not in Literal
        "seed": 0,
    })
    assert r.status_code == 422  # pydantic validation


def test_start_sim_rejects_negative_max_steps(client: TestClient) -> None:
    r = client.post("/sim/start", json={
        "agent": "random",
        "density": "medium",
        "max_steps": -1,
    })
    assert r.status_code == 422


def test_stop_sim_idempotent(client: TestClient) -> None:
    """Stopping when nothing's running returns 200, not 409."""
    r1 = client.post("/sim/stop")
    r2 = client.post("/sim/stop")
    assert r1.status_code == 200
    assert r2.status_code == 200
    assert r1.json() == r2.json() == {"status": "stopped"}


def test_websocket_stream_rejects_when_no_sim(client: TestClient) -> None:
    """Connecting to /sim/stream with no sim sends an error frame + closes."""
    with client.websocket_connect("/sim/stream") as ws:
        frame = ws.receive_json()
        assert frame["type"] == "error"
        assert "no sim running" in frame["message"].lower()


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------


def test_start_sim_request_defaults() -> None:
    from backend.app.schemas import StartSimRequest

    req = StartSimRequest(agent="random")
    assert req.density == "medium"
    assert req.seed is None
    assert req.max_steps == 1000


def test_vehicle_schema_round_trip() -> None:
    from backend.app.schemas import Vehicle

    v = Vehicle(id="ego_0", x=100.0, y=3.2, speed=20.0, lane=1,
                is_ego=True, vtype="ego")
    d = v.model_dump()
    v2 = Vehicle.model_validate(d)
    assert v == v2


def test_simtick_serializes_to_compact_json() -> None:
    """Sanity check: SimTick payload is small enough for 10 Hz streaming."""
    from backend.app.schemas import Decision, SimTick, Vehicle

    vehicles = [
        Vehicle(id=f"v{i}", x=i * 30.0, y=(i % 3) * 3.2, speed=25.0,
                lane=i % 3, is_ego=(i == 0), vtype="normal")
        for i in range(30)
    ]
    tick = SimTick(
        t=10.0, step=100, ego_id="v0", vehicles=vehicles,
        decision=Decision(action="stay", explanation="ok"),
    )
    payload = tick.model_dump_json()
    # Hard ceiling at ~5KB; in practice expect ~2KB.
    assert len(payload) < 5000, f"SimTick too large: {len(payload)} bytes"
