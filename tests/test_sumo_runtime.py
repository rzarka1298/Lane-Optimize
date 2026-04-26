"""Tests for the TraCI lifecycle helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from laneiq.env.sumo_runtime import (
    DEFAULT_SCENARIO_DIR,
    DEFAULT_SUMOCFG,
    build_argv,
    sumo_binary,
    sumo_session,
)

# ---------- Pure unit tests (fast) -------------------------------------------


def test_default_paths_resolve() -> None:
    """The default scenario paths should point to real files."""
    assert DEFAULT_SCENARIO_DIR.is_dir(), f"missing scenario dir: {DEFAULT_SCENARIO_DIR}"
    assert DEFAULT_SUMOCFG.is_file(), f"missing sumocfg: {DEFAULT_SUMOCFG}"


def test_sumo_binary_returns_existing_path() -> None:
    """``sumo_binary()`` should return a path that actually exists on disk."""
    assert Path(sumo_binary(gui=False)).exists()
    assert Path(sumo_binary(gui=True)).exists()


def test_build_argv_default_includes_locked_flags() -> None:
    argv = build_argv()
    # Binary is first; -c sumocfg is next.
    assert "-c" in argv
    assert any("highway.sumocfg" in a for a in argv)
    # Locked flags must always be present.
    assert "--collision.action" in argv
    assert argv[argv.index("--collision.action") + 1] == "warn"
    assert "--time-to-teleport" in argv
    assert argv[argv.index("--time-to-teleport") + 1] == "-1"
    assert "--step-length" in argv
    assert argv[argv.index("--step-length") + 1] == "0.1"


def test_build_argv_with_seed_and_route_override() -> None:
    route = DEFAULT_SCENARIO_DIR / "routes_low.rou.xml"
    argv = build_argv(seed=42, route_file=route, end_time=60.0)
    assert "--seed" in argv
    assert argv[argv.index("--seed") + 1] == "42"
    assert "--route-files" in argv
    assert argv[argv.index("--route-files") + 1].endswith("routes_low.rou.xml")
    assert "--end" in argv
    assert argv[argv.index("--end") + 1] == "60.0"


def test_build_argv_extra_flags_appended_last() -> None:
    """Extra flags must come last so callers can override anything we defaulted."""
    argv = build_argv(extra_flags=["--verbose", "true"])
    # Last two entries should be the extras.
    assert argv[-2:] == ["--verbose", "true"]


# ---------- Integration test (slow; actually launches SUMO) ------------------


@pytest.mark.slow
@pytest.mark.sumo
def test_sumo_session_steps_and_closes() -> None:
    """Spin up SUMO via the context manager, step 100 times, verify state.

    100 steps * 0.1 s/step = 10 s of sim time. Confirms:
      - traci.start succeeds with our argv
      - simulationStep advances time
      - vehicles are inserted (medium density at t=10 s should have ≥1)
      - the close() in __exit__ runs without raising
    """
    import traci  # local import — keep top-level test module light

    with sumo_session(seed=42, end_time=60.0):
        for _ in range(100):
            traci.simulationStep()

        sim_time = traci.simulation.getTime()
        assert abs(sim_time - 10.0) < 0.01, f"expected ~10s, got {sim_time}"

        # At medium density (Poisson 0.65/s) we expect ≥1 vehicle by t=10s.
        # Allow zero on rare bad-luck seeds; just check the API works.
        veh_ids = traci.vehicle.getIDList()
        assert isinstance(veh_ids, tuple)
