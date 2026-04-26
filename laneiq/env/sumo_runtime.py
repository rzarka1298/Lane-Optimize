"""Process-level lifecycle for the SUMO simulator.

LaneIQ talks to SUMO via TraCI (libsumo intentionally not used — see
`Project-Documentation/setup.md`). This module is the single place that
owns launching and tearing down the SUMO process and the TraCI connection.

Why a dedicated module (not inlined into `highway_env.py`):

1. Tests and ad-hoc scripts need the same lifecycle without instantiating
   a Gymnasium env (profiling, smoke checks, debugging).
2. Centralizes the long flag list (collision policy, step length, seed,
   route override) in one auditable place.
3. Single seam where we'd swap to libsumo if Week 3 profiling justifies
   the from-source build.

The Gym env (``laneiq.env.highway_env.LaneIQEnv``) does NOT use the context
manager here — it owns its TraCI connection across many ``reset()`` calls
and manages start/close itself via the same ``build_argv()`` builder.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from pathlib import Path

import sumolib
import traci

# Resolve scenario paths relative to the repo, regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SCENARIO_DIR = _REPO_ROOT / "sumo_scenarios" / "highway_3lane"
DEFAULT_SUMOCFG = DEFAULT_SCENARIO_DIR / "highway.sumocfg"

# Flags we want to LOCK regardless of what's in the sumocfg. CLI flags
# override file values; these stay identical across all training/eval runs
# so behavior is reproducible. They mirror the sumocfg today, but explicit
# beats implicit when training is involved.
_LOCKED_FLAGS: list[str] = [
    "--no-step-log", "true",
    "--time-to-teleport", "-1",        # never silently teleport stuck cars
    "--collision.action", "warn",       # let TraCI see collisions; don't auto-remove
    "--collision.mingap-factor", "0",   # collisions register at true zero overlap
    "--step-length", "0.1",             # 10 Hz sim — matches WS stream rate
]


def sumo_binary(gui: bool = False) -> str:
    """Path to the bundled SUMO binary (handles the ``eclipse-sumo`` wheel).

    Resolves via ``sumolib.checkBinary()``, which knows how to find the
    binaries inside the wheel even though they live under ``site-packages/sumo/bin/``.
    """
    return sumolib.checkBinary("sumo-gui" if gui else "sumo")


def build_argv(
    sumocfg: str | os.PathLike[str] = DEFAULT_SUMOCFG,
    *,
    gui: bool = False,
    seed: int | None = None,
    route_file: str | os.PathLike[str] | None = None,
    end_time: float | None = None,
    extra_flags: list[str] | None = None,
) -> list[str]:
    """Build the argv for ``traci.start(...)``.

    Args:
        sumocfg: Path to the ``.sumocfg`` config. Defaults to the 3-lane highway.
        gui: If True, launch ``sumo-gui`` instead of headless ``sumo``.
        seed: Optional fixed seed (for reproducibility).
        route_file: Optional override of the routes file in the sumocfg.
        end_time: Optional override of the simulation end time (seconds).
        extra_flags: Tail-appended raw flags (escape hatch for one-off needs).

    Returns:
        argv list ready to pass to ``traci.start(argv, ...)``.
    """
    argv: list[str] = [
        sumo_binary(gui),
        "-c", str(Path(sumocfg).resolve()),
        *_LOCKED_FLAGS,
    ]
    if seed is not None:
        argv += ["--seed", str(seed)]
    if route_file is not None:
        argv += ["--route-files", str(Path(route_file).resolve())]
    if end_time is not None:
        argv += ["--end", str(end_time)]
    if extra_flags:
        argv += list(extra_flags)
    return argv


@contextmanager
def sumo_session(
    sumocfg: str | os.PathLike[str] = DEFAULT_SUMOCFG,
    *,
    gui: bool = False,
    seed: int | None = None,
    route_file: str | os.PathLike[str] | None = None,
    end_time: float | None = None,
    label: str = "laneiq",
) -> Iterator[None]:
    """Context manager that starts SUMO, yields, and guarantees close.

    Use in tests and ad-hoc scripts. Do NOT wrap a ``LaneIQEnv`` in this —
    the env owns its own connection.

    Example::

        from laneiq.env.sumo_runtime import sumo_session
        import traci

        with sumo_session(seed=42, end_time=60):
            for _ in range(100):
                traci.simulationStep()
            assert abs(traci.simulation.getTime() - 10.0) < 0.01

    Notes:
        - ``label`` is the TraCI connection name. Default ``"laneiq"`` is fine
          unless you need multiple simultaneous connections (multi-agent
          eval doesn't — it's still one SUMO with multiple egos).
        - The ``finally`` block swallows close errors. TraCI sometimes raises
          if the SUMO process already died (e.g., a sumocfg validation
          error) — the context manager shouldn't mask that with a confusing
          double-exception.
    """
    argv = build_argv(
        sumocfg,
        gui=gui,
        seed=seed,
        route_file=route_file,
        end_time=end_time,
    )
    traci.start(argv, label=label)
    try:
        yield
    finally:
        # SUMO process may have died before we got here (e.g., bad sumocfg);
        # don't mask the original exception with a misleading close error.
        with suppress(traci.exceptions.FatalTraCIError):
            traci.close(False)
