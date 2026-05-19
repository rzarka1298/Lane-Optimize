"""Backend's owns-the-sim service.

One `SimulatorService` instance per backend process. TraCI's module-level
connection state forbids two parallel sims; the service enforces this.

Architecture:

- `start()` constructs the env + agent, kicks off an asyncio task that
  drives the sim loop at ~10 Hz, returns a `run_id`.
- The loop pushes `SimTick` payloads to every subscribed asyncio.Queue.
- WebSocket handlers register themselves via `subscribe()` and read from
  their queue; on disconnect they `unsubscribe()`.
- `stop()` cancels the task, closes the env, cleans state.

Thread/process model: all sync TraCI calls run on the event loop thread
(no executor wrap). LaneIQEnv.step is fast (~3-5 ms with policy.act);
not enough to block the event loop noticeably. If multi-client load ever
matters, we can move env.step into a thread pool — but for the recruiter
demo with one viewer, this is fine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import traci

from backend.app.schemas import Decision, SimTick, StreamFrame, Vehicle
from laneiq.agents.base import Policy
from laneiq.env.actions import Action
from laneiq.env.highway_env import LaneIQEnv

_log = logging.getLogger(__name__)

DensityT = Literal["low", "medium", "high", "gauntlet", "rush"]
_EGO_ID = "ego_0"  # mirrors highway_env constant
_TICK_RATE_HZ = 10.0
_TICK_INTERVAL_S = 1.0 / _TICK_RATE_HZ


@dataclass
class _ActiveRun:
    """Bookkeeping for the currently-running sim. `env` and `agent` are
    None for replay-mode runs (which stream pre-recorded ticks)."""

    run_id: str
    agent_name: str
    density: DensityT
    seed: int | None
    env: LaneIQEnv | None
    agent: Policy | None = None
    is_replay: bool = False
    task: asyncio.Task | None = None
    subscribers: set[asyncio.Queue] = field(default_factory=set)
    step: int = 0
    sim_time_s: float = 0.0


class SimulatorService:
    """Per-process singleton — start/stop a sim, stream ticks."""

    def __init__(self) -> None:
        self._active: _ActiveRun | None = None
        self._lock = asyncio.Lock()

    @property
    def is_running(self) -> bool:
        return self._active is not None and self._active.task is not None and not self._active.task.done()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    async def start(
        self,
        *,
        agent: Policy,
        agent_name: str,
        density: DensityT,
        seed: int | None,
        max_steps: int,
    ) -> str:
        """Start a new sim. Raises if one is already running."""
        async with self._lock:
            if self.is_running:
                raise RuntimeError(
                    f"a sim is already running (run_id={self._active.run_id})",
                )

            env = LaneIQEnv(
                density=density,
                max_steps=max_steps,
                warmup_steps=50,
                seed=seed,
                # The rush scenario auto-loops in the live demo: when the
                # ego reaches the end of the road, it's re-inserted at
                # x=0 so the demo runs minutes, not ~25 s. Eval doesn't
                # set this (single-lap consistency with other agents).
                loop=(density == "rush"),
            )
            run_id = uuid.uuid4().hex[:8]
            run = _ActiveRun(
                run_id=run_id, agent=agent, agent_name=agent_name,
                density=density, seed=seed, env=env,
            )
            run.task = asyncio.create_task(self._run_loop(run))
            self._active = run
            _log.info("sim started run_id=%s agent=%s density=%s seed=%s",
                      run_id, agent_name, density, seed)
            return run_id

    async def start_replay(self, *, replay_path: Path, name: str) -> str:
        """Start a replay-mode run that streams pre-recorded ticks at 10 Hz.

        Reads the JSON produced by `scripts/record_replay.py`. No SUMO is
        started — the simulator just yields the ticks back over WS at the
        same cadence the live loop would.
        """
        async with self._lock:
            if self.is_running:
                raise RuntimeError(
                    f"a sim is already running (run_id={self._active.run_id})",
                )
            payload = json.loads(replay_path.read_text())
            ticks_raw = payload["ticks"]
            run_id = uuid.uuid4().hex[:8]
            run = _ActiveRun(
                run_id=run_id, agent_name=name,
                density=payload.get("density", "medium"),
                seed=payload.get("seed"),
                env=None, agent=None, is_replay=True,
            )
            run.task = asyncio.create_task(self._replay_loop(run, ticks_raw))
            self._active = run
            _log.info("replay started run_id=%s name=%s n_ticks=%d",
                      run_id, name, len(ticks_raw))
            return run_id

    async def stop(self) -> None:
        """Stop the running sim. No-op if nothing's running."""
        async with self._lock:
            if self._active is None:
                return
            run = self._active
            if run.task is not None and not run.task.done():
                run.task.cancel()
                with suppress(asyncio.CancelledError):
                    await run.task
            if run.env is not None:
                try:
                    run.env.close()
                except Exception:
                    _log.exception("error closing env on stop")
            # Notify subscribers + clear them.
            for q in list(run.subscribers):
                await q.put(StreamFrame(type="stopped", message="sim stopped"))
            run.subscribers.clear()
            self._active = None
            _log.info("sim stopped run_id=%s", run.run_id)

    # ------------------------------------------------------------------
    # Status (for GET /sim/status)
    # ------------------------------------------------------------------

    def status(self) -> dict:
        if self._active is None:
            return {"running": False, "step": 0, "sim_time_s": 0.0}
        return {
            "running": self.is_running,
            "run_id": self._active.run_id,
            "agent": self._active.agent_name,
            "density": self._active.density,
            "seed": self._active.seed,
            "step": self._active.step,
            "sim_time_s": self._active.sim_time_s,
            "is_replay": self._active.is_replay,
        }

    # ------------------------------------------------------------------
    # Subscription (WebSocket handlers)
    # ------------------------------------------------------------------

    def subscribe(self) -> asyncio.Queue:
        """Register a subscriber queue. Caller must `unsubscribe(queue)` on disconnect."""
        if self._active is None:
            raise RuntimeError("no sim running")
        q: asyncio.Queue = asyncio.Queue(maxsize=50)  # backpressure
        self._active.subscribers.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        if self._active is None:
            return
        self._active.subscribers.discard(q)

    # ------------------------------------------------------------------
    # Loop body
    # ------------------------------------------------------------------

    async def _replay_loop(self, run: _ActiveRun, ticks: list[dict]) -> None:
        """Stream pre-recorded ticks at the live tick rate."""
        try:
            for q in list(run.subscribers):
                await q.put(StreamFrame(
                    type="started",
                    message=f"replay={run.agent_name}",
                ))
            for tick_dict in ticks:
                t_loop_start = time.monotonic()
                tick = SimTick(**tick_dict)
                run.step = tick.step
                run.sim_time_s = tick.t
                self._broadcast(run, StreamFrame(type="tick", tick=tick))
                elapsed = time.monotonic() - t_loop_start
                if elapsed < _TICK_INTERVAL_S:
                    await asyncio.sleep(_TICK_INTERVAL_S - elapsed)
            for q in list(run.subscribers):
                await q.put(StreamFrame(type="stopped", message="replay ended"))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.exception("replay loop crashed")
            for q in list(run.subscribers):
                await q.put(StreamFrame(type="error", message=str(exc)))

    async def _run_loop(self, run: _ActiveRun) -> None:
        """The hot loop. Stepped once per ~100 ms."""
        assert run.env is not None and run.agent is not None
        try:
            obs, _info = run.env.reset(seed=run.seed)

            # Push a 'started' frame to any subscribers connected at start
            # (none usually — clients subscribe AFTER receiving the start
            # response — but it doesn't hurt).
            for q in list(run.subscribers):
                await q.put(StreamFrame(
                    type="started",
                    message=f"agent={run.agent_name} density={run.density}",
                ))

            terminated = truncated = False
            while not (terminated or truncated):
                t_loop_start = time.monotonic()

                mask = run.env.action_masks()
                action = run.agent.act(obs, action_mask=mask)
                obs, _reward, terminated, truncated, info = run.env.step(action)
                run.step += 1
                run.sim_time_s += 0.1  # step_length

                # Build + broadcast the tick. If broadcast lags (no subscribers),
                # we skip it rather than block.
                tick = self._build_tick(run, action, mask, obs, info,
                                        terminated, truncated)
                frame = StreamFrame(type="tick", tick=tick)
                self._broadcast(run, frame)

                # Sleep up to the tick interval. Drop the wait if the
                # step itself took longer than the interval.
                elapsed = time.monotonic() - t_loop_start
                if elapsed < _TICK_INTERVAL_S:
                    await asyncio.sleep(_TICK_INTERVAL_S - elapsed)

            # Episode ended — push one more frame so the client knows.
            for q in list(run.subscribers):
                await q.put(StreamFrame(
                    type="stopped",
                    message="terminated" if terminated else "truncated",
                ))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            _log.exception("sim loop crashed")
            for q in list(run.subscribers):
                await q.put(StreamFrame(type="error", message=str(exc)))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_tick(
        self,
        run: _ActiveRun,
        action: int,
        mask,
        obs,
        info: dict,
        terminated: bool,
        truncated: bool,
    ) -> SimTick:
        return build_tick(
            step=run.step,
            sim_time_s=run.sim_time_s,
            action=action,
            mask=mask,
            obs=obs,
            info=info,
            terminated=terminated,
            truncated=truncated,
        )

    @staticmethod
    def _explain(action: int, obs, mask) -> str:
        return explain(action)

    def _broadcast(self, run: _ActiveRun, frame: StreamFrame) -> None:
        """Push a frame to every subscriber; drop oldest if queue is full."""
        for q in list(run.subscribers):
            try:
                q.put_nowait(frame)
            except asyncio.QueueFull:
                # Backpressure: client is slow. Drop the oldest frame
                # rather than block the sim loop.
                try:
                    q.get_nowait()
                    q.put_nowait(frame)
                except (asyncio.QueueEmpty, asyncio.QueueFull):
                    pass  # give up; queue will catch up


# --------------------------------------------------------------------
# Module-level helpers (reused by scripts/record_replay.py so recorded
# JSON is byte-identical to live-stream payloads).
# --------------------------------------------------------------------


def build_tick(
    *,
    step: int,
    sim_time_s: float,
    action: int,
    mask,
    obs,
    info: dict,
    terminated: bool,
    truncated: bool,
) -> SimTick:
    """Snapshot the current TraCI vehicle list into a SimTick wire payload.

    Caller must hold an active TraCI connection (e.g., inside
    LaneIQEnv's lifetime). `mask` is unused today but kept in the
    signature for future use (e.g., embedding the mask in the tick).
    """
    del mask  # reserved for future use
    vehicles: list[Vehicle] = []
    try:
        veh_ids = traci.vehicle.getIDList()
    except Exception:
        veh_ids = ()

    for vid in veh_ids:
        try:
            x = float(traci.vehicle.getLanePosition(vid))
            lane = int(traci.vehicle.getLaneIndex(vid))
            lat = float(traci.vehicle.getLateralLanePosition(vid))
            # Lane y = lane center; one lane is ~3.2 m wide.
            y = lane * 3.2 + lat
            speed = float(traci.vehicle.getSpeed(vid))
            vtype_raw = traci.vehicle.getTypeID(vid)
            vtype = vtype_raw if vtype_raw in ("aggressive", "normal", "slow", "ego") else "normal"
            vehicles.append(Vehicle(
                id=vid, x=x, y=y, speed=speed, lane=max(0, min(2, lane)),
                is_ego=(vid == _EGO_ID), vtype=vtype,  # type: ignore[arg-type]
            ))
        except Exception:
            continue

    action_name = ("stay", "change_left", "change_right")[int(action)]
    return SimTick(
        t=sim_time_s,
        step=step,
        ego_id=_EGO_ID,
        vehicles=vehicles,
        decision=Decision(action=action_name, explanation=explain(action)),  # type: ignore[arg-type]
        terminated=terminated,
        truncated=truncated,
        info={k: float(v) for k, v in info.get("reward_components", {}).items()},
        lap_count=int(info.get("lap_count", 0)),
    )


def explain(action: int) -> str:
    """One-sentence English description of an action — same template for
    live and replay payloads."""
    a = Action(int(action))
    if a == Action.STAY:
        return "Holding lane — current lane is acceptable."
    if a == Action.CHANGE_LEFT:
        return "Changing left — opportunity in adjacent lane."
    return "Changing right — slower traffic ahead or rebalancing."


# Module-level singleton.
SIM_SERVICE: SimulatorService = SimulatorService()
