import { useEffect, useState } from "react";
import { api, ApiError } from "./api/client";
import { useSimStream } from "./api/useSimStream";
import type { Density, ReplayListEntry, RunListEntry } from "./api/types";
import { HighwayCanvas } from "./components/HighwayCanvas";
import { DecisionTicker } from "./components/DecisionTicker";
import { MetricsPanel } from "./components/MetricsPanel";

// Smoke-test app — proves the full plumbing (REST start + WS stream) works.
// Layout components arrive in the next session; this is deliberately ugly.
export function App() {
  const [runs, setRuns] = useState<RunListEntry[]>([]);
  const [replays, setReplays] = useState<ReplayListEntry[]>([]);
  const [agent, setAgent] = useState<string>("");
  const [replay, setReplay] = useState<string>("");
  const [density, setDensity] = useState<Density>("medium");
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stream = useSimStream(streaming);

  async function loadRuns(preserve?: string) {
    try {
      const rs = await api.getRuns();
      setRuns(rs);
      const stillThere = preserve && rs.some((r) => r.name === preserve);
      if (stillThere) {
        setAgent(preserve);
      } else if (rs.length > 0 && rs[0]) {
        setAgent(rs[0].name);
      } else {
        setAgent("");
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(`Failed to load /runs: ${msg}`);
    }
  }

  async function loadReplays() {
    try {
      const rs = await api.getReplays();
      setReplays(rs);
      if (rs.length > 0 && rs[0] && !replay) setReplay(rs[0].name);
    } catch {
      // Replays endpoint may 404 on older backends; non-fatal.
    }
  }

  useEffect(() => {
    void loadRuns();
    void loadReplays();
  }, []);

  async function onStart() {
    setError(null);
    try {
      await api.startSim({ agent, density, seed: 42 });
      setStreaming(true);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(`start agent='${agent}' density='${density}' → ${e.status} ${e.message}`);
        // 404 = agent not found. Refetch /runs so the dropdown self-heals
        // if the checkpoint folder changed since page load.
        if (e.status === 404) void loadRuns(agent);
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      }
    }
  }

  async function onStartReplay() {
    setError(null);
    try {
      await api.startReplay({ replay });
      setStreaming(true);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(`replay '${replay}' → ${e.status} ${e.message}`);
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      }
    }
  }

  async function onStop() {
    setStreaming(false);
    try {
      await api.stopSim();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }

  const tick = stream.lastTick;

  return (
    <main className="mx-auto max-w-6xl space-y-4 p-6 font-mono text-sm">
      <header className="flex items-center justify-between">
        <h1 className="text-xl font-semibold">LaneIQ — live demo</h1>
        <span
          className={`rounded px-2 py-1 text-xs ${
            stream.status === "open"
              ? "bg-emerald-600"
              : stream.status === "connecting"
                ? "bg-amber-600"
                : stream.status === "error"
                  ? "bg-red-600"
                  : "bg-slate-700"
          }`}
        >
          ws: {stream.status}
        </span>
      </header>

      <section className="flex flex-wrap items-center gap-3 rounded border border-slate-800 p-3">
        <label className="flex items-center gap-2">
          agent
          <select
            className="rounded bg-slate-800 px-2 py-1"
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
            disabled={streaming}
          >
            {runs.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} ({r.algorithm})
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void loadRuns(agent)}
            disabled={streaming}
            title="Re-fetch /runs"
            className="rounded bg-slate-700 px-2 py-1 text-xs hover:bg-slate-600 disabled:opacity-40"
          >
            ↻
          </button>
        </label>
        <label className="flex items-center gap-2">
          density
          <select
            className="rounded bg-slate-800 px-2 py-1"
            value={density}
            onChange={(e) => setDensity(e.target.value as Density)}
            disabled={streaming}
          >
            <option value="low">low</option>
            <option value="medium">medium</option>
            <option value="high">high</option>
          </select>
        </label>
        {streaming ? (
          <button
            onClick={onStop}
            className="rounded bg-red-600 px-3 py-1 hover:bg-red-500"
          >
            Stop
          </button>
        ) : (
          <button
            onClick={onStart}
            disabled={!agent}
            className="rounded bg-emerald-600 px-3 py-1 hover:bg-emerald-500 disabled:opacity-40"
          >
            Start live
          </button>
        )}
      </section>

      {replays.length > 0 && (
        <section className="flex flex-wrap items-center gap-3 rounded border border-slate-800 p-3">
          <span className="text-xs uppercase tracking-wider text-slate-500">
            replay
          </span>
          <select
            className="rounded bg-slate-800 px-2 py-1"
            value={replay}
            onChange={(e) => setReplay(e.target.value)}
            disabled={streaming}
          >
            {replays.map((r) => (
              <option key={r.name} value={r.name}>
                {r.name} — {r.agent} · {r.density} · {r.n_ticks} ticks
              </option>
            ))}
          </select>
          <button
            onClick={onStartReplay}
            disabled={streaming || !replay}
            className="rounded bg-cyan-600 px-3 py-1 hover:bg-cyan-500 disabled:opacity-40"
          >
            Play replay
          </button>
        </section>
      )}

      {error && (
        <div className="rounded border border-red-600 bg-red-950 p-2 text-red-200">
          {error}
        </div>
      )}
      {stream.lastMessage && (
        <div className="rounded border border-slate-700 bg-slate-900 p-2 text-slate-300">
          control: {stream.lastMessage}
        </div>
      )}

      <DecisionTicker
        decision={tick?.decision ?? null}
        step={tick?.step ?? null}
        simTimeS={tick?.t ?? null}
      />

      <HighwayCanvas
        prevTick={stream.prevTick}
        lastTick={stream.lastTick}
        lastTickArrivedAt={stream.lastTickArrivedAt}
      />

      <MetricsPanel history={stream.history} lastTick={stream.lastTick} />

      <section className="rounded border border-slate-800 p-3 text-xs">
        <h2 className="mb-2 text-slate-400">diagnostics</h2>
        {tick ? (
          <dl className="grid grid-cols-[10rem_1fr] gap-x-4 gap-y-1">
            <dt className="text-slate-500">vehicles in scene</dt>
            <dd>{tick.vehicles.length}</dd>
            <dt className="text-slate-500">terminated</dt>
            <dd>{String(tick.terminated)}</dd>
            <dt className="text-slate-500">truncated</dt>
            <dd>{String(tick.truncated)}</dd>
            <dt className="text-slate-500">history buffer</dt>
            <dd>{stream.history.length} ticks</dd>
          </dl>
        ) : (
          <p className="text-slate-500">no ticks yet</p>
        )}
      </section>
    </main>
  );
}
