import { useEffect, useRef, useState } from "react";
import { api, ApiError } from "./api/client";
import { useSimStream } from "./api/useSimStream";
import type { Density, ReplayListEntry, RunListEntry } from "./api/types";
import { HighwayCanvas } from "./components/HighwayCanvas";
import { DecisionTicker } from "./components/DecisionTicker";
import { MetricsPanel } from "./components/MetricsPanel";

// Recruiter-facing labels for the canonical agents. Kept in the frontend
// (not in the backend's /runs response) because they're a presentation
// concern: which trade-off this checkpoint embodies.
// Numbers: 20-seed evals in the current env (fixed side-lane obs, 1.5 s
// lane-change cooldown, safety-checked lane changes, ego speed pinned). Checkpoints marked "pre-fix" were trained on
// mirrored side-lane observations (LESSONS #18) and kept for comparison.
const AGENT_LABELS: Record<string, string> = {
  ppo_v12_safety_cd_150k:
    "PPO canonical — safety reward + lane-change cooldown",
  ppo_v10_gauntlet_cd_300k:
    "PPO gauntlet — escapes the slow lane (use with Gauntlet density)",
  mappo_v3_coop_cd_300k:
    "MAPPO cooperative — shared-speed bonus 0.1 + cooldown",
  dqn_v4_mixed_200k_obsfix: "DQN generalist — mixed-density training",
  ppo_v2_safety_heavy_150k: "PPO conservative (pre-fix)",
  ppo_v1_mixed_500k: "PPO aggressive (pre-fix)",
  ppo_v5_gauntlet_300k: "PPO gauntlet (pre-fix)",
  dqn_v3_mixed_200k: "DQN generalist (pre-fix)",
  dqn_v2_100k_huber: "DQN low-density specialist (pre-fix)",
  mappo_v1_300k: "MAPPO independent rewards (pre-fix)",
  mappo_v2_coop_300k: "MAPPO cooperative (pre-fix)",
  overtaker: "Aggressive overtaker — rule-based weaver (rush)",
};

// Builtin rule-based agents. These live in the backend's baseline registry
// (laneiq/agents/baselines), NOT in checkpoints/, so they don't appear in
// the /runs response — we list them here so they show in the dropdown and
// the quick-pick availability check passes.
const BUILTIN_AGENTS: string[] = ["overtaker"];

interface QuickPick {
  label: string;
  agent: string;
  tooltip: string;
  density?: Density; // if set, the pick also switches density
  maxSteps?: number; // override default max_steps (e.g. for the looping rush)
}

const QUICK_PICKS: QuickPick[] = [
  {
    label: "Conservative",
    agent: "ppo_v12_safety_cd_150k",
    density: "medium",
    tooltip:
      "Safety-tuned PPO at medium density. Picks its lane changes (~3 per " +
      "minute); SUMO's lane-change assist refuses any unsafe one. 0% " +
      "collisions, fastest PPO finish on high density (30.6 s / km).",
  },
  {
    label: "Aggressive",
    agent: "overtaker",
    density: "rush",
    // 1800 steps ≈ 3 min sim. The rush road is a 10 km straight; at
    // ~45-50 m/s the ego covers ~9 km in 1800 steps, so the episode
    // truncates cleanly before reaching the end (no exit, no wrap).
    maxSteps: 1800,
    tooltip:
      "Rule-based overtaker on the rush scenario: 50 m/s lane limit, the " +
      "ego is the fastest car on a 10 km straight and weaves around slower " +
      "traffic — passes on either side, climbs back once the lane is open. " +
      "~42 m/s average vs 33 m/s traffic. Runs ~3 min; tick the loop box " +
      "for a continuous demo.",
  },
  {
    label: "Multi-agent",
    agent: "mappo_v3_coop_cd_300k",
    density: "medium",
    tooltip:
      "Parameter-sharing PPO, 4 egos, trained with a cooperative " +
      "shared-speed bonus (0.1): each ego is rewarded for the team's mean " +
      "speed, so yielding pays off. 0% per-ego collisions at every " +
      "density over a 20-seed eval.",
  },
  {
    label: "Gauntlet",
    agent: "ppo_v10_gauntlet_cd_300k",
    density: "gauntlet",
    tooltip:
      "Trained on the gauntlet scenario — starts stuck in the slow lane and " +
      "weaves out. 32.9 m/s mean, 0% collisions over a 20-seed eval.",
  },
];

function randomSeed(): number {
  return Math.floor(Math.random() * 1_000_000);
}

export function App() {
  const [runs, setRuns] = useState<RunListEntry[]>([]);
  const [replays, setReplays] = useState<ReplayListEntry[]>([]);
  const [agent, setAgent] = useState<string>("");
  const [replay, setReplay] = useState<string>("");
  const [density, setDensity] = useState<Density>("medium");
  const [seed, setSeed] = useState<number>(() => randomSeed());
  const [loop, setLoop] = useState<boolean>(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const stream = useSimStream(streaming);

  // Track the last episode's terminal step so the auto-restart effect fires
  // exactly once per episode-end (lastTick stays "terminated=true" until we
  // tear down).
  const lastEndedStep = useRef<number | null>(null);
  // Guards against double-firing the loop restart. Lives in a ref so it
  // survives re-renders triggered by setStreaming(false) mid-restart.
  const restartingRef = useRef(false);
  // Latest values for the loop callback to read without re-binding the
  // effect. Prevents stale-closure bugs when user changes agent/density
  // mid-loop.
  const liveValuesRef = useRef({ agent, density, loop });
  liveValuesRef.current = { agent, density, loop };

  // max_steps the next Start should use. Quick-picks override this
  // (the rush pick wants 3000 = ~5 min sim so the loop has room).
  const [maxSteps, setMaxSteps] = useState<number>(1000);

  async function loadRuns(preserve?: string) {
    try {
      const rs = await api.getRuns();
      setRuns(rs);
      const stillThere = preserve && rs.some((r) => r.name === preserve);
      if (stillThere) {
        setAgent(preserve);
      } else if (rs.length > 0 && rs[0]) {
        // Default to the canonical PPO if it's available, else first entry.
        const canonical = rs.find(
          (r) => r.name === "ppo_v12_safety_cd_150k",
        );
        setAgent(canonical?.name ?? rs[0].name);
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

  async function startWith(seedToUse: number) {
    setError(null);
    try {
      await api.startSim({
        agent, density, seed: seedToUse, max_steps: maxSteps,
      });
      setStreaming(true);
    } catch (e) {
      if (e instanceof ApiError) {
        setError(
          `start agent='${agent}' density='${density}' seed=${seedToUse} → ${e.status} ${e.message}`,
        );
        if (e.status === 404) void loadRuns(agent);
      } else {
        const msg = e instanceof Error ? e.message : String(e);
        setError(msg);
      }
    }
  }

  async function onStart() {
    const fresh = randomSeed();
    setSeed(fresh);
    lastEndedStep.current = null;
    await startWith(fresh);
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
    lastEndedStep.current = null;
    try {
      await api.stopSim();
    } catch (e) {
      const msg = e instanceof Error ? e.message : String(e);
      setError(msg);
    }
  }

  // Fire-and-forget loop restart. NOT wrapped in useEffect cleanup —
  // re-renders during the restart (caused by setStreaming(false) and the
  // intermediate setSeed) would otherwise fire the effect's cleanup,
  // killing the in-flight async before startSim runs. The ref guard
  // prevents double-fire if multiple terminal ticks arrive back-to-back.
  async function performLoopRestart(): Promise<void> {
    if (restartingRef.current) return;
    restartingRef.current = true;
    try {
      // Pause so the viewer registers the end state before the next
      // episode starts.
      await new Promise((r) => setTimeout(r, 600));
      // User may have toggled loop off during the pause.
      if (!liveValuesRef.current.loop) return;

      setStreaming(false);
      try {
        await api.stopSim();
      } catch {
        /* idempotent */
      }
      // Brief gap for the backend to fully tear down + the WS hook to
      // reset its state (history, lastTick) so the next episode starts
      // with a clean slate.
      await new Promise((r) => setTimeout(r, 200));
      if (!liveValuesRef.current.loop) return;

      const fresh = randomSeed();
      setSeed(fresh);
      await startWith(fresh);
    } finally {
      restartingRef.current = false;
    }
  }

  // Detect episode-end and trigger the loop restart. The effect itself
  // is intentionally tiny — all the lifecycle work happens in
  // performLoopRestart, which runs outside the effect.
  useEffect(() => {
    if (!loop || !streaming) return;
    const tick = stream.lastTick;
    if (!tick) return;
    if (!(tick.terminated || tick.truncated)) return;
    if (lastEndedStep.current === tick.step) return;
    lastEndedStep.current = tick.step;
    void performLoopRestart();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [stream.lastTick, loop, streaming]);

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

      <section className="flex flex-wrap items-center gap-2 rounded border border-slate-800 p-3">
        <span className="text-xs uppercase tracking-wider text-slate-500">
          quick pick
        </span>
        {QUICK_PICKS.map((qp) => {
          const available =
            BUILTIN_AGENTS.includes(qp.agent) ||
            runs.some((r) => r.name === qp.agent);
          // Active iff BOTH agent and density match. Without the density
          // check, Conservative and Aggressive (both ppo_v2) would both
          // light up whenever ppo_v2 is the selected agent.
          const isActive = agent === qp.agent && density === qp.density;
          return (
            <button
              key={qp.agent}
              onClick={() => {
                setAgent(qp.agent);
                if (qp.density) setDensity(qp.density);
                setMaxSteps(qp.maxSteps ?? 1000);
              }}
              disabled={streaming || !available}
              title={qp.tooltip}
              className={`rounded px-3 py-1 text-xs transition ${
                isActive
                  ? "bg-yellow-500 text-slate-900"
                  : "bg-slate-700 hover:bg-slate-600"
              } disabled:opacity-40`}
            >
              {qp.label}
            </button>
          );
        })}
      </section>

      <section className="flex flex-wrap items-center gap-3 rounded border border-slate-800 p-3">
        <label className="flex items-center gap-2">
          agent
          <select
            className="rounded bg-slate-800 px-2 py-1"
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
            disabled={streaming}
          >
            {BUILTIN_AGENTS.map((name) => (
              <option key={name} value={name}>
                {AGENT_LABELS[name] ?? name}
              </option>
            ))}
            {runs.map((r) => (
              <option key={r.name} value={r.name}>
                {AGENT_LABELS[r.name] ?? `${r.name} (${r.algorithm})`}
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
            <option value="gauntlet">gauntlet</option>
            <option value="rush">rush</option>
          </select>
        </label>
        <label className="flex items-center gap-2">
          seed
          <input
            type="number"
            value={seed}
            onChange={(e) => setSeed(Number(e.target.value) || 0)}
            disabled={streaming}
            className="w-24 rounded bg-slate-800 px-2 py-1 font-mono"
          />
          <button
            type="button"
            onClick={() => setSeed(randomSeed())}
            disabled={streaming}
            title="Re-roll random seed"
            className="rounded bg-slate-700 px-2 py-1 text-xs hover:bg-slate-600 disabled:opacity-40"
          >
            🎲
          </button>
        </label>
        <label
          className="flex items-center gap-2 text-xs"
          title="When an episode ends, automatically start the next one with a fresh seed"
        >
          <input
            type="checkbox"
            checked={loop}
            onChange={(e) => setLoop(e.target.checked)}
            className="h-3 w-3"
          />
          loop episodes
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

      <HighwayCanvas buffer={stream.buffer} hasTick={stream.lastTick !== null} />

      <MetricsPanel history={stream.history} lastTick={stream.lastTick} />

      <details className="rounded border border-slate-800 bg-slate-900 p-3 text-xs">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          How it learned — training curves for the canonical agents
        </summary>
        <p className="mt-2 text-slate-500">
          Each panel shows mean episode return (or value-function explained
          variance, for MAPPO) over training steps. The dashboard above
          plays back checkpoints from the end of these runs.
        </p>
        <img
          src="/training_curves.png"
          alt="LaneIQ training curves for DQN, PPO v1, PPO v2, and MAPPO"
          className="mt-2 w-full rounded border border-slate-800"
        />
      </details>

      <details className="rounded border border-slate-800 bg-slate-900 p-3 text-xs">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          How it uses lanes — per-replay heatmaps with ego trajectory
        </summary>
        <p className="mt-2 text-slate-500">
          Background = mean vehicles per lane per time bin (darker = more
          traffic). Gold line = ego&apos;s lane over time. Dots mark lane
          changes. The red X marks where the episode ended (collision or
          end-of-road). The conservative ppo_v2 picks the open lane and
          holds it; ppo_v1&apos;s collision panel shows the lane change
          that ended the episode.
        </p>
        <img
          src="/lane_efficiency.png"
          alt="Per-replay lane occupancy heatmaps with ego trajectory overlay"
          className="mt-2 w-full rounded border border-slate-800"
        />
      </details>

      <details className="rounded border border-slate-800 bg-slate-900 p-3 text-xs">
        <summary className="cursor-pointer text-slate-400 hover:text-slate-200">
          Overtaking — trained vs untrained on the gauntlet scenario
        </summary>
        <p className="mt-2 text-slate-500">
          The gauntlet floods lane 0 with slow traffic and starts the ego
          there. Top: ppo_v2 (not trained for it) gives up — stays pinned
          to lane 0 at 24.8 m/s, zero lane changes. Bottom: ppo_v5
          (trained on the gauntlet) escapes within the first second and
          cruises at 32.7 m/s. Try it live with the &quot;Gauntlet&quot;
          quick-pick.
        </p>
        <img
          src="/gauntlet_comparison.png"
          alt="Gauntlet comparison — ppo_v2 stuck in lane 0 vs ppo_v5 escaping"
          className="mt-2 w-full rounded border border-slate-800"
        />
      </details>

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
