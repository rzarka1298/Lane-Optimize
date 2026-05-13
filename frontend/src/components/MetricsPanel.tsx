import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { SimTick } from "../api/types";

// 6-component reward, mirrored from laneiq/env/rewards.py:components dict.
// Order here drives the legend + line render order.
const COMPONENTS = [
  { key: "speed", color: "#10b981" }, // emerald — positive cruising reward
  { key: "alive", color: "#34d399" }, // light emerald — small dense bonus
  { key: "lane_cost", color: "#94a3b8" }, // slate — small lane-change penalty
  { key: "jerk", color: "#f97316" }, // orange — smoothness penalty
  { key: "unsafe_gap", color: "#fbbf24" }, // amber — close-following warning
  { key: "collision", color: "#ef4444" }, // red — terminal penalty
] as const;

const RECENT_WINDOW = 60; // ticks of recent reward chart (= 6 s at 10 Hz)

const LANE_LABEL: Record<number, string> = {
  0: "right",
  1: "middle",
  2: "left",
};

interface Props {
  history: SimTick[];
  lastTick: SimTick | null;
}

export function MetricsPanel({ history, lastTick }: Props) {
  // Cumulative return at each step. useMemo because history changes every
  // tick and we don't want to re-cumsum 600 entries on unrelated re-renders.
  const cumulative = useMemo(() => {
    let running = 0;
    return history.map((t) => {
      const stepReward = Object.values(t.info).reduce((a, b) => a + b, 0);
      running += stepReward;
      return { step: t.step, return: round2(running) };
    });
  }, [history]);

  const recent = useMemo(() => {
    return history.slice(-RECENT_WINDOW).map((t) => ({
      step: t.step,
      speed: round2(t.info["speed"] ?? 0),
      alive: round2(t.info["alive"] ?? 0),
      lane_cost: round2(t.info["lane_cost"] ?? 0),
      jerk: round2(t.info["jerk"] ?? 0),
      unsafe_gap: round2(t.info["unsafe_gap"] ?? 0),
      collision: round2(t.info["collision"] ?? 0),
    }));
  }, [history]);

  const ego = lastTick?.vehicles.find((v) => v.is_ego);
  const totalReturn = cumulative.length
    ? cumulative[cumulative.length - 1]?.["return"]
    : null;

  return (
    <section className="space-y-3">
      <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
        <Stat label="ego speed" value={ego ? `${ego.speed.toFixed(1)} m/s` : "—"} />
        <Stat
          label="ego lane"
          value={
            ego
              ? `${ego.lane} (${LANE_LABEL[ego.lane] ?? "?"})`
              : "—"
          }
        />
        <Stat label="step" value={lastTick ? String(lastTick.step) : "—"} />
        <Stat
          label="episode return"
          value={totalReturn !== null && totalReturn !== undefined ? totalReturn.toFixed(1) : "—"}
          accent
        />
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <Chart title="cumulative return">
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={cumulative} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="step"
                tick={{ fill: "#64748b", fontSize: 10 }}
                stroke="#334155"
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10 }}
                stroke="#334155"
                width={36}
              />
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid #334155",
                  fontSize: 12,
                }}
                labelStyle={{ color: "#94a3b8" }}
              />
              <Line
                type="monotone"
                dataKey="return"
                stroke="#facc15"
                strokeWidth={2}
                dot={false}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </Chart>

        <Chart title={`reward components (last ${RECENT_WINDOW} ticks)`}>
          <ResponsiveContainer width="100%" height={170}>
            <LineChart data={recent} margin={{ top: 8, right: 16, bottom: 0, left: 0 }}>
              <CartesianGrid stroke="#1e293b" vertical={false} />
              <XAxis
                dataKey="step"
                tick={{ fill: "#64748b", fontSize: 10 }}
                stroke="#334155"
              />
              <YAxis
                tick={{ fill: "#64748b", fontSize: 10 }}
                stroke="#334155"
                width={36}
              />
              <Tooltip
                contentStyle={{
                  background: "#0f172a",
                  border: "1px solid #334155",
                  fontSize: 12,
                }}
                labelStyle={{ color: "#94a3b8" }}
              />
              {COMPONENTS.map(({ key, color }) => (
                <Line
                  key={key}
                  type="monotone"
                  dataKey={key}
                  stroke={color}
                  strokeWidth={1.5}
                  dot={false}
                  isAnimationActive={false}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
          <Legend />
        </Chart>
      </div>
    </section>
  );
}

function Stat({
  label,
  value,
  accent,
}: {
  label: string;
  value: string;
  accent?: boolean;
}) {
  return (
    <div className="rounded border border-slate-800 bg-slate-900 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wider text-slate-500">
        {label}
      </div>
      <div
        className={`mt-0.5 font-mono text-lg ${accent ? "text-yellow-400" : "text-slate-100"}`}
      >
        {value}
      </div>
    </div>
  );
}

function Chart({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <div className="rounded border border-slate-800 bg-slate-900 p-3">
      <div className="mb-1 text-[10px] uppercase tracking-wider text-slate-500">
        {title}
      </div>
      {children}
    </div>
  );
}

function Legend() {
  return (
    <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-[10px] text-slate-400">
      {COMPONENTS.map(({ key, color }) => (
        <div key={key} className="flex items-center gap-1">
          <span
            className="inline-block h-2 w-2 rounded-sm"
            style={{ background: color }}
          />
          <span>{key}</span>
        </div>
      ))}
    </div>
  );
}

function round2(x: number): number {
  return Math.round(x * 100) / 100;
}
