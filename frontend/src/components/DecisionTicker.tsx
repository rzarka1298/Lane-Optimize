import { Fragment } from "react";
import type { Action, Decision } from "../api/types";

const ACTION_LABEL: Record<Action, string> = {
  stay: "STAY",
  change_left: "CHANGE LEFT",
  change_right: "CHANGE RIGHT",
};

const ACTION_GLYPH: Record<Action, string> = {
  stay: "■",
  change_left: "◀",
  change_right: "▶",
};

// Tailwind class strings per action — kept as full strings so the JIT
// content-scan picks them up.
const ACTION_CLASSES: Record<Action, string> = {
  stay: "bg-slate-700 text-slate-100 border-slate-500",
  change_left: "bg-cyan-700 text-cyan-50 border-cyan-400",
  change_right: "bg-amber-700 text-amber-50 border-amber-400",
};

interface Props {
  decision: Decision | null;
  step: number | null;
  simTimeS: number | null;
}

export function DecisionTicker({ decision, step, simTimeS }: Props) {
  if (!decision) {
    return (
      <section className="rounded border border-slate-800 bg-slate-900 px-4 py-3 text-sm text-slate-500">
        AI decision — waiting for first tick…
      </section>
    );
  }

  const { action, explanation, q_values } = decision;
  const cls = ACTION_CLASSES[action];

  return (
    <section className="rounded border border-slate-800 bg-slate-900 px-4 py-3">
      <div className="flex items-start gap-4">
        <div
          // Re-keying on `action` re-mounts the chip and re-fires the flash.
          key={action}
          className={`flex shrink-0 items-center gap-2 rounded border px-3 py-2 text-base font-bold tracking-wide animate-flash ${cls}`}
        >
          <span aria-hidden className="text-lg leading-none">
            {ACTION_GLYPH[action]}
          </span>
          <span>{ACTION_LABEL[action]}</span>
        </div>
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-3 text-xs text-slate-500">
            <span className="uppercase tracking-wider">AI decision</span>
            {step !== null && simTimeS !== null && (
              <span className="font-mono">
                step {step} · {simTimeS.toFixed(1)}s
              </span>
            )}
          </div>
          <p className="mt-1 text-slate-100">{explanation}</p>
          {q_values && <QValueBars qValues={q_values} chosen={action} />}
        </div>
      </div>
    </section>
  );
}

const Q_LABELS: readonly Action[] = ["stay", "change_left", "change_right"];

function QValueBars({
  qValues,
  chosen,
}: {
  qValues: number[];
  chosen: Action;
}) {
  // Normalize to [0, 1] for bar widths. Map by shifting so min = 0; this
  // visually matches the *relative* preference, which is what argmax sees.
  const minQ = Math.min(...qValues);
  const maxQ = Math.max(...qValues);
  const span = maxQ - minQ || 1;

  return (
    <div className="mt-2 grid grid-cols-[6rem_1fr_3.5rem] gap-x-2 gap-y-1 text-xs">
      {Q_LABELS.map((a, i) => {
        const q = qValues[i];
        if (q === undefined) return null;
        const pct = ((q - minQ) / span) * 100;
        const isChosen = a === chosen;
        return (
          <Fragment key={a}>
            <span
              className={`text-right ${isChosen ? "text-slate-100 font-semibold" : "text-slate-500"}`}
            >
              {a}
            </span>
            <div className="relative h-3 self-center overflow-hidden rounded bg-slate-800">
              <div
                className={`h-full ${isChosen ? "bg-yellow-400" : "bg-slate-600"}`}
                style={{ width: `${pct}%` }}
              />
            </div>
            <span
              className={`text-right font-mono ${isChosen ? "text-slate-100" : "text-slate-500"}`}
            >
              {q.toFixed(2)}
            </span>
          </Fragment>
        );
      })}
    </div>
  );
}

