import type { SimTick } from "./types";

// Snapshot interpolation buffer — the same trick multiplayer games use to
// hide network jitter. Ticks are keyed by their *sim* time `t`, and the
// canvas renders a fixed PLAYOUT_DELAY_S behind the newest tick, so there
// are (almost) always two real snapshots bracketing the render time. How
// late any single tick arrives stops mattering: only `t` drives motion.
//
// Lives outside React state on purpose. The WS handler pushes into it and
// the canvas rAF loop reads from it directly; a React re-render is never
// on the path between a tick arriving and it reaching the screen.

// How far behind the newest tick we render: 2 ticks at 10 Hz. The render
// clock must coast up to one full tick interval past the newest arrival,
// so this leaves ~100 ms of headroom for jitter (measured: ±4 ms). At
// 150 ms, simulated 60 ms spikes still caused occasional hold frames.
const PLAYOUT_DELAY_S = 0.2;
// Keep ~5 s of ticks. Only the newest few are ever bracketed; the rest
// is slack for a render clock that briefly falls behind.
const MAX_TICKS = 50;
// Offset estimator gains (per tick). The offset is "arrival ms − t·1000";
// its true value is the minimum over arrivals (the least-delayed tick), so
// it moves down fast and up more slowly. The upward gain must also track
// steady drift — the backend loop runs at ~101 ms per 100 ms of sim time,
// i.e. +1 ms/tick, which leaves a steady-state lag of 1 ms / gain (10 ms
// here; 0.02 lagged 50 ms and starved the buffer in simulation).
const OFFSET_GAIN_DOWN = 0.2;
const OFFSET_GAIN_UP = 0.1;
// The render clock never jumps: it runs at real-time speed, nudged up to
// ±RATE_TRIM toward the target so corrections are invisible.
const RATE_GAIN = 2.0; // rate change per second of clock error
const RATE_TRIM = 0.1;
// Clock error beyond this means something discontinuous happened (tab was
// backgrounded, stream stalled for seconds) — snap instead of slewing.
const SNAP_ERROR_S = 1.0;

export interface Bracket {
  a: SimTick; // tick at or before the render time
  b: SimTick; // tick at or after the render time (=== a when clamped)
  alpha: number; // 0..1 position between a.t and b.t
}

export class TickBuffer {
  private ticks: SimTick[] = [];
  private offsetMs: number | null = null;
  private renderT: number | null = null;
  private lastFrameMs: number | null = null;

  push(tick: SimTick, arrivedAtMs: number): void {
    const newest = this.ticks[this.ticks.length - 1];
    // Sim time went backwards → a new episode (loop restart, replay start)
    // on the same socket. Start over rather than interpolate across runs.
    if (newest && tick.t <= newest.t) this.clear();

    this.ticks.push(tick);
    if (this.ticks.length > MAX_TICKS) this.ticks.shift();

    const sample = arrivedAtMs - tick.t * 1000;
    if (this.offsetMs === null) {
      this.offsetMs = sample;
    } else {
      const gain = sample < this.offsetMs ? OFFSET_GAIN_DOWN : OFFSET_GAIN_UP;
      this.offsetMs += (sample - this.offsetMs) * gain;
    }
  }

  clear(): void {
    this.ticks = [];
    this.offsetMs = null;
    this.renderT = null;
    this.lastFrameMs = null;
  }

  get latest(): SimTick | null {
    return this.ticks[this.ticks.length - 1] ?? null;
  }

  // Advance the render clock to `nowMs` and return the two ticks to blend.
  // Call once per animation frame. Null until the first tick arrives.
  sample(nowMs: number): Bracket | null {
    const first = this.ticks[0];
    const last = this.ticks[this.ticks.length - 1];
    if (!first || !last || this.offsetMs === null) return null;

    const target = (nowMs - this.offsetMs) / 1000 - PLAYOUT_DELAY_S;
    if (this.renderT === null || this.lastFrameMs === null) {
      this.renderT = target;
    } else {
      // Frame dt capped so a background-tab pause doesn't fling the clock.
      const dt = Math.min(0.1, Math.max(0, (nowMs - this.lastFrameMs) / 1000));
      const error = target - this.renderT;
      if (Math.abs(error) > SNAP_ERROR_S) {
        this.renderT = target;
      } else {
        const rate = 1 + clamp(error * RATE_GAIN, -RATE_TRIM, RATE_TRIM);
        this.renderT += dt * rate;
      }
    }
    this.lastFrameMs = nowMs;

    // Never extrapolate: if the buffer starves, hold at the newest tick.
    this.renderT = clamp(this.renderT, first.t, last.t);
    const t = this.renderT;

    // Newest-first scan — the bracket is almost always the last pair.
    for (let i = this.ticks.length - 1; i > 0; i--) {
      const a = this.ticks[i - 1]!;
      const b = this.ticks[i]!;
      if (a.t <= t) {
        const span = b.t - a.t;
        return { a, b, alpha: span > 0 ? clamp((t - a.t) / span, 0, 1) : 1 };
      }
    }
    return { a: first, b: first, alpha: 0 };
  }
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
