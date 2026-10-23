import { useEffect, useRef, useState } from "react";
import type { SimTick, StreamFrame } from "./types";
import { WS_URL } from "./client";
import { TickBuffer } from "./tickBuffer";

export type ConnectionStatus =
  | "idle"
  | "connecting"
  | "open"
  | "closed"
  | "error";

export interface SimStreamState {
  status: ConnectionStatus;
  // Every tick, pushed the moment it arrives. Stable object for the hook's
  // lifetime — HighwayCanvas reads it from its rAF loop (see tickBuffer.ts).
  buffer: TickBuffer;
  // lastTick + history are React state, committed at most every
  // COMMIT_INTERVAL_MS (sooner on episode end or action change). They feed
  // the panels, never the canvas.
  lastTick: SimTick | null;
  history: SimTick[];
  // Last control-frame message (started/stopped/error envelope text).
  lastMessage: string | null;
}

const HISTORY_LIMIT = 600; // 60 sec at 10 Hz; rolling buffer for charts.
// ~4 Hz. Re-rendering App + two Recharts charts at the full 10 Hz tick rate
// competed with the canvas for the main thread and caused dropped frames.
const COMMIT_INTERVAL_MS = 250;

// Hook lifecycle:
//   enabled=false → socket stays closed, status="idle"
//   enabled=true  → connect; on close 4004 ("no sim") give up; on other
//                   transient close, exponential backoff up to 5s.
// The caller is responsible for calling api.startSim() before flipping
// enabled to true — this hook only consumes the stream.
export function useSimStream(enabled: boolean): SimStreamState {
  const bufferRef = useRef<TickBuffer>(new TickBuffer());
  const [state, setState] = useState<SimStreamState>({
    status: "idle",
    buffer: bufferRef.current,
    lastTick: null,
    history: [],
    lastMessage: null,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number | null>(null);
  const attemptRef = useRef(0);
  // Uncommitted tick state + throttle bookkeeping (see COMMIT_INTERVAL_MS).
  const historyRef = useRef<SimTick[]>([]);
  const pendingTickRef = useRef<SimTick | null>(null);
  const committedActionRef = useRef<string | null>(null);
  const commitTimerRef = useRef<number | null>(null);
  const lastCommitAtRef = useRef(0);

  useEffect(() => {
    const cancelCommit = () => {
      if (commitTimerRef.current !== null) {
        window.clearTimeout(commitTimerRef.current);
        commitTimerRef.current = null;
      }
    };

    if (!enabled) {
      wsRef.current?.close();
      wsRef.current = null;
      if (retryRef.current !== null) {
        window.clearTimeout(retryRef.current);
        retryRef.current = null;
      }
      cancelCommit();
      attemptRef.current = 0;
      bufferRef.current.clear();
      historyRef.current = [];
      pendingTickRef.current = null;
      committedActionRef.current = null;
      setState({
        status: "idle",
        buffer: bufferRef.current,
        lastTick: null,
        history: [],
        lastMessage: null,
      });
      return;
    }

    let cancelled = false;

    const commit = () => {
      cancelCommit();
      if (cancelled) return;
      lastCommitAtRef.current = performance.now();
      const tick = pendingTickRef.current;
      committedActionRef.current = tick?.decision?.action ?? null;
      const history = historyRef.current.slice();
      setState((s) => ({ ...s, lastTick: tick, history }));
    };

    const connect = () => {
      if (cancelled) return;
      setState((s) => ({ ...s, status: "connecting" }));
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => {
        attemptRef.current = 0;
        setState((s) => ({ ...s, status: "open" }));
      };

      ws.onmessage = (ev) => {
        let frame: StreamFrame;
        try {
          frame = JSON.parse(ev.data as string) as StreamFrame;
        } catch {
          return;
        }
        if (frame.type === "tick" && frame.tick) {
          const tick = frame.tick;
          const now = performance.now();
          bufferRef.current.push(tick, now);
          // A new episode on the same socket (t restarted) drops old history.
          const prev = historyRef.current[historyRef.current.length - 1];
          if (prev && tick.t <= prev.t) historyRef.current = [];
          historyRef.current.push(tick);
          if (historyRef.current.length > HISTORY_LIMIT) {
            historyRef.current.shift();
          }
          pendingTickRef.current = tick;

          // Commit now if something the user must see promptly happened:
          // the episode ended (App's loop-restart watches lastTick) or the
          // AI's action changed (DecisionTicker). Otherwise throttle.
          const urgent =
            tick.terminated ||
            tick.truncated ||
            (tick.decision?.action ?? null) !== committedActionRef.current;
          const wait = COMMIT_INTERVAL_MS - (now - lastCommitAtRef.current);
          if (urgent || wait <= 0) {
            commit();
          } else if (commitTimerRef.current === null) {
            commitTimerRef.current = window.setTimeout(commit, wait);
          }
        } else if (frame.message) {
          setState((s) => ({ ...s, lastMessage: frame.message ?? null }));
        }
      };

      ws.onerror = () => {
        setState((s) => ({ ...s, status: "error" }));
      };

      ws.onclose = (ev) => {
        wsRef.current = null;
        setState((s) => ({ ...s, status: "closed" }));
        // Application close codes (see backend/app/routes/stream.py):
        //   4004 = no sim running. Don't retry — caller must start a sim first.
        //   4500 = server error. Retry with backoff.
        //   1000/1001 = normal closure. Don't retry.
        if (cancelled) return;
        if (ev.code === 4004 || ev.code === 1000 || ev.code === 1001) return;
        const delay = Math.min(5000, 250 * 2 ** attemptRef.current);
        attemptRef.current += 1;
        retryRef.current = window.setTimeout(connect, delay);
      };
    };

    connect();

    return () => {
      cancelled = true;
      cancelCommit();
      if (retryRef.current !== null) {
        window.clearTimeout(retryRef.current);
        retryRef.current = null;
      }
      wsRef.current?.close();
      wsRef.current = null;
    };
  }, [enabled]);

  return state;
}
