import { useEffect, useRef, useState } from "react";
import type { SimTick, StreamFrame } from "./types";
import { WS_URL } from "./client";

export type ConnectionStatus =
  | "idle"
  | "connecting"
  | "open"
  | "closed"
  | "error";

export interface SimStreamState {
  status: ConnectionStatus;
  // prevTick + lastTick + lastTickArrivedAt let consumers interpolate
  // smoothly between the 10 Hz frames the backend sends.
  prevTick: SimTick | null;
  lastTick: SimTick | null;
  // performance.now() timestamp of when lastTick arrived. null if no ticks yet.
  lastTickArrivedAt: number | null;
  history: SimTick[];
  // Last control-frame message (started/stopped/error envelope text).
  lastMessage: string | null;
}

const HISTORY_LIMIT = 600; // 60 sec at 10 Hz; rolling buffer for charts.

// Hook lifecycle:
//   enabled=false → socket stays closed, status="idle"
//   enabled=true  → connect; on close 4004 ("no sim") give up; on other
//                   transient close, exponential backoff up to 5s.
// The caller is responsible for calling api.startSim() before flipping
// enabled to true — this hook only consumes the stream.
export function useSimStream(enabled: boolean): SimStreamState {
  const [state, setState] = useState<SimStreamState>({
    status: "idle",
    prevTick: null,
    lastTick: null,
    lastTickArrivedAt: null,
    history: [],
    lastMessage: null,
  });
  const wsRef = useRef<WebSocket | null>(null);
  const retryRef = useRef<number | null>(null);
  const attemptRef = useRef(0);

  useEffect(() => {
    if (!enabled) {
      wsRef.current?.close();
      wsRef.current = null;
      if (retryRef.current !== null) {
        window.clearTimeout(retryRef.current);
        retryRef.current = null;
      }
      attemptRef.current = 0;
      setState({
        status: "idle",
        prevTick: null,
        lastTick: null,
        lastTickArrivedAt: null,
        history: [],
        lastMessage: null,
      });
      return;
    }

    let cancelled = false;

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
          const arrivedAt = performance.now();
          setState((s) => {
            const next = [...s.history, tick];
            if (next.length > HISTORY_LIMIT) next.shift();
            return {
              ...s,
              prevTick: s.lastTick,
              lastTick: tick,
              lastTickArrivedAt: arrivedAt,
              history: next,
            };
          });
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
