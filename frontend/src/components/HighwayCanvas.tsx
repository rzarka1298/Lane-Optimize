import { useEffect, useRef } from "react";
import type { SimTick, Vehicle, VType } from "../api/types";

// World geometry — see Project-Documentation/sumo_scenarios/network.md.
// The real road is 1000 m × ~9.6 m (3 lanes × 3.2 m). At true aspect ratio
// (~104:1) lanes would be unreadable, so we exaggerate vertically.
const ROAD_LENGTH_M = 1000;
const N_LANES = 3;
const LANE_WIDTH_M = 3.2;

// Backend sim loop targets 10 Hz (backend/app/services/simulator.py:_run_loop).
// One tick = 100 ms of sim time. We lerp positions over this window.
const TICK_DURATION_MS = 100;

// Vehicles are 4.5–5.5 m long; at full road scale (1.2 px/m on a 1200 px
// canvas) that's ~5 px wide. Floor it so they're actually visible.
const MIN_VEHICLE_PX = 14;

const VEHICLE_FILL: Record<VType, string> = {
  ego: "#facc15",
  aggressive: "#ef4444",
  normal: "#e5e7eb",
  slow: "#3b82f6",
};

interface Props {
  prevTick: SimTick | null;
  lastTick: SimTick | null;
  lastTickArrivedAt: number | null;
}

export function HighwayCanvas({
  prevTick,
  lastTick,
  lastTickArrivedAt,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  // Refs hold the latest props so the rAF loop reads fresh state without
  // having to re-bind on every render.
  const stateRef = useRef({ prevTick, lastTick, lastTickArrivedAt });
  stateRef.current = { prevTick, lastTick, lastTickArrivedAt };

  useEffect(() => {
    const canvas = canvasRef.current;
    const container = containerRef.current;
    if (!canvas || !container) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const resize = () => {
      const dpr = window.devicePixelRatio || 1;
      const { width, height } = container.getBoundingClientRect();
      canvas.width = Math.floor(width * dpr);
      canvas.height = Math.floor(height * dpr);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };
    resize();
    const ro = new ResizeObserver(resize);
    ro.observe(container);

    const loop = () => {
      drawFrame(ctx, canvas, stateRef.current);
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);

    return () => {
      ro.disconnect();
      if (rafRef.current !== null) cancelAnimationFrame(rafRef.current);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative h-72 w-full overflow-hidden rounded border border-slate-800 bg-slate-950"
    >
      <canvas ref={canvasRef} className="block h-full w-full" />
      {!lastTick && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-slate-600">
          waiting for first tick…
        </div>
      )}
    </div>
  );
}

interface RenderState {
  prevTick: SimTick | null;
  lastTick: SimTick | null;
  lastTickArrivedAt: number | null;
}

function drawFrame(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  state: RenderState,
) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.width / dpr;
  const h = canvas.height / dpr;

  ctx.fillStyle = "#020617";
  ctx.fillRect(0, 0, w, h);

  const laneH = h / N_LANES;

  // Lane bands — alternating tones so the 3 lanes read at a glance.
  for (let lane = 0; lane < N_LANES; lane++) {
    const top = (N_LANES - 1 - lane) * laneH;
    ctx.fillStyle = lane === 1 ? "#0f172a" : "#1e293b";
    ctx.fillRect(0, top, w, laneH);
  }

  // Dashed lane separators.
  ctx.strokeStyle = "#475569";
  ctx.lineWidth = 1;
  ctx.setLineDash([14, 14]);
  for (let i = 1; i < N_LANES; i++) {
    const y = i * laneH;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.setLineDash([]);

  // 100 m gridlines for spatial reference.
  ctx.strokeStyle = "#1e293b";
  ctx.fillStyle = "#475569";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  for (let m = 100; m < ROAD_LENGTH_M; m += 100) {
    const x = (m / ROAD_LENGTH_M) * w;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, h);
    ctx.stroke();
    ctx.fillText(`${m}m`, x + 2, 2);
  }

  if (!state.lastTick) return;

  const now = performance.now();
  let progress = 1;
  if (state.prevTick && state.lastTickArrivedAt !== null) {
    progress = Math.min(
      1,
      Math.max(0, (now - state.lastTickArrivedAt) / TICK_DURATION_MS),
    );
  }

  const prevById = new Map<string, Vehicle>();
  if (state.prevTick) {
    for (const v of state.prevTick.vehicles) prevById.set(v.id, v);
  }

  const xScale = w / ROAD_LENGTH_M;
  const vehicleH = laneH * 0.55;

  for (const v of state.lastTick.vehicles) {
    const prev = prevById.get(v.id);
    const x = prev ? prev.x + (v.x - prev.x) * progress : v.x;
    const y = prev ? prev.y + (v.y - prev.y) * progress : v.y;

    const screenX = x * xScale;
    // Map "lateral offset, in meters from lane 0 center" → screen y, with
    // lane 2 at the top. See backend/app/schemas.md for the y formula.
    const screenY = h - (y / LANE_WIDTH_M + 0.5) * laneH;

    const trueWidth = 4.5 * xScale; // worst-case length for screen sizing
    const width = Math.max(MIN_VEHICLE_PX, trueWidth);

    ctx.fillStyle = VEHICLE_FILL[v.vtype];
    ctx.fillRect(screenX - width / 2, screenY - vehicleH / 2, width, vehicleH);

    if (v.is_ego) {
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 2;
      ctx.strokeRect(
        screenX - width / 2,
        screenY - vehicleH / 2,
        width,
        vehicleH,
      );
      // Label above the ego so a viewer instantly clocks "this is the AI."
      ctx.fillStyle = "#facc15";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textBaseline = "bottom";
      ctx.textAlign = "center";
      ctx.fillText("EGO", screenX, screenY - vehicleH / 2 - 2);
      ctx.textAlign = "start";
      ctx.textBaseline = "top";
    }
  }
}
