import { useEffect, useRef } from "react";
import type { SimTick, Vehicle, VType } from "../api/types";

// World geometry — see Project-Documentation/sumo_scenarios/network.md.
// Roads are 3 lanes × 3.2 m, straight: 1 km for most scenarios, 10 km for
// rush. At true aspect ratio lanes would be unreadable, so we exaggerate
// vertically. No scenario loops — the follow-camera scrolls a straight road.
const N_LANES = 3;
const LANE_WIDTH_M = 3.2;

// Backend sim loop targets 10 Hz (backend/app/services/simulator.py:_run_loop).
// One tick = 100 ms of sim time. We lerp positions over this window.
const TICK_DURATION_MS = 100;

// Follow-camera: show this much road centered on the ego (m). With this
// the road feels infinite — the ego is always visible at center and the
// world scrolls past. Without it, audiences see vehicles compressed into
// a few-px specks on the full road.
const WINDOW_M = 500;
// Off-screen margin so vehicles entering the window don't pop in abruptly.
const OFFSCREEN_MARGIN_M = 30;
// Min vehicle width on screen — at WINDOW_M=500 on a 1200 px canvas this
// is barely the true vehicle length, but ensures they stay visible.
const MIN_VEHICLE_PX = 18;

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

  if (!state.lastTick) return;

  const now = performance.now();
  let progress = 1;
  if (state.prevTick && state.lastTickArrivedAt !== null) {
    progress = Math.min(
      1,
      Math.max(0, (now - state.lastTickArrivedAt) / TICK_DURATION_MS),
    );
  }

  // Find the ego. Without it we can't camera-follow — render the lane
  // bands (already done) and bail.
  const ego = state.lastTick.vehicles.find((v) => v.is_ego);
  if (!ego) return;
  const prevEgo = state.prevTick?.vehicles.find((v) => v.is_ego);

  // Camera center = ego's lerped world x. Straight road, so a plain lerp.
  let cameraX = ego.x;
  if (prevEgo) {
    cameraX = prevEgo.x + (ego.x - prevEgo.x) * progress;
  }

  // px per meter under the follow-camera transform.
  const xScale = w / WINDOW_M;
  const halfWindow = WINDOW_M / 2;
  const vehicleH = laneH * 0.55;

  // 100 m gridlines, follow-camera style. Walk the world positions that
  // could be visible in the current window and render those that are.
  ctx.strokeStyle = "#1e293b";
  ctx.fillStyle = "#475569";
  ctx.font = "10px ui-monospace, monospace";
  ctx.textBaseline = "top";
  // Start a bit behind the leftmost visible meter, rounded down to 100m.
  const leftWorld = cameraX - halfWindow - OFFSCREEN_MARGIN_M;
  const rightWorld = cameraX + halfWindow + OFFSCREEN_MARGIN_M;
  const firstGrid = Math.max(0, Math.floor(leftWorld / 100) * 100);
  for (let m = firstGrid; m <= rightWorld + 100; m += 100) {
    const rel = m - cameraX;
    if (Math.abs(rel) > halfWindow + OFFSCREEN_MARGIN_M) continue;
    const screenX = w / 2 + rel * xScale;
    ctx.beginPath();
    ctx.moveTo(screenX, 0);
    ctx.lineTo(screenX, h);
    ctx.stroke();
    ctx.fillText(`${m}m`, screenX + 2, 2);
  }

  // Vehicles. Build prev-by-id once.
  const prevById = new Map<string, Vehicle>();
  if (state.prevTick) {
    for (const v of state.prevTick.vehicles) prevById.set(v.id, v);
  }

  // Render non-ego first so the ego paints on top.
  const sorted = [...state.lastTick.vehicles].sort(
    (a, b) => (a.is_ego ? 1 : 0) - (b.is_ego ? 1 : 0),
  );

  for (const v of sorted) {
    const prev = prevById.get(v.id);
    const vx = prev ? prev.x + (v.x - prev.x) * progress : v.x;
    const vy = prev ? prev.y + (v.y - prev.y) * progress : v.y;

    const rel = vx - cameraX;
    if (Math.abs(rel) > halfWindow + OFFSCREEN_MARGIN_M) continue;

    const screenX = w / 2 + rel * xScale;
    // y formula mirrored from backend/app/schemas.md.
    const screenY = h - (vy / LANE_WIDTH_M + 0.5) * laneH;

    const trueWidth = 4.5 * xScale;
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
      ctx.fillStyle = "#facc15";
      ctx.font = "bold 11px ui-monospace, monospace";
      ctx.textBaseline = "bottom";
      ctx.textAlign = "center";
      ctx.fillText("EGO", screenX, screenY - vehicleH / 2 - 2);
      ctx.textAlign = "start";
      ctx.textBaseline = "top";
    }
  }

  // Camera overlay: km marker showing the ego's absolute road position
  // (helps the viewer orient when the road scrolls and looks endless).
  ctx.fillStyle = "rgba(15, 23, 42, 0.85)";
  ctx.fillRect(w - 110, 6, 104, 18);
  ctx.fillStyle = "#94a3b8";
  ctx.font = "11px ui-monospace, monospace";
  ctx.textBaseline = "top";
  ctx.textAlign = "right";
  ctx.fillText(`road pos: ${cameraX.toFixed(0)}m`, w - 10, 10);
  ctx.textAlign = "start";
}
