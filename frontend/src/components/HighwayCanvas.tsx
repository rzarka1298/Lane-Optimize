import { useEffect, useRef } from "react";
import type { Bracket, TickBuffer } from "../api/tickBuffer";
import type { Vehicle, VType } from "../api/types";

// World geometry — see Project-Documentation/sumo_scenarios/network.md.
// Roads are 3 lanes × 3.2 m, straight: 1 km for most scenarios, 10 km for
// rush. At true aspect ratio lanes would be unreadable, so we exaggerate
// vertically. No scenario loops — the follow-camera scrolls a straight road.
const N_LANES = 3;
const LANE_WIDTH_M = 3.2;

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

// SUMO (no sublane model) changes lanes instantly: y jumps 3.2 m in one
// 0.1 s step. We ease the *displayed* y with a critically-damped spring so
// a lane change reads as a ~0.5 s glide. Visual only — the sim is untouched.
const LATERAL_OMEGA = 9; // rad/s; settles a full lane in ~0.5 s, no overshoot
// Vehicles entering / leaving the scene fade instead of popping.
const FADE_MS = 300;
// An x jump this large between adjacent ticks is a teleport (re-insert,
// new episode), never motion — snap instead of sweeping across the road.
const TELEPORT_M = 100;

const VEHICLE_FILL: Record<VType, string> = {
  ego: "#facc15",
  aggressive: "#ef4444",
  normal: "#e5e7eb",
  slow: "#3b82f6",
};

interface Props {
  buffer: TickBuffer;
  hasTick: boolean; // only drives the "waiting" overlay
}

// Per-vehicle display state that persists across frames.
interface Track {
  y: number; // eased display y (m)
  vy: number; // spring velocity (m/s)
  firstSeenMs: number;
  goneAtMs: number | null; // set when the vehicle leaves the scene
  last: Vehicle; // last interpolated pose, drawn while fading out
  lastX: number;
}

export function HighwayCanvas({ buffer, hasTick }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const rafRef = useRef<number | null>(null);
  // The buffer is a stable object, but read it through a ref so the rAF
  // loop never needs re-binding.
  const bufferRef = useRef(buffer);
  bufferRef.current = buffer;

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

    const tracks = new Map<string, Track>();
    let lastFrameMs: number | null = null;
    const loop = () => {
      const now = performance.now();
      const dt = lastFrameMs === null ? 0 : Math.min(0.05, (now - lastFrameMs) / 1000);
      lastFrameMs = now;
      drawFrame(ctx, canvas, bufferRef.current.sample(now), tracks, now, dt);
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
      {!hasTick && (
        <div className="pointer-events-none absolute inset-0 flex items-center justify-center text-sm text-slate-600">
          waiting for first tick…
        </div>
      )}
    </div>
  );
}

// Blend the two bracketing ticks into one pose per vehicle id. Vehicles
// only in `b` (just spawned) take b's pose; vehicles only in `a` have left.
function interpolate({ a, b, alpha }: Bracket): Vehicle[] {
  const prevById = new Map<string, Vehicle>();
  for (const v of a.vehicles) prevById.set(v.id, v);
  return b.vehicles.map((v) => {
    const p = prevById.get(v.id);
    if (!p || a === b || Math.abs(v.x - p.x) > TELEPORT_M) return v;
    return {
      ...v,
      x: p.x + (v.x - p.x) * alpha,
      y: p.y + (v.y - p.y) * alpha,
    };
  });
}

// Update per-vehicle tracks from this frame's poses: spring the display y,
// start fades for arrivals and departures, drop fully-faded tracks.
function updateTracks(
  tracks: Map<string, Track>,
  poses: Vehicle[],
  now: number,
  dt: number,
) {
  const present = new Set<string>();
  for (const v of poses) {
    present.add(v.id);
    let tr = tracks.get(v.id);
    if (!tr || tr.goneAtMs !== null) {
      // First sighting (or back after leaving): start at the true y — no
      // slide-in from wherever it was — and fade in.
      tr = { y: v.y, vy: 0, firstSeenMs: now, goneAtMs: null, last: v, lastX: v.x };
      tracks.set(v.id, tr);
    } else if (Math.abs(v.x - tr.lastX) > TELEPORT_M) {
      tr.y = v.y;
      tr.vy = 0;
    }
    // Critically-damped spring toward the target y (semi-implicit Euler).
    const accel = LATERAL_OMEGA * LATERAL_OMEGA * (v.y - tr.y) - 2 * LATERAL_OMEGA * tr.vy;
    tr.vy += accel * dt;
    tr.y += tr.vy * dt;
    tr.last = v;
    tr.lastX = v.x;
  }
  for (const [id, tr] of tracks) {
    if (present.has(id)) continue;
    if (tr.goneAtMs === null) tr.goneAtMs = now;
    else if (now - tr.goneAtMs > FADE_MS) tracks.delete(id);
  }
}

function trackAlpha(tr: Track, now: number): number {
  if (tr.last.is_ego) return 1;
  const fadeIn = Math.min(1, (now - tr.firstSeenMs) / FADE_MS);
  const fadeOut = tr.goneAtMs === null ? 1 : Math.max(0, 1 - (now - tr.goneAtMs) / FADE_MS);
  return fadeIn * fadeOut;
}

function drawFrame(
  ctx: CanvasRenderingContext2D,
  canvas: HTMLCanvasElement,
  bracket: Bracket | null,
  tracks: Map<string, Track>,
  now: number,
  dt: number,
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

  // Stream reset (stop, new run): forget every track so nothing from the
  // previous episode fades or eases into the next one.
  if (!bracket) {
    tracks.clear();
    return;
  }

  const poses = interpolate(bracket);
  updateTracks(tracks, poses, now, dt);

  // Find the ego. Without it we can't camera-follow — render the lane
  // bands (already done) and bail.
  const ego = poses.find((v) => v.is_ego);
  if (!ego) return;

  // Camera center = ego's interpolated world x. Straight road, so x only.
  const cameraX = ego.x;

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

  // Vehicles, from tracks so departed cars keep drawing while they fade.
  // Non-ego first so the ego paints on top.
  const sorted = [...tracks.values()].sort(
    (a, b) => (a.last.is_ego ? 1 : 0) - (b.last.is_ego ? 1 : 0),
  );

  for (const tr of sorted) {
    const v = tr.last;
    const rel = v.x - cameraX;
    if (Math.abs(rel) > halfWindow + OFFSCREEN_MARGIN_M) continue;

    const screenX = w / 2 + rel * xScale;
    // y formula mirrored from backend/app/schemas.md; eased y, not raw.
    const screenY = h - (tr.y / LANE_WIDTH_M + 0.5) * laneH;

    const trueWidth = 4.5 * xScale;
    const width = Math.max(MIN_VEHICLE_PX, trueWidth);

    ctx.globalAlpha = trackAlpha(tr, now);
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
  ctx.globalAlpha = 1;

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
