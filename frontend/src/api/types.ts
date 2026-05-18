// Wire format mirrored from backend/app/schemas.py.
// Single source of truth: Project-Documentation/backend/schemas.md.
// Renaming a field on the Python side is a breaking change for these types.

export type Density = "low" | "medium" | "high" | "gauntlet" | "rush";

export type Action = "stay" | "change_left" | "change_right";

export type VType = "aggressive" | "normal" | "slow" | "ego";

export type Algorithm = "dqn" | "ppo";

export type FrameType = "tick" | "started" | "stopped" | "error";

export interface Vehicle {
  id: string;
  x: number;
  y: number;
  speed: number;
  lane: number;
  is_ego: boolean;
  vtype: VType;
}

export interface Decision {
  action: Action;
  explanation: string;
  q_values: number[] | null;
}

export interface SimTick {
  t: number;
  step: number;
  ego_id: string;
  vehicles: Vehicle[];
  decision: Decision | null;
  terminated: boolean;
  truncated: boolean;
  info: Record<string, number>;
}

// WebSocket envelope. Backend sends with exclude_none=True, so optional
// fields may be missing; FrameType discriminator tells you which to expect.
export interface StreamFrame {
  type: FrameType;
  tick?: SimTick;
  message?: string;
}

export interface StartSimRequest {
  agent: string;
  density: Density;
  seed?: number | null;
  max_steps?: number;
}

export interface StartSimResponse {
  run_id: string;
  agent: string;
  density: Density;
  seed: number | null;
}

export interface SimStatusResponse {
  running: boolean;
  run_id: string | null;
  agent: string | null;
  density: Density | null;
  seed: number | null;
  step: number;
  sim_time_s: number;
  is_replay: boolean;
}

export interface RunListEntry {
  name: string;
  algorithm: Algorithm;
  path: string;
  size_bytes: number;
}

export interface ReplayListEntry {
  name: string;
  agent: string;
  density: string;
  n_ticks: number;
  size_bytes: number;
}

export interface StartReplayRequest {
  replay: string;
}
