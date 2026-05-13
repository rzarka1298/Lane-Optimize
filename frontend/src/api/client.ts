import type {
  RunListEntry,
  SimStatusResponse,
  StartSimRequest,
  StartSimResponse,
} from "./types";

const HTTP_BASE =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ??
  "http://localhost:8000";

export const WS_URL = HTTP_BASE.replace(/^http/, "ws") + "/sim/stream";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(HTTP_BASE + path, {
    headers: { "Content-Type": "application/json" },
    ...init,
  });
  if (!res.ok) {
    const raw = await res.text().catch(() => "");
    // FastAPI errors come back as {"detail": "..."}; unwrap if present.
    let detail = raw || res.statusText;
    try {
      const parsed = JSON.parse(raw) as { detail?: unknown };
      if (typeof parsed.detail === "string") detail = parsed.detail;
    } catch {
      // raw wasn't JSON; keep it as-is.
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message);
    this.name = "ApiError";
  }
}

export const api = {
  getRuns: () => request<RunListEntry[]>("/runs"),
  getStatus: () => request<SimStatusResponse>("/sim/status"),
  startSim: (body: StartSimRequest) =>
    request<StartSimResponse>("/sim/start", {
      method: "POST",
      body: JSON.stringify(body),
    }),
  stopSim: () =>
    request<{ status: string }>("/sim/stop", { method: "POST" }),
};
