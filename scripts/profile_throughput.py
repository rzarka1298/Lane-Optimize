"""Throughput profile — env steps/sec vs gradient steps/sec.

Output decides: do we need to build libsumo from source, or is TraCI fast
enough? Per the master plan, this is Task #5 (Week 3 gate before training).

Logic:
    wall_per_env_step ≈ max(1/env_rate, 1/(train_freq · grad_rate))

If `1/env_rate >> 1/(train_freq · grad_rate)`, env is the bottleneck →
libsumo would help. Otherwise the GPU is starving for data and libsumo
wouldn't move the needle.

Run:
    PYTORCH_ENABLE_MPS_FALLBACK=1 uv run python scripts/profile_throughput.py
"""

from __future__ import annotations

import time

import numpy as np
import torch

from laneiq.agents.dqn.dqn_agent import DEFAULT_DQN_CONFIG, DQNAgent
from laneiq.env.highway_env import LaneIQEnv


def profile_env_steps(*, density: str, n_steps: int, seed: int = 0) -> tuple[float, float]:
    """Return (steps_per_sec, sim_seconds_simulated_per_wall_second)."""
    env = LaneIQEnv(density=density, max_steps=n_steps + 100, warmup_steps=30, seed=seed)
    try:
        env.reset()
        agent_action = 0  # STAY — pure env throughput, no policy cost
        # Warm a bit before timing.
        for _ in range(50):
            _, _, term, trunc, _ = env.step(agent_action)
            if term or trunc:
                env.reset()

        t0 = time.monotonic()
        for _ in range(n_steps):
            _, _, term, trunc, _ = env.step(agent_action)
            if term or trunc:
                env.reset()
        elapsed = time.monotonic() - t0
    finally:
        env.close()

    steps_per_sec = n_steps / elapsed
    # Sim runs at 0.1s per step → sim seconds per wall second:
    sim_speed_ratio = (n_steps * 0.1) / elapsed
    return steps_per_sec, sim_speed_ratio


def profile_gradient_steps(
    *, n_steps: int, device: str, batch_size: int = 64,
) -> float:
    """Return gradient steps/sec on the configured device.

    Pre-fills the buffer with synthetic transitions; not measuring any
    env time here.
    """
    agent = DQNAgent(
        obs_dim=25,
        n_actions=3,
        config=DEFAULT_DQN_CONFIG,
        device=device,
        seed=0,
    )
    rng = np.random.default_rng(0)
    # Pre-fill buffer above learning_starts.
    fill_n = max(DEFAULT_DQN_CONFIG.learning_starts + batch_size, 2 * batch_size)
    for _ in range(fill_n):
        agent.push_transition(
            rng.standard_normal(25).astype(np.float32),
            int(rng.integers(0, 3)),
            float(rng.standard_normal()),
            rng.standard_normal(25).astype(np.float32),
            bool(rng.integers(0, 2)),
        )

    # Warm a few steps.
    for _ in range(20):
        agent.train_step()

    t0 = time.monotonic()
    for _ in range(n_steps):
        agent.train_step()
    elapsed = time.monotonic() - t0
    return n_steps / elapsed


def main() -> None:
    print("=" * 70)
    print("LaneIQ throughput profile (Task #5)")
    print("=" * 70)

    # 1. Env step rate per density.
    print("\n[1/2] Env step rate (TraCI, headless)")
    print("-" * 70)
    env_results: dict[str, tuple[float, float]] = {}
    for density in ("low", "medium", "high"):
        n = 500
        rate, sim_ratio = profile_env_steps(density=density, n_steps=n)
        env_results[density] = (rate, sim_ratio)
        print(
            f"  density={density:<6}  {rate:7.1f} env-steps/sec  "
            f"({sim_ratio:.2f}x real-time, {n} steps)",
        )

    # 2. Gradient step rate on each available device.
    print("\n[2/2] Gradient step rate (DQN train_step)")
    print("-" * 70)
    devices: list[str] = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    if torch.cuda.is_available():
        devices.append("cuda")

    grad_results: dict[str, float] = {}
    for dev in devices:
        rate = profile_gradient_steps(n_steps=500, device=dev)
        grad_results[dev] = rate
        print(f"  device={dev:<5}  {rate:7.1f} grad-steps/sec")

    # 3. Bottleneck analysis.
    print("\n" + "=" * 70)
    print("Analysis")
    print("=" * 70)
    train_freq = DEFAULT_DQN_CONFIG.train_freq
    print(f"train_freq = {train_freq}  →  one grad-step per {train_freq} env-steps")
    best_dev = max(grad_results, key=grad_results.get)
    grad_rate = grad_results[best_dev]
    grad_throughput_in_env_units = grad_rate * train_freq

    print(
        f"\n  Best gradient throughput: {grad_rate:.1f}/s on {best_dev}\n"
        f"  Equivalent env throughput needed: "
        f"{grad_throughput_in_env_units:.1f} env-steps/sec\n",
    )

    print("  Per-density bottleneck:")
    for density, (env_rate, _sim_ratio) in env_results.items():
        if env_rate < grad_throughput_in_env_units:
            ratio = grad_throughput_in_env_units / env_rate
            verdict = (
                f"ENV BOTTLENECK ({ratio:.1f}x too slow for GPU; "
                "libsumo could help)"
            )
        else:
            verdict = "GPU bottleneck (libsumo wouldn't help)"
        print(f"    {density:<6}: env={env_rate:7.1f}/s  →  {verdict}")

    # 4. Wall-clock projection for full training.
    print("\n  Projected wall-clock for full training runs:")
    for total_steps_label, total_steps in (("100k smoke", 100_000), ("500k full", 500_000)):
        for density, (env_rate, _) in env_results.items():
            effective_rate = min(env_rate, grad_throughput_in_env_units)
            wall_min = total_steps / effective_rate / 60
            print(
                f"    {total_steps_label:<11} @ {density:<6}: "
                f"{wall_min:6.1f} min wall-clock  "
                f"(effective {effective_rate:.0f} steps/sec)",
            )

    print("\n" + "=" * 70)
    print(
        "Decision rule (per Project-Documentation/setup.md): build libsumo only "
        "if env is the bottleneck AND projected wall-clock is unworkable.",
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
