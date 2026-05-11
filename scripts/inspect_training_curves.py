"""Read a TensorBoard event file and print summary stats per scalar tag.

Used for Task #17 — diagnosing the DQN loss spike on the 100k smoke run
without needing the full TensorBoard server. Prints min/max/mean/last
plus a 10-bucket histogram over the recorded steps, so we can see where
the loss diverged.

Run:
    uv run python scripts/inspect_training_curves.py runs/dqn_medium_100k/
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def load_scalars(log_dir: Path) -> dict[str, list[tuple[int, float]]]:
    """Return {tag: [(step, value), ...]} for every scalar in the dir."""
    acc = EventAccumulator(str(log_dir), size_guidance={"scalars": 0})  # 0 = all
    acc.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in acc.Tags()["scalars"]:
        out[tag] = [(ev.step, ev.value) for ev in acc.Scalars(tag)]
    return out


def _bucketed(values: list[float], n_buckets: int = 10) -> list[float]:
    """Split `values` into n equal-time buckets, return mean per bucket."""
    if not values:
        return []
    arr = np.array(values)
    chunks = np.array_split(arr, n_buckets)
    return [float(c.mean()) if c.size else float("nan") for c in chunks]


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <tb_log_dir>", file=sys.stderr)
        return 2
    log_dir = Path(argv[1])

    scalars = load_scalars(log_dir)
    if not scalars:
        print(f"no scalars found in {log_dir}", file=sys.stderr)
        return 1

    print(f"\nTensorBoard event-file inspection: {log_dir}")
    print(f"Tags found: {len(scalars)}\n")

    # Pretty-print summary per tag.
    for tag in sorted(scalars):
        points = scalars[tag]
        steps = [p[0] for p in points]
        vals = [p[1] for p in points]
        arr = np.array(vals)
        n = len(arr)
        if n == 0:
            continue
        print(f"== {tag}  ({n} points, steps {min(steps)} .. {max(steps)})")
        print(
            f"   min={arr.min():12.4f}  max={arr.max():12.4f}  "
            f"mean={arr.mean():12.4f}  last={vals[-1]:12.4f}",
        )
        buckets = _bucketed(vals, n_buckets=10)
        bucket_strs = "  ".join(f"{b:9.3f}" for b in buckets)
        print(f"   bucket means (10 eq-time bins): {bucket_strs}")
        print()

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
