#!/usr/bin/env python3
"""Recover the G9 training curve from the run's own TensorBoard event file.

The event file is the authoritative per-iteration record: `train_strafer_navigation.py`
writes one scalar point per iteration per tag, so the curve does not depend on stdout
surviving. (It did not survive here — the launcher's final copy overwrote the complete
stdout log with an earlier partial recovery of it. The event file was unaffected, and
`model_99.pt` in the same directory independently confirms the run reached iteration 99.)

`--max_iterations 100` runs iterations 0 through 99 — there is no row 100 — and iteration 0
carries first-iteration warm-up, so it is excluded from the throughput summary. Both facts
are §11.6 of the scoping report.

Usage:

    python3 extract_training_curve.py <events.out.tfevents.*> --out training/training-curve.csv
"""

import argparse
import csv
import pathlib
import statistics as st

# TensorBoard tag -> column name, chosen to line up with the baseline record's table.
COLUMNS = {
    "Perf/total_fps": "steps_per_second",
    "Perf/learning_time": "learning_s",
    "Perf/collection_time": "collection_s",
    "Train/mean_reward": "mean_reward",
    "Train/mean_episode_length": "mean_ep_length",
    "Loss/value": "value_loss",
    "Loss/surrogate": "surrogate_loss",
    "Loss/entropy": "entropy",
    "Loss/learning_rate": "learning_rate",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("events", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, required=True)
    args = ap.parse_args()

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    ea = EventAccumulator(str(args.events), size_guidance={"scalars": 0})
    ea.Reload()
    available = set(ea.Tags()["scalars"])

    rows: dict[int, dict] = {}
    for tag, col in COLUMNS.items():
        if tag not in available:
            continue
        for s in ea.Scalars(tag):
            rows.setdefault(s.step, {"iteration": s.step})[col] = s.value

    ordered = [rows[k] for k in sorted(rows)]
    cols = ["iteration"] + [c for c in COLUMNS.values() if any(c in r for r in ordered)]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in ordered:
            w.writerow({c: r.get(c, "") for c in cols})

    steps = [r["steps_per_second"] for r in ordered
             if r["iteration"] > 0 and "steps_per_second" in r]
    print(f"iterations: {len(ordered)} ({ordered[0]['iteration']}..{ordered[-1]['iteration']})")
    print(f"steps/s excluding iteration 0: n={len(steps)} mean={st.mean(steps):.1f} "
          f"min={min(steps):.0f} max={max(steps):.0f}")
    bad = [r["iteration"] for r in ordered
           if any(isinstance(v, float) and v != v for v in r.values())]
    print(f"NaN rows: {bad or 'none'}")
    for it in (0, 10, 25, 50, 75, 99):
        r = rows.get(it)
        if r:
            print(f"  iter {it:3d}: steps/s={r.get('steps_per_second', 0):5.0f} "
                  f"reward={r.get('mean_reward', 0):8.2f} ep_len={r.get('mean_ep_length', 0):8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
