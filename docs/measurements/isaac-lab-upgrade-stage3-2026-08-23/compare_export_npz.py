#!/usr/bin/env python3
"""Compare two export-trajectory dumps array by array.

The baseline dump stores the observation sequences alongside the outputs, so a
post-bump dump produced with ``--replay <baseline npz>`` is driven by the very
same inputs. What is left to compare is therefore the policy's response: the
per-step action ``(steps, 3)`` and, for recurrent artifacts, the threaded hidden
state ``(steps, 1, 1, 128)``.

Reports the max absolute delta per array and the two-tier verdict the Stage 3
gate is written against: investigate at 1e-6 action / 1e-5 hidden, stop at
1e-5 / 1e-4. Observation arrays are checked for equality too — a nonzero delta
there would mean the replay did not actually replay, which would make every
other number meaningless.
"""

import argparse
import json
import pathlib
import sys

import numpy as np

INVESTIGATE = {"actions": 1e-6, "hidden": 1e-5}
STOP = {"actions": 1e-5, "hidden": 1e-4}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("baseline", type=pathlib.Path)
    ap.add_argument("candidate", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    ap.add_argument(
        "--label", default="", help="Name for this comparison in the printed table."
    )
    args = ap.parse_args()

    base = np.load(args.baseline)
    cand = np.load(args.candidate)

    only_base = sorted(set(base.files) - set(cand.files))
    only_cand = sorted(set(cand.files) - set(base.files))

    rows = []
    worst = {"actions": 0.0, "hidden": 0.0, "obs": 0.0}
    for key in sorted(set(base.files) & set(cand.files)):
        a, b = base[key], cand[key]
        if a.shape != b.shape:
            rows.append({"key": key, "shape_mismatch": [list(a.shape), list(b.shape)]})
            continue
        delta = float(np.max(np.abs(a.astype(np.float64) - b.astype(np.float64))))
        kind = key.rsplit("|", 1)[-1]
        bucket = kind if kind in worst else "obs"
        worst[bucket] = max(worst[bucket], delta)
        rows.append({"key": key, "kind": kind, "max_abs_delta": delta})

    verdict = "STRICT PASS (0.0)"
    if worst["obs"] != 0.0:
        verdict = "INVALID — replay did not reproduce the observation inputs"
    elif worst["actions"] >= STOP["actions"] or worst["hidden"] >= STOP["hidden"]:
        verdict = "STOP"
    elif worst["actions"] >= INVESTIGATE["actions"] or worst["hidden"] >= INVESTIGATE["hidden"]:
        verdict = "INVESTIGATE"
    elif worst["actions"] or worst["hidden"]:
        verdict = "PASS (nonzero but below investigate floor)"

    print(f"=== export comparison {args.label} ===")
    print(f"baseline : {args.baseline}")
    print(f"candidate: {args.candidate}")
    if only_base:
        print(f"arrays only in baseline : {only_base}")
    if only_cand:
        print(f"arrays only in candidate: {only_cand}")
    print(f"{'array':56s} {'max|delta|':>12s}")
    for r in rows:
        if "shape_mismatch" in r:
            print(f"{r['key']:56s} SHAPE MISMATCH {r['shape_mismatch']}")
        else:
            print(f"{r['key']:56s} {r['max_abs_delta']:12.3e}")
    print("-" * 70)
    print(f"worst obs delta     : {worst['obs']:.3e}  (must be 0.0)")
    print(f"worst action delta  : {worst['actions']:.3e}")
    print(f"worst hidden delta  : {worst['hidden']:.3e}")
    print(f"VERDICT: {verdict}")

    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "label": args.label,
                    "baseline": str(args.baseline),
                    "candidate": str(args.candidate),
                    "arrays_only_in_baseline": only_base,
                    "arrays_only_in_candidate": only_cand,
                    "rows": rows,
                    "worst": worst,
                    "verdict": verdict,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
