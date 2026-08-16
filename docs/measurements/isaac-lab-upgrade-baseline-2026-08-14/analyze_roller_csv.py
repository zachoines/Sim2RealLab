#!/usr/bin/env python3
"""Recompute the roller-probe metrics from its CSV, and compare runs.

The probe prints a metric table, but Isaac Sim tears the process down with
``os._exit`` and buffered stdout does not survive it, so the table is not a
reliable artifact. The CSV is. This recomputes the same quantities with the
same window definitions the probe uses (``roller_bounce_probe.py``: early =
first quarter of the spin, late = second half, growth = late/early p2p,
dominant frequency = largest rFFT magnitude at >= 3 Hz), so the numbers are
reproducible from committed data rather than from a console scrollback.

It also diffs runs pairwise, which is the determinism-floor (D0) question:
identical CSVs mean physics gates after the bump can be hash gates, and any
spread is the noise floor every band must clear.

Usage:

    python3 docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/analyze_roller_csv.py \
      docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/physics/roller_z_*.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import pathlib

import numpy as np

SIM_DT = 1.0 / 120.0


def load(path: pathlib.Path) -> dict[float, np.ndarray]:
    """frac -> (n, 2) array of (z, vz), in file order."""
    per: dict[float, list[tuple[float, float]]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            per.setdefault(float(row["omega_frac"]), []).append(
                (float(row["z"]), float(row["vz"]))
            )
    return {k: np.asarray(v, dtype=np.float64) for k, v in per.items()}


def metrics(z: np.ndarray) -> dict:
    n = len(z)
    q, h = n // 4, n // 2
    early = float(z[:q].max() - z[:q].min())
    late = float(z[h:].max() - z[h:].min())
    freqs = np.fft.rfftfreq(n, d=SIM_DT)
    mag = np.abs(np.fft.rfft(z - z.mean()))
    band = freqs >= 3.0
    peak = float(freqs[band][int(np.argmax(mag[band]))]) if band.any() else 0.0
    return {
        "early_p2p_mm": round(early * 1000, 4),
        "late_p2p_mm": round(late * 1000, 4),
        "growth": round(late / early, 4) if early > 1e-9 else None,
        "peak_f_hz": round(peak, 3),
        "mean_z_mm": round(float(z.mean()) * 1000, 4),
        "max_z_mm": round(float(z.max()) * 1000, 4),
        "min_z_mm": round(float(z.min()) * 1000, 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+", type=pathlib.Path)
    ap.add_argument("--out", type=pathlib.Path, default=None)
    args = ap.parse_args()

    report: dict = {"runs": {}, "pairwise": {}}
    loaded: dict[str, dict[float, np.ndarray]] = {}

    for path in args.csvs:
        data = load(path)
        loaded[path.name] = data
        report["runs"][path.name] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": sum(len(v) for v in data.values()),
            "per_frac": {str(k): metrics(v[:, 0]) for k, v in sorted(data.items())},
        }
        print(f"\n=== {path.name} ===")
        print(f"{'frac':>6} {'late_p2p_mm':>12} {'growth':>8} {'peak_f_hz':>10} {'mean_z_mm':>10}")
        per_frac = report["runs"][path.name]["per_frac"]
        for frac, m in sorted(per_frac.items(), key=lambda kv: float(kv[0])):
            print(
                f"{frac:>6} {m['late_p2p_mm']:>12.3f} {str(m['growth']):>8} "
                f"{m['peak_f_hz']:>10.2f} {m['mean_z_mm']:>10.3f}"
            )

    for a, b in itertools.combinations(sorted(loaded), 2):
        da, db = loaded[a], loaded[b]
        if set(da) != set(db):
            report["pairwise"][f"{a} vs {b}"] = {"comparable": False}
            continue
        dz = max(float(np.max(np.abs(da[k][:, 0] - db[k][:, 0]))) for k in da)
        dvz = max(float(np.max(np.abs(da[k][:, 1] - db[k][:, 1]))) for k in da)
        identical = dz == 0.0 and dvz == 0.0
        report["pairwise"][f"{a} vs {b}"] = {
            "comparable": True,
            "bit_identical": identical,
            "max_abs_dz_m": dz,
            "max_abs_dvz_m_per_s": dvz,
        }
        print(
            f"{a} vs {b}: "
            + (
                "BIT-IDENTICAL"
                if identical
                else f"max |dz| = {dz:.3e} m, max |dvz| = {dvz:.3e} m/s"
            )
        )

    if args.out:
        args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
