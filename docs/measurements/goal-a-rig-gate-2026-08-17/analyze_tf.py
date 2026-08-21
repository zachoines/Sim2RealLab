#!/usr/bin/env python3
"""Ride-along analysis: map->odom correction statistics + drift autocorrelation.

map->odom is piecewise constant: it holds steady between SLAM corrections and
steps when RTAB-Map revises the estimate. So the interesting quantity is not the
sample-to-sample noise (there is none) but the STEP events -- their magnitude
distribution and their rate. Autocorrelation is computed on the correction
series to pin a timescale tau.
"""
import json
import math
import sys

import numpy as np

path = sys.argv[1] if len(sys.argv) > 1 else "tf_riggate2.jsonl"
rows = []
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue

if len(rows) < 10:
    print(json.dumps({"error": "too few samples", "n": len(rows)}))
    sys.exit(0)

wall = np.array([r["wall"] for r in rows])
sim = np.array([r["sim"] for r in rows])
x = np.array([r["x"] for r in rows])
y = np.array([r["y"] for r in rows])
yaw = np.unwrap(np.array([r["yaw"] for r in rows]))

dx = np.diff(x)
dy = np.diff(y)
dyaw = np.diff(yaw)
dtrans = np.hypot(dx, dy)
dt_wall = np.diff(wall)
dt_sim = np.diff(sim)

# A "jump" = a correction event. Threshold well above float noise but below any
# meaningful map revision, so events are counted, not smoothed.
THRESH_M = 1e-4
THRESH_RAD = 1e-4
jmask = (dtrans > THRESH_M) | (np.abs(dyaw) > THRESH_RAD)
jump_trans = dtrans[jmask]
jump_yaw = np.abs(dyaw[jmask])
jump_wall = wall[1:][jmask]
jump_sim = sim[1:][jmask]

span_wall = wall[-1] - wall[0]
span_sim = sim[-1] - sim[0]


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


out = {
    "samples": len(rows),
    "span_wall_s": round(float(span_wall), 1),
    "span_sim_s": round(float(span_sim), 1),
    "mean_sample_dt_wall_s": round(float(np.mean(dt_wall)), 4),
    "effective_RTF": round(float(span_sim / span_wall), 4) if span_wall > 0 else None,
    "total_map_odom_travel_m": round(float(np.sum(dtrans)), 4),
    "net_map_odom_shift_m": round(
        float(math.hypot(x[-1] - x[0], y[-1] - y[0])), 4
    ),
    "net_map_odom_yaw_deg": round(float(math.degrees(yaw[-1] - yaw[0])), 3),
    "correction_events": int(jmask.sum()),
    "correction_rate_per_min_wall": (
        round(float(jmask.sum() / (span_wall / 60.0)), 2) if span_wall > 0 else None
    ),
    "correction_rate_per_min_sim": (
        round(float(jmask.sum() / (span_sim / 60.0)), 2) if span_sim > 0 else None
    ),
}

if len(jump_trans):
    out["jump_trans_m"] = {
        "min": round(float(jump_trans.min()), 5),
        "p50": round(pct(jump_trans, 50), 5),
        "p95": round(pct(jump_trans, 95), 5),
        "max": round(float(jump_trans.max()), 5),
        "mean": round(float(jump_trans.mean()), 5),
    }
    out["jump_yaw_deg"] = {
        "p50": round(math.degrees(pct(jump_yaw, 50)), 4),
        "p95": round(math.degrees(pct(jump_yaw, 95)), 4),
        "max": round(math.degrees(float(jump_yaw.max())), 4),
    }
    gaps_sim = np.diff(jump_sim)
    if len(gaps_sim):
        out["inter_correction_gap_sim_s"] = {
            "p50": round(pct(gaps_sim, 50), 3),
            "p95": round(pct(gaps_sim, 95), 3),
            "max": round(float(gaps_sim.max()), 3),
        }

# Autocorrelation of the correction series, resampled onto a uniform sim-time
# grid so tau is expressed in the robot's own experienced time.
if span_sim > 1.0:
    grid_dt = 0.5
    n = int(span_sim / grid_dt)
    if n > 20:
        tgrid = sim[0] + np.arange(n) * grid_dt
        sx = np.interp(tgrid, sim, x)
        sy = np.interp(tgrid, sim, y)
        sig = np.hypot(sx - sx.mean(), sy - sy.mean())
        sig = sig - sig.mean()
        if np.std(sig) > 1e-12:
            ac = np.correlate(sig, sig, mode="full")[len(sig) - 1:]
            ac = ac / ac[0]
            tau_idx = int(np.argmax(ac < 1.0 / math.e)) if np.any(ac < 1.0 / math.e) else -1
            out["drift_autocorr"] = {
                "grid_dt_sim_s": grid_dt,
                "tau_1_over_e_sim_s": (
                    round(tau_idx * grid_dt, 2) if tau_idx > 0 else None
                ),
                "ac_lag_1": round(float(ac[1]), 4) if len(ac) > 1 else None,
                "ac_lag_10": round(float(ac[10]), 4) if len(ac) > 10 else None,
                "ac_lag_60": round(float(ac[60]), 4) if len(ac) > 60 else None,
            }
        else:
            out["drift_autocorr"] = {"note": "map->odom constant; no variance to correlate"}

print(json.dumps(out, indent=2))
