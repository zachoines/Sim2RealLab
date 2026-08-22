#!/usr/bin/env python3
"""Post-run analysis of the same-pose probe outputs (plain python, no Kit)."""
import json
import math
import statistics as st

OUT = "docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
NODE = "/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"

DEPTH_LO = 19
DEPTH_HI = 3619  # exclusive

meta = json.load(open(f"{OUT}/meta.json"))
cl = json.load(open(f"{OUT}/closed_loop_summary.json"))

gym_recs = [json.loads(l) for l in open(f"{OUT}/gym_obs.jsonl")]
node_first = json.loads(open(NODE).readline())

# Depth summary over all gym records
mins, maxs, means, sat_hi, sat_lo = [], [], [], [], []
for r in gym_recs:
    d = r["obs"][DEPTH_LO:DEPTH_HI]
    mins.append(min(d)); maxs.append(max(d)); means.append(sum(d) / len(d))
    sat_hi.append(sum(1 for v in d if v >= 0.999) / len(d))
    sat_lo.append(sum(1 for v in d if v <= 0.001) / len(d))

depth_summary = {
    "records": len(gym_recs),
    "scale": "scaled 1/6 (raw meters = value*6, DEPTH_MAX=6.0)",
    "scaled_min": min(mins), "scaled_max": max(maxs),
    "scaled_mean_over_records": sum(means) / len(means),
    "frac_saturated_at_max": sum(sat_hi) / len(sat_hi),
    "frac_at_zero": sum(sat_lo) / len(sat_lo),
    "mean_stability_std_across_records": st.pstdev(means),
}

# Field-level compare: gym record 1 vs node record 1 (same nominal pose;
# node record is mid-session, robot within ~mm of the placed pose)
g = gym_recs[0]["obs"]; n = node_first["obs"]
gd = g[DEPTH_LO:DEPTH_HI]; nd = n[DEPTH_LO:DEPTH_HI]
diff = [a - b for a, b in zip(gd, nd)]
mad = sum(abs(x) for x in diff) / len(diff)
mg = sum(gd) / len(gd); mn = sum(nd) / len(nd)
cov = sum((a - mg) * (b - mn) for a, b in zip(gd, nd))
den = math.sqrt(sum((a - mg) ** 2 for a in gd) * sum((b - mn) ** 2 for b in nd))
corr = cov / den if den > 0 else float("nan")

first_rec_compare = {
    "note": "coarse S2 color only — full field diff is the parent's replay job",
    "depth_mean_abs_diff_scaled": mad,
    "depth_mean_abs_diff_m": mad * 6.0,
    "depth_pearson_corr": corr,
    "gym_depth_mean_m": mg * 6.0, "node_depth_mean_m": mn * 6.0,
    "imu_accel_gym": g[0:3], "imu_accel_node": n[0:3],
    "imu_gyro_gym": g[3:6], "imu_gyro_node": n[3:6],
    "encoders_gym": g[6:10], "encoders_node": n[6:10],
    "body_vel_gym": g[14:16], "body_vel_node": n[14:16],
    "node_quartet_scaled": n[10:14],
}

# Env quartet vs node quartet (raw units)
q = meta["quartet_clean_terms"]
env_quartet_raw = (q["goal_position_relative_raw_m"]
                   + q["goal_distance_raw_m"]
                   + q["goal_heading_to_goal_raw_rad"])
node_quartet_raw = [n[10] * 10, n[11] * 10, n[12] * 10, n[13] * math.pi]

out = {
    "depth_summary": depth_summary,
    "first_record_compare": first_rec_compare,
    "env_quartet_raw": env_quartet_raw,
    "node_first_quartet_raw": node_quartet_raw,
    "closed_loop": cl,
}
json.dump(out, open(f"{OUT}/analysis.json", "w"), indent=2)
print(json.dumps(out, indent=2))
