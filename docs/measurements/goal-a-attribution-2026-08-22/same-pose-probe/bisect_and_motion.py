#!/usr/bin/env python
"""Diagnostics: (1) node-capture motion onset within records 0-29 (scene-guard
confound); (2) tick-0-only scene guard; (3) bisection ticks: clean vs
corruption-noise-bearing sim depth through v2 ONNX."""
import json
import math

import numpy as np
import onnxruntime as ort

SP = "docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
NODE = "/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"
ONNX = "/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v2_998.onnx"
GOAL_BEARING = -8.1
DEPTH0 = 19
H, W = 45, 80


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def load_jsonl(path, skip_header=False, n=None):
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if skip_header and r.get("header"):
                continue
            recs.append(r)
            if n and len(recs) >= n:
                break
    return recs


def pctstats(delta):
    a = np.abs(np.asarray(delta, dtype=np.float64)).ravel()
    return {"p50": float(np.percentile(a, 50)), "p95": float(np.percentile(a, 95)),
            "max": float(a.max()), "mean": float(a.mean())}


node = load_jsonl(NODE, n=30)
gym = load_jsonl(f"{SP}/gym_obs.jsonl")
env = load_jsonl(f"{SP}/env_obsbuf.jsonl", skip_header=True)
node_m = np.array([r["obs"] for r in node], dtype=np.float64)
gym_m = np.array([r["obs"] for r in gym], dtype=np.float64)
env_m = np.array([r["obs"] for r in env], dtype=np.float64)

out = {}

# ---------------------------------------------------------------- motion onset
gd_mean = gym_m[:, DEPTH0:].mean(axis=0)
node_depth = node_m[:, DEPTH0:]
rows = []
for i in range(30):
    rows.append({
        "rec": i,
        "corr_vs_gym_mean": float(np.corrcoef(node_depth[i], gd_mean)[0, 1]),
        "corr_vs_node0": float(np.corrcoef(node_depth[i], node_depth[0])[0, 1]),
        "depth_mean_m": float(node_depth[i].mean() * 6),
        "gyro_z": float(node_m[i, 5]),
        "body_vel": node_m[i, 14:16].tolist(),
        "enc_absmax": float(np.abs(node_m[i, 6:10]).max()),
        "last_action": node_m[i, 16:19].tolist(),
    })
out["node_motion_onset"] = rows

# stationary-tick scene guard: node record 0 vs gym mean and per gym record
nd0 = node_depth[0]
d0 = nd0 - gd_mean
out["scene_guard_tick0"] = {
    "corr_node0_vs_gym_mean": float(np.corrcoef(nd0, gd_mean)[0, 1]),
    "corr_node0_vs_gym0": float(np.corrcoef(nd0, gym_m[0, DEPTH0:])[0, 1]),
    "diff_scaled": pctstats(d0),
    "diff_m": {k: v * 6 for k, v in pctstats(d0).items()},
    "signed_mean_offset_m": float(d0.mean() * 6),
    "node0_mean_m": float(nd0.mean() * 6),
    "gym_mean_m": float(gd_mean.mean() * 6),
}
# linear fit + edge structure for tick-0 depth diff
A = np.vstack([gd_mean, np.ones_like(gd_mean)]).T
(a_fit, b_fit), *_ = np.linalg.lstsq(A, nd0, rcond=None)
gy, gx = np.gradient(gd_mean.reshape(H, W))
gradmag = np.hypot(gx, gy).ravel()
out["scene_guard_tick0"]["linear_fit"] = {"scale_a": float(a_fit), "offset_b": float(b_fit)}
out["scene_guard_tick0"]["absdiff_vs_edge_corr"] = float(
    np.corrcoef(np.abs(d0), gradmag)[0, 1])
from scipy.ndimage import median_filter
med_g = median_filter(gd_mean.reshape(H, W), size=3).ravel()
out["scene_guard_tick0"]["corr_node0_vs_median3x3_gym"] = float(
    np.corrcoef(nd0, med_g)[0, 1])
out["scene_guard_tick0"]["mad_node0_vs_median3x3_gym_scaled"] = float(
    np.abs(nd0 - med_g).mean())
# tick0 block analysis
diff_img0 = d0.reshape(H, W)
blocks = []
for bi in range(0, H, 5):
    for bj in range(0, W, 5):
        blk = diff_img0[bi:bi + 5, bj:bj + 5]
        blocks.append({"rows": [bi, bi + 5], "cols": [bj, bj + 5],
                       "mean_abs_scaled": float(np.abs(blk).mean()),
                       "mean_signed_scaled": float(blk.mean())})
blocks.sort(key=lambda b: -b["mean_abs_scaled"])
out["scene_guard_tick0"]["top_blocks"] = blocks[:6]
# nearfield structure at tick0
near = gd_mean <= np.percentile(gd_mean, 10)
far = gd_mean >= np.percentile(gd_mean, 90)
out["scene_guard_tick0"]["nearfield_bottom10"] = pctstats(d0[near])
out["scene_guard_tick0"]["farfield_top10"] = pctstats(d0[far])

# ------------------------------------------------- clean vs noise-bearing diff
# gym clean vs env obsbuf (same instants, paired records)
d_depth_noise = env_m[:, DEPTH0:] - gym_m[:, DEPTH0:]
out["env_noise_vs_clean_depth"] = {
    "paired_scaled": pctstats(d_depth_noise),
    "paired_m": {k: v * 6 for k, v in pctstats(d_depth_noise).items()},
    "signed_mean_scaled": float(d_depth_noise.mean()),
    "corr_rec15": float(np.corrcoef(env_m[15, DEPTH0:], gym_m[15, DEPTH0:])[0, 1]),
    "note": "difference = realism-profile corruption noise the manager injects",
}
for name, a, b in [("imu_accel", 0, 3), ("imu_gyro", 3, 6), ("encoders", 6, 10),
                   ("quartet", 10, 14), ("body_vel", 14, 16), ("last_action", 16, 19)]:
    out["env_noise_vs_clean_depth"][f"small_{name}"] = pctstats(
        env_m[:, a:b] - np.nan_to_num(gym_m[:, a:b], nan=0.0) if name in ("quartet", "last_action")
        else env_m[:, a:b] - gym_m[:, a:b])

# ---------------------------------------------------------------- bisect ticks
sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])


def tick(obs):
    h0 = np.zeros((1, 1, 128), dtype=np.float32)
    a, _ = sess.run(["actions", "h_out"],
                    {"obs": obs.astype(np.float32).reshape(1, -1), "h_in": h0})
    return a.squeeze(0).astype(np.float64)


def analyze(name, obs, desc):
    act = tick(obs)
    dir_deg = math.degrees(math.atan2(act[1], act[0]))
    off = wrap_deg(dir_deg - GOAL_BEARING)
    return {"patch": name, "description": desc, "command": act.tolist(),
            "cmd_dir_body_deg": dir_deg, "off_goal_deg": off,
            "abs_off_goal_deg": abs(off),
            "flips_toward_goal": bool(abs(off) <= 45.0), "wz": float(act[2])}

node0 = node_m[0].copy()
gym15 = gym_m[15].copy()
ticks = []

# T1/T2: env obsbuf rows straight through (noise-bearing full sim obs)
for i in (0, 15):
    ticks.append(analyze(f"t_envbuf_rec{i}", env_m[i].copy(),
                         f"env ObservationManager row {i} (noise-bearing), unmodified"))

# T3: node obs with env noise-bearing depth
o = node0.copy(); o[DEPTH0:] = env_m[15, DEPTH0:]
ticks.append(analyze("t_node0_envnoise_depth", o,
                     "node tick-0 with depth <- env obsbuf rec 15 (noise-bearing sim depth)"))

# T4: clean all-sim cell (c) with env noise-bearing depth
o = gym15.copy(); o[10:14] = node0[10:14]; o[16:19] = node0[16:19]
o[DEPTH0:] = env_m[15, DEPTH0:]
ticks.append(analyze("t_cleancell_envnoise_depth", o,
                     "cell (c) gym15+node bookkeeping, depth <- env rec 15 noise-bearing"))

# T5: env obsbuf rec 15 with clean gym depth (inverse)
o = env_m[15].copy(); o[DEPTH0:] = gym15[DEPTH0:]
ticks.append(analyze("t_envbuf15_clean_depth", o,
                     "env obsbuf rec 15 with depth <- gym clean rec 15 (inverse)"))

# T6: env obsbuf rec 15 with node depth
o = env_m[15].copy(); o[DEPTH0:] = node0[DEPTH0:]
ticks.append(analyze("t_envbuf15_node_depth", o,
                     "env obsbuf rec 15 with depth <- node tick-0"))

# T7: node0 with clean gym depth AND env-noise small fields (complement check)
o = env_m[15].copy(); o[DEPTH0:] = env_m[15, DEPTH0:]  # identity guard (same as T2)

# T8: sweep all 30 env rows: how many drive toward goal
sweep = []
for i in range(30):
    r = analyze(f"sweep_env_{i}", env_m[i].copy(), "")
    sweep.append({"rec": i, "cmd": r["command"], "off": r["off_goal_deg"],
                  "flips": r["flips_toward_goal"]})
out["env_sweep_toward_goal_count"] = int(sum(s["flips"] for s in sweep))
out["env_sweep"] = sweep

# T9: sweep all 30 clean gym rows (with node bookkeeping fill)
sweep_c = []
for i in range(30):
    o = gym_m[i].copy(); o[10:14] = node0[10:14]; o[16:19] = node0[16:19]
    r = analyze(f"sweep_gymclean_{i}", o, "")
    sweep_c.append({"rec": i, "cmd": r["command"], "off": r["off_goal_deg"],
                    "flips": r["flips_toward_goal"]})
out["gymclean_sweep_toward_goal_count"] = int(sum(s["flips"] for s in sweep_c))
out["gymclean_sweep"] = sweep_c

out["bisect_ticks"] = ticks

with open(f"{SP}/bisect_and_motion_results.json", "w") as f:
    json.dump(out, f, indent=2)
print(json.dumps({k: v for k, v in out.items()
                  if k not in ("node_motion_onset", "env_sweep", "gymclean_sweep")},
                 indent=2))
print("--- motion onset (first 12 rows) ---")
for r in out["node_motion_onset"][:12]:
    print(r)
print("--- env sweep flips:", out["env_sweep_toward_goal_count"],
      "gymclean sweep flips:", out["gymclean_sweep_toward_goal_count"])
