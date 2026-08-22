#!/usr/bin/env python
"""CPU half of the same-pose probe: field-diff gym vs node obs at the identical
pose, then patch replays through v2 ONNX (CPU EP) to isolate the causal field.

Pre-registered bands (scaled units): depth p95>0.01 OR systematic pattern;
imu_accel +-0.005; gyro +-0.005; encoders +-0.01; body_velocity +-0.005;
quartet +-0.01 vs analytic-from-pose.
"""
import json
import math

import numpy as np
import onnxruntime as ort

SP = "docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
NODE = "/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"
GYM = f"{SP}/gym_obs.jsonl"
ENVBUF = f"{SP}/env_obsbuf.jsonl"
ONNX = "/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v2_998.onnx"

# probe-report achieved pose + injected referent (world/map frame)
POSE = (-0.14677724242210388, -2.1721723079681396, 2.2137026846438372)
REFERENT = (0.22009654669384432, -1.4071505734336296)
GOAL_BEARING_BODY_DEG = -8.1  # rig nominal body-frame goal bearing
RIG_T0_CMD = (-0.0029, -0.4190, -0.6397)

H, W = 45, 80
DEPTH0 = 19


def wrap_deg(d):
    return (d + 180.0) % 360.0 - 180.0


def load_node_first(n=30):
    recs = []
    with open(NODE) as f:
        for line in f:
            recs.append(json.loads(line))
            if len(recs) >= n:
                break
    return recs


def load_jsonl(path, skip_header=False):
    recs = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if skip_header and r.get("header"):
                continue
            recs.append(r)
    return recs


def pctstats(delta):
    a = np.abs(np.asarray(delta, dtype=np.float64)).ravel()
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "mean": float(a.mean())}


def main():
    node = load_node_first(30)
    gym = load_jsonl(GYM)
    env = load_jsonl(ENVBUF, skip_header=True)
    assert len(gym) == 30 and len(env) == 30, (len(gym), len(env))

    node_m = np.array([r["obs"] for r in node], dtype=np.float64)
    gym_m = np.array([r["obs"] for r in gym], dtype=np.float64)
    env_m = np.array([r["obs"] for r in env], dtype=np.float64)
    assert node_m.shape == (30, 3619) and gym_m.shape == (30, 3619)

    out = {}

    # ------------------------------------------------------------------ 1
    # SCENE GUARD: mean depth images, per-pixel diff + Pearson over 3600 px
    node_depth = node_m[:, DEPTH0:]            # scaled 1/6
    gym_depth = gym_m[:, DEPTH0:]
    nd_mean = node_depth.mean(axis=0)
    gd_mean = gym_depth.mean(axis=0)
    diff_px = nd_mean - gd_mean                # node - gym, scaled
    corr = float(np.corrcoef(nd_mean, gd_mean)[0, 1])

    nd_img = nd_mean.reshape(H, W)
    gd_img = gd_mean.reshape(H, W)
    diff_img = diff_px.reshape(H, W)

    # block analysis: 5x5 blocks (9 x 16 grid)
    blocks = []
    for bi in range(0, H, 5):
        for bj in range(0, W, 5):
            blk = diff_img[bi:bi + 5, bj:bj + 5]
            blocks.append({
                "rows": [bi, bi + 5], "cols": [bj, bj + 5],
                "mean_abs_scaled": float(np.abs(blk).mean()),
                "mean_signed_scaled": float(blk.mean()),
                "gym_mean_depth_m": float(gd_img[bi:bi + 5, bj:bj + 5].mean() * 6),
            })
    blocks.sort(key=lambda b: -b["mean_abs_scaled"])

    scene_guard = {
        "node_records": "0-29 mean", "gym_records": "all 30 mean",
        "pearson_corr_3600px": corr,
        "same_geometry": bool(corr >= 0.9),
        "per_pixel_diff_scaled": pctstats(diff_px),
        "per_pixel_diff_m": {k: v * 6 for k, v in pctstats(diff_px).items()},
        "mean_depth_m": {"node": float(nd_mean.mean() * 6),
                         "gym": float(gd_mean.mean() * 6)},
        "top_disagreeing_blocks_5x5": blocks[:8],
    }
    out["scene_guard"] = scene_guard

    # ------------------------------------------------------------------ 2
    # FIELD DIFFS (paired by record index; imu verdict on settled 10-29)
    fields = {
        "imu_accel": (0, 3, 0.005),
        "imu_gyro": (3, 6, 0.005),
        "encoders": (6, 10, 0.01),
        "body_velocity": (14, 16, 0.005),
    }
    field_diffs = {}
    for name, (a, b, band) in fields.items():
        d_all = node_m[:, a:b] - gym_m[:, a:b]
        d_settled = node_m[10:, a:b] - gym_m[10:, a:b]
        st_all, st_set = pctstats(d_all), pctstats(d_settled)
        verdict_window = st_set if name == "imu_accel" else st_all
        field_diffs[name] = {
            "band": band,
            "all30": st_all,
            "settled_10_29": st_set,
            "node_mean": node_m[:, a:b].mean(axis=0).tolist(),
            "gym_mean": gym_m[:, a:b].mean(axis=0).tolist(),
            "gym_settled_mean": gym_m[10:, a:b].mean(axis=0).tolist(),
            "within_band": bool(verdict_window["p95"] <= band),
            "verdict_basis": ("settled records 10-29 (probe caveat: gym IMU "
                              "settle transient in early records)"
                              if name == "imu_accel" else "all 30 records"),
        }
    # gravity check
    field_diffs["imu_accel"]["gravity_az"] = {
        "node_mean": float(node_m[:, 2].mean()),
        "gym_settled_mean": float(gym_m[10:, 2].mean()),
        "expected": 0.063,
    }

    # depth field diff: paired records, all 30 x 3600
    d_depth = node_depth - gym_depth
    st = pctstats(d_depth)
    # structure: offset & scale via least squares node ~ a*gym + b on mean imgs
    A = np.vstack([gd_mean, np.ones_like(gd_mean)]).T
    (a_fit, b_fit), res, _, _ = np.linalg.lstsq(A, nd_mean, rcond=None)
    # edge alignment: |diff| vs gradient magnitude of gym mean image
    gy, gx = np.gradient(gd_img)
    gradmag = np.hypot(gx, gy).ravel()
    edge_corr = float(np.corrcoef(np.abs(diff_px), gradmag)[0, 1])
    # median-filter signature: 3x3 median of gym mean image
    from scipy.ndimage import median_filter  # may not exist; fallback below
    med_g = median_filter(gd_img, size=3).ravel()
    corr_med = float(np.corrcoef(nd_mean, med_g)[0, 1])
    mad_med = float(np.abs(nd_mean - med_g).mean())
    # nearfield: diffs among nearest 10% pixels vs farthest
    near_mask = gd_mean <= np.percentile(gd_mean, 10)
    far_mask = gd_mean >= np.percentile(gd_mean, 90)
    field_diffs["depth"] = {
        "band": "p95>0.01 OR systematic pattern => MATERIAL (quant floor 1.7e-4)",
        "paired_30rec_scaled": st,
        "paired_30rec_m": {k: v * 6 for k, v in st.items()},
        "mean_img_scaled": pctstats(diff_px),
        "signed_mean_offset_scaled": float(diff_px.mean()),
        "signed_mean_offset_m": float(diff_px.mean() * 6),
        "linear_fit_node_vs_gym": {"scale_a": float(a_fit), "offset_b": float(b_fit)},
        "abs_diff_vs_edge_gradient_corr": edge_corr,
        "median3x3_of_gym": {"corr_node_vs_median_gym": corr_med,
                             "mad_node_vs_median_gym_scaled": mad_med,
                             "corr_node_vs_raw_gym": corr},
        "nearfield_bottom10pct": pctstats(diff_px[near_mask]),
        "farfield_top10pct": pctstats(diff_px[far_mask]),
        "material": bool(st["p95"] > 0.01),
        "temporal_stability": {
            "node_per_record_mean_std": float(node_depth.mean(axis=1).std()),
            "gym_per_record_mean_std": float(gym_depth.mean(axis=1).std()),
        },
    }

    # quartet analytic from achieved pose + injected referent
    px, py, yaw = POSE
    dx, dy = REFERENT[0] - px, REFERENT[1] - py
    c, s = math.cos(yaw), math.sin(yaw)
    bx, by = c * dx + s * dy, -s * dx + c * dy
    dist = math.hypot(bx, by)
    head = math.atan2(by, bx)
    analytic_scaled = np.array([bx / 10, by / 10, dist / 10, head / math.pi])
    node_q0 = node_m[0, 10:14]
    # node referent is the injected one through record ~10 (first change t=361.2)
    node_q_early = node_m[:11, 10:14].mean(axis=0)
    env_q0 = env_m[0, 10:14]
    env_q_mean = env_m[:, 10:14].mean(axis=0)
    quartet = {
        "band": 0.01,
        "analytic_raw": [bx, by, dist, head],
        "analytic_scaled": analytic_scaled.tolist(),
        "node_tick0_scaled": node_q0.tolist(),
        "node_tick0_abs_delta": np.abs(node_q0 - analytic_scaled).tolist(),
        "node_tick0_within_band": bool(np.abs(node_q0 - analytic_scaled).max() <= 0.01),
        "node_early_mean_records0_10": node_q_early.tolist(),
        "env_clean_raw_probe_report": [0.3922194838523865, -0.7519649267196655,
                                       0.8484424352645874, -1.0900686979293823],
        "env_clean_abs_delta_scaled": [
            abs(0.3922194838523865 / 10 - analytic_scaled[0]),
            abs(-0.7519649267196655 / 10 - analytic_scaled[1]),
            abs(0.8484424352645874 / 10 - analytic_scaled[2]),
            abs(-1.0900686979293823 / math.pi - analytic_scaled[3])],
        "env_clean_within_band": True,  # recomputed below
        "env_obsbuf_tick0_scaled_noisebearing": env_q0.tolist(),
        "env_obsbuf_tick0_abs_delta": np.abs(env_q0 - analytic_scaled).tolist(),
        "env_obsbuf_mean30_abs_delta": np.abs(env_q_mean - analytic_scaled).tolist(),
    }
    quartet["env_clean_within_band"] = bool(
        max(quartet["env_clean_abs_delta_scaled"]) <= 0.01)
    quartet["env_obsbuf_within_band"] = bool(
        max(quartet["env_obsbuf_tick0_abs_delta"]) <= 0.01)
    field_diffs["quartet_analytic"] = quartet

    out["field_diffs"] = field_diffs

    # ------------------------------------------------------------------ 3
    # PATCH REPLAYS: v2 ONNX, CPU EP, zero hidden, single tick each
    sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])

    def tick(obs):
        h0 = np.zeros((1, 1, 128), dtype=np.float32)
        a, _ = sess.run(["actions", "h_out"],
                        {"obs": obs.astype(np.float32).reshape(1, -1), "h_in": h0})
        return a.squeeze(0).astype(np.float64)

    node0 = node_m[0].copy()
    gym15 = gym_m[15].copy()
    gym_mean_depth = gym_depth.mean(axis=0)

    def analyze(name, obs, desc):
        act = tick(obs)
        dir_deg = math.degrees(math.atan2(act[1], act[0]))
        off = wrap_deg(dir_deg - GOAL_BEARING_BODY_DEG)
        return {
            "patch": name, "description": desc,
            "command": act.tolist(),
            "cmd_dir_body_deg": dir_deg,
            "off_goal_deg": off,
            "abs_off_goal_deg": abs(off),
            "flips_toward_goal": bool(abs(off) <= 45.0),
            "wz": act[2],
        }

    replays = []

    # (a) harness check
    r = analyze("a_node0_unmodified", node0,
                "node tick-0 obs, unmodified (harness check vs rig cmd)")
    r["rig_ground_truth"] = list(RIG_T0_CMD)
    r["harness_max_abs_dev"] = float(
        np.abs(np.array(r["command"]) - np.array(RIG_T0_CMD)).max())
    replays.append(r)

    # (b) depth swap into node obs
    o = node0.copy(); o[DEPTH0:] = gym_mean_depth
    replays.append(analyze("b_node0_gym_mean_depth", o,
                           "node tick-0 with depth 19:3619 <- gym mean depth"))

    # (c) all-sim cell
    o = gym15.copy()
    o[10:14] = node0[10:14]
    o[16:19] = node0[16:19]
    assert not np.isnan(o).any()
    replays.append(analyze("c_gym15_node_bookkeeping", o,
                           "gym record 15 with NaN dims (quartet 10-13, "
                           "last_action 16-18) <- node tick-0"))

    # (d) per-group swaps into node obs (gym record-15 values; quartet uses
    # the analytic clean values since gym's quartet is NaN by design)
    groups = [("imu_accel", 0, 3, gym15[0:3]),
              ("imu_gyro", 3, 6, gym15[3:6]),
              ("encoders", 6, 10, gym15[6:10]),
              ("quartet", 10, 14, analytic_scaled),
              ("body_vel", 14, 16, gym15[14:16])]
    for gname, a, b, vals in groups:
        o = node0.copy(); o[a:b] = vals
        replays.append(analyze(f"d_node0_swap_{gname}", o,
                               f"node tick-0 with {gname} [{a}:{b}] <- gym/clean values"))

    # (e) inverse of whichever single patch flips toward goal
    flippers = [r for r in replays
                if r["patch"].startswith(("b_", "d_")) and r["flips_toward_goal"]]
    for fl in flippers:
        if fl["patch"] == "b_node0_gym_mean_depth":
            o = gym15.copy()
            o[10:14] = node0[10:14]; o[16:19] = node0[16:19]
            o[DEPTH0:] = node0[DEPTH0:]
            replays.append(analyze("e_inverse_gymcell_node_depth", o,
                                   "INVERSE: all-sim cell (c) with node tick-0 depth"))
        else:
            gname = fl["patch"].replace("d_node0_swap_", "")
            a, b = dict(imu_accel=(0, 3), imu_gyro=(3, 6), encoders=(6, 10),
                        quartet=(10, 14), body_vel=(14, 16))[gname]
            o = gym15.copy()
            o[10:14] = node0[10:14]; o[16:19] = node0[16:19]
            o[a:b] = node0[a:b]
            replays.append(analyze(f"e_inverse_gymcell_node_{gname}", o,
                                   f"INVERSE: all-sim cell (c) with node {gname}"))

    out["patch_replays"] = replays
    out["goal_bearing_body_deg"] = GOAL_BEARING_BODY_DEG
    out["ort_version"] = ort.__version__

    with open(f"{SP}/field_diff_and_patch_results.json", "w") as f:
        json.dump(out, f, indent=2)
    np.save(f"{SP}/node_depth_mean.npy", nd_mean)
    np.save(f"{SP}/gym_depth_mean.npy", gd_mean)
    np.save(f"{SP}/depth_diff_mean_img.npy", diff_img)
    print(json.dumps(out, indent=2, default=float))


if __name__ == "__main__":
    main()
