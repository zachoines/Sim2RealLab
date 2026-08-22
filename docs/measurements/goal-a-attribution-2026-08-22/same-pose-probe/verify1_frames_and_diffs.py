#!/usr/bin/env python
"""Adversarial verifier, part 1: frame arithmetic, independent field-diff
reproduction, alignment attacks. Written from scratch (does not import the
diff agent's scripts)."""
import json, math
import numpy as np

SP = "docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
NODE = "/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"

# ---------------------------------------------------------------- CHECK 1: frames
spawn = np.array([0.35000020265579224, -1.75])       # probe spawn world xy, yaw 0
start_map = np.array([-0.499, -0.451]); start_yaw = 2.22
ref_map = np.array([-0.12990365596194792, 0.3428494265663703])
goal_map = np.array([-2.0, 2.25])

target_world = spawn + start_map
ref_world = spawn + ref_map
goal_world = spawn + goal_map

probe_target = np.array([-0.14899979734420776, -2.201])
probe_ref = np.array([0.22009654669384432, -1.4071505734336296])

ach = np.array([-0.14677724242210388, -2.1721723079681396]); ach_yaw = 2.2137026846438372

def body_rel(p, yaw, q):
    d = q - p
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([c*d[0] + s*d[1], -s*d[0] + c*d[1]])

rel = body_rel(ach, ach_yaw, ref_world)
dist = float(np.hypot(*rel)); head = math.atan2(rel[1], rel[0])
quartet_raw = [rel[0], rel[1], dist, head]
quartet_scaled = [rel[0]/10, rel[1]/10, dist/10, head/math.pi]
probe_quartet_raw = [0.3922194838523865, -0.7519649267196655, 0.8484424352645874, -1.0900686979293823]
node_tick0_scaled = None  # filled after load

# goal bearings
def bearing_body_deg(p, yaw, q):
    r = body_rel(p, yaw, q)
    return math.degrees(math.atan2(r[1], r[0]))

goal_bear_ach = bearing_body_deg(ach, ach_yaw, goal_world)          # expect ~ -7.5
goal_bear_nom = bearing_body_deg(target_world, 2.22, goal_world)    # expect ~ -8.1 (rig nominal)

def wrap(d): return (d + 180.0) % 360.0 - 180.0
def offgoal(cmd, bear):
    return wrap(math.degrees(math.atan2(cmd[1], cmd[0])) - bear)

v2_first = (0.4975191354751587, 0.016867876052856445)
v1_first = (-0.22986948490142822, 0.1973506212234497)
rig_first = (-0.0029388, -0.4190325)

frames = {
    "target_world_delta": np.abs(target_world - probe_target).max(),
    "ref_world_delta": np.abs(ref_world - probe_ref).max(),
    "goal_world": goal_world.tolist(),
    "quartet_raw_analytic": quartet_raw,
    "quartet_raw_probe_env": probe_quartet_raw,
    "quartet_raw_absdelta": [abs(a-b) for a, b in zip(quartet_raw, probe_quartet_raw)],
    "quartet_scaled_analytic": quartet_scaled,
    "goal_bearing_body_deg_achieved": goal_bear_ach,
    "goal_bearing_body_deg_nominal": goal_bear_nom,
    "v2_first_offgoal_vs_ach": offgoal(v2_first, goal_bear_ach),
    "v1_first_offgoal_vs_ach": offgoal(v1_first, goal_bear_ach),
    "rig_first_offgoal_vs_nom": offgoal(rig_first, -8.1),
    "rig_first_offgoal_vs_recomputed_nom": offgoal(rig_first, goal_bear_nom),
}

# ---------------------------------------------------------------- load data
def load(path, n=None, skip_header=False):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if skip_header and r.get("header"): continue
            out.append(r["obs"])
            if n and len(out) >= n: break
    return np.array(out, dtype=np.float64)

node = load(NODE, n=30)
gym = load(f"{SP}/gym_obs.jsonl")
env = load(f"{SP}/env_obsbuf.jsonl", skip_header=True)
assert node.shape == (30, 3619) and gym.shape == (30, 3619) and env.shape == (30, 3619)

D0 = 19; H, W = 45, 80
nodeD, gymD, envD = node[:, D0:], gym[:, D0:], env[:, D0:]

# expected node tick0 quartet (scaled) from task
node_q0 = node[0, 10:14]
frames["node_tick0_quartet_scaled"] = node_q0.tolist()
frames["node_tick0_vs_analytic_absdelta"] = np.abs(node_q0 - np.array(quartet_scaled)).tolist()
frames["task_expected_node_tick0"] = [0.040453, -0.077633, 0.087541, -0.347095]
frames["node_tick0_vs_task_expected_absdelta"] = np.abs(
    node_q0 - np.array(frames["task_expected_node_tick0"])).tolist()
env_q0 = env[0, 10:14]
frames["env_noisebearing_q0_vs_analytic"] = np.abs(env_q0 - np.array(quartet_scaled)).tolist()

# ---------------------------------------------------------------- CHECK 2: diffs
def pct(a):
    a = np.abs(np.asarray(a, float)).ravel()
    return dict(p50=float(np.percentile(a, 50)), p95=float(np.percentile(a, 95)),
                max=float(a.max()))

nd_mean, gd_mean = nodeD.mean(0), gymD.mean(0)
def corr(a, b): return float(np.corrcoef(a, b)[0, 1])

d0 = nodeD[0] - gd_mean
A = np.vstack([gd_mean, np.ones_like(gd_mean)]).T
(a_fit, b_fit), *_ = np.linalg.lstsq(A, nodeD[0], rcond=None)

diffs = {
    "scene_guard_pearson_mean30": corr(nd_mean, gd_mean),           # claim 0.8951
    "node0_vs_gym_mean": corr(nodeD[0], gd_mean),                   # claim 0.9829
    "node0_vs_gym0": corr(nodeD[0], gymD[0]),                       # claim 0.9779
    "tick0_signed_offset_m": float(d0.mean() * 6),                  # claim -2.1 mm
    "tick0_fit_scale": float(a_fit),                                # claim 0.969
    "tick0_p50_absdelta_m": float(np.percentile(np.abs(d0), 50) * 6),  # claim 2.6 mm
    "tick0_depth_scaled": pct(d0),                                  # claim p50 .00043 p95 .0194 max .612
    "tick0_depth_m": {k: v*6 for k, v in pct(d0).items()},
    "gym_depth_raw_range_m": [float(gymD.min()*6), float(gymD.max()*6), float(gymD.mean()*6)],
    "gym_depth_per_record_mean_std_scaled": float(gymD.mean(1).std()),
    "node0_mean_m": float(nodeD[0].mean()*6),
    "gym_mean_m": float(gd_mean.mean()*6),
}

# small fields: node tick0 vs gym settled mean (records 10-29)
gs = gym[10:].mean(0)
for name, a, b in [("imu_accel", 0, 3), ("imu_gyro", 3, 6),
                   ("encoders", 6, 10), ("body_velocity", 14, 16)]:
    diffs[f"tick0_{name}_absdelta"] = np.abs(node[0, a:b] - gs[a:b]).tolist()
diffs["gravity_az"] = {"node_mean30": float(node[:, 2].mean()),
                       "node_tick0": float(node[0, 2]),
                       "gym_settled": float(gs[2])}

# motion-onset facts in the node capture
diffs["node_rec1_last_action"] = node[1, 16:19].tolist()
diffs["node_rec0_last_action"] = node[0, 16:19].tolist()
diffs["node_enc_absmax_by_rec"] = [float(np.abs(node[i, 6:10]).max()) for i in range(30)]
diffs["node_gyroz_by_rec"] = [float(node[i, 5]) for i in range(6)] + ["..."]
diffs["node_depth_corr_vs_rec0"] = [corr(nodeD[i], nodeD[0]) for i in (1, 2, 7, 11, 29)]
diffs["node_depth_mean_m_by_rec"] = [float(nodeD[i].mean()*6) for i in (0, 10, 29)]

# corruption stats in env obsbuf depth
slam = envD == 1.0
diffs["env_frac_exactly_1"] = float(slam.mean())                    # claim 0.192
clean_at_slam = gymD[slam]
diffs["clean_p50_at_slammed"] = float(np.percentile(clean_at_slam, 50))  # claim 0.0333
rowfrac = slam.reshape(30, H, W).mean(axis=(0, 2))
diffs["rowslam_top5"] = rowfrac[:5].tolist()
diffs["rowslam_rows25_44_minmax"] = [float(rowfrac[25:].min()), float(rowfrac[25:].max())]
uns = ~slam
diffs["unslammed_corr"] = corr(envD[uns], gymD[uns])                # claim 0.997
diffs["unslammed_delta_std"] = float((envD[uns] - gymD[uns]).std())  # claim 0.021
# jaccard across consecutive records
m = slam.reshape(30, -1)
jac = [ (m[i] & m[i+1]).sum() / max(1, (m[i] | m[i+1]).sum()) for i in range(29)]
diffs["mask_jaccard_mean_consecutive"] = float(np.mean(jac))        # claim ~0.33

# NaN audit of the gym dump
nanmask = np.isnan(gym)
nan_cols = sorted(set(np.where(nanmask)[1].tolist()))
diffs["gym_nan_columns"] = nan_cols                                  # expect [10,11,12,13,16,17,18]
diffs["gym_nan_all_records"] = bool(nanmask[:, nan_cols].all())
diffs["node_nan_any"] = bool(np.isnan(node).any())
diffs["env_nan_any"] = bool(np.isnan(env).any())

# ---------------------------------------------------------------- CHECK 4: alignment
n0 = nodeD[0].reshape(H, W)
gm = gd_mean.reshape(H, W)
def c2(a, b): return corr(a.ravel(), b.ravel())
align = {
    "identity": c2(n0, gm),
    "fliplr": c2(np.fliplr(n0), gm),
    "flipud": c2(np.flipud(n0), gm),
    "flip_both": c2(np.flipud(np.fliplr(n0)), gm),
    "reshape_80x45_T": c2(nodeD[0].reshape(W, H).T, gm),
    "reshape_80x45_T_fliplr": c2(np.fliplr(nodeD[0].reshape(W, H).T), gm),
    "reshape_80x45_T_flipud": c2(np.flipud(nodeD[0].reshape(W, H).T), gm),
}
align["argmax"] = max(align, key=lambda k: align[k] if k != "argmax" else -9)
# gravity direction sanity: bottom rows should be nearer (floor) in both
align["node_bottom5_mean_m"] = float(n0[-5:].mean()*6)
align["node_top5_mean_m"] = float(n0[:5].mean()*6)
align["gym_bottom5_mean_m"] = float(gm[-5:].mean()*6)
align["gym_top5_mean_m"] = float(gm[:5].mean()*6)
# scale conventions: heading raw from scaled (1/pi) vs manifest -62.5 deg first
align["node_tick0_heading_deg_from_1overpi"] = float(node[0, 13] * 180.0)
align["node_tick0_subgoal_dist_m_from_1over10"] = float(node[0, 12] * 10)

out = {"frames": frames, "diffs": diffs, "align": align}
print(json.dumps(out, indent=2, default=float))
with open("docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe/verify1_out.json", "w") as f:
    json.dump(out, f, indent=2, default=float)
