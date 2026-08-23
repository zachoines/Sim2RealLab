#!/usr/bin/env python
"""Arm 3 offline CPU replay: node-captured obs -> ONNX policy actions.

Open-loop teacher forcing: obs embed the TRT run's last_action + trajectory.
Hidden state zero-init at record 0 (mission start), rolled across ticks —
mirrors _RecurrentOnnxPolicy (strafer_shared/policy_interface.py) and the
mission-boundary reset in inference_node.py:906.
"""
import json
import math
import sys

import numpy as np
import onnxruntime as ort

CAP = "/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"
OUT_DIR = "docs/measurements/goal-a-attribution-2026-08-22/replay"
V2 = "/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v2_998.onnx"
V1 = "/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v1.onnx"

START_POSE = (-0.499, -0.451, 2.22)
GOAL = (-2.0, 2.25)


def load_records():
    recs = []
    with open(CAP) as f:
        for line in f:
            recs.append(json.loads(line))
    return recs


def replay(model_path, obs_mat, out_path, t_sims):
    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    in_names = sorted(i.name for i in sess.get_inputs())
    assert in_names == ["h_in", "obs"], in_names
    h_shape = [d if isinstance(d, int) and d > 0 else 1
               for d in next(i for i in sess.get_inputs() if i.name == "h_in").shape]
    h = np.zeros(h_shape, dtype=np.float32)
    actions = np.zeros((len(obs_mat), 3), dtype=np.float64)
    with open(out_path, "w") as f:
        for t, obs in enumerate(obs_mat):
            a, h = sess.run(["actions", "h_out"],
                            {"obs": obs.reshape(1, -1), "h_in": h})
            h = h.astype(np.float32, copy=False)
            act = a.squeeze(0).astype(np.float64)
            actions[t] = act
            f.write(json.dumps({"t_sim": t_sims[t], "action": act.tolist()}) + "\n")
    return actions


def ang_deg(x):
    return math.degrees(math.atan2(math.sin(x), math.cos(x)))


def sign_changes(v, eps=0.0):
    s = np.sign(v)
    s = s[s != 0]
    return int(np.sum(s[1:] != s[:-1]))


def main():
    recs = load_records()
    n = len(recs)
    t_sims = np.array([r["t_sim"] for r in recs])
    obs_mat = np.array([r["obs"] for r in recs], dtype=np.float32)
    assert obs_mat.shape == (n, 3619), obs_mat.shape
    assert all(r["variant"] == "DEPTH_SUBGOAL" for r in recs)
    # record 0 sanity: last_action zeros
    print("record0 last_action:", obs_mat[0, 16:19].tolist(), file=sys.stderr)

    t0 = t_sims[0]
    first8 = t_sims <= t0 + 8.0

    # ---- v2 replay -------------------------------------------------------
    act_v2 = replay(V2, obs_mat, f"{OUT_DIR}/actions_v2_cpu.jsonl", t_sims)

    # ---- goal bearing in body frame at start -----------------------------
    dx, dy = GOAL[0] - START_POSE[0], GOAL[1] - START_POSE[1]
    bearing_map = math.atan2(dy, dx)
    bearing_body = ang_deg(bearing_map - START_POSE[2])  # deg

    mean_v_first8 = act_v2[first8, :2].mean(axis=0)
    cmd_dir_first8 = math.degrees(math.atan2(mean_v_first8[1], mean_v_first8[0]))
    off_goal = ang_deg(math.radians(cmd_dir_first8 - bearing_body))

    sig = {
        "vx_mean_all": float(act_v2[:, 0].mean()),
        "vy_mean_all": float(act_v2[:, 1].mean()),
        "wz_mean_all": float(act_v2[:, 2].mean()),
        "vx_mean_first8": float(act_v2[first8, 0].mean()),
        "vy_mean_first8": float(act_v2[first8, 1].mean()),
        "vy_duty_positive_all": float((act_v2[:, 1] > 0).mean()),
        "vy_duty_positive_first8": float((act_v2[first8, 1] > 0).mean()),
        "vy_sign_changes_first8": sign_changes(act_v2[first8, 1]),
        "goal_bearing_body_deg": bearing_body,
        "cmd_dir_body_first8_deg": cmd_dir_first8,
        "cmd_off_goal_first8_deg": off_goal,
    }

    # ---- agreement vs ground truth (record t+1 fields 16:19) -------------
    gt = obs_mat[1:, 16:19].astype(np.float64)   # action actually taken at tick t
    pred = act_v2[:-1]
    diff = np.abs(pred - gt)
    agree = {
        "mean_abs_diff": diff.mean(axis=0).tolist(),
        "max_abs_diff": diff.max(axis=0).tolist(),
        "corr": [float(np.corrcoef(pred[:, i], gt[:, i])[0, 1]) for i in range(3)],
        "gt_vx_mean": float(gt[:, 0].mean()),
        "gt_vy_mean": float(gt[:, 1].mean()),
        "gt_wz_mean": float(gt[:, 2].mean()),
        "gt_vy_duty_positive": float((gt[:, 1] > 0).mean()),
    }

    # ---- first-tick probe (fresh session, zero h, record 0 alone) --------
    sess = ort.InferenceSession(V2, providers=["CPUExecutionProvider"])
    h0 = np.zeros((1, 1, 128), dtype=np.float32)
    a0, _ = sess.run(["actions", "h_out"], {"obs": obs_mat[0].reshape(1, -1), "h_in": h0})
    first_tick = a0.squeeze(0).astype(np.float64).tolist()

    # ---- referent timeline ----------------------------------------------
    refs = [(r["referent"]["x"], r["referent"]["y"]) for r in recs]
    changes = [i for i in range(1, n) if refs[i] != refs[i - 1]]
    change_times = [float(t_sims[i]) for i in changes]
    distinct = len(set(refs))
    # ground-truth strafe onset: gt action at tick t lives in record t+1;
    # its command time is t_sims[t]
    gt_vy = gt[:, 1]
    onset_idx = None
    W = 30  # 1 s sim at 30 Hz
    for t in range(len(gt_vy) - W + 1):
        if np.all(gt_vy[t:t + W] > 0):
            onset_idx = t
            break
    ref_timeline = {
        "n_records": n,
        "t0": float(t0),
        "distinct_referent_positions": distinct,
        "n_changes": len(changes),
        "change_times_sim": change_times,
        "last_change_time_sim": change_times[-1] if changes else None,
        "gt_strafe_onset_t_sim": float(t_sims[onset_idx]) if onset_idx is not None else None,
        "gt_strafe_onset_rel_s": float(t_sims[onset_idx] - t0) if onset_idx is not None else None,
        "gt_vy_first_positive_t_sim": float(t_sims[int(np.argmax(gt_vy > 0))]) if (gt_vy > 0).any() else None,
    }

    # ---- v1 replay -------------------------------------------------------
    act_v1 = replay(V1, obs_mat, f"{OUT_DIR}/actions_v1_cpu.jsonl", t_sims)
    v1 = {
        "vx_mean_all": float(act_v1[:, 0].mean()),
        "vy_mean_all": float(act_v1[:, 1].mean()),
        "wz_mean_all": float(act_v1[:, 2].mean()),
        "vx_mean_first8": float(act_v1[first8, 0].mean()),
        "vy_mean_first8": float(act_v1[first8, 1].mean()),
        "vx_duty_positive_all": float((act_v1[:, 0] > 0).mean()),
        "first_tick": None,
    }
    sess1 = ort.InferenceSession(V1, providers=["CPUExecutionProvider"])
    a1, _ = sess1.run(["actions", "h_out"], {"obs": obs_mat[0].reshape(1, -1), "h_in": h0.copy()})
    v1["first_tick"] = a1.squeeze(0).astype(np.float64).tolist()

    out = {"signature_v2_cpu": sig, "agreement": agree, "first_tick_v2": first_tick,
           "referent_timeline": ref_timeline, "v1_probe": v1,
           "ort_version": ort.__version__}
    print(json.dumps(out, indent=2))
    with open(f"{OUT_DIR}/analysis.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
