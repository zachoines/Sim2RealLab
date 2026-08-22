#!/usr/bin/env python
"""Adversarial verification of arm3 CPU replay. Fully independent re-derivation."""
import json, math
import numpy as np
import onnxruntime as ort

CAP = "/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"
PRIOR = "docs/measurements/goal-a-attribution-2026-08-22/replay"
V2 = "/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v2_998.onnx"
V1 = "/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v1.onnx"

recs = [json.loads(l) for l in open(CAP)]
n = len(recs)
assert n == 1799, n
t = np.array([r["t_sim"] for r in recs])
obs = np.array([r["obs"] for r in recs], dtype=np.float32)
assert obs.shape == (n, 3619)

# ---- model I/O contract ----
sess = ort.InferenceSession(V2, providers=["CPUExecutionProvider"])
io = {"inputs": [(i.name, i.shape, i.type) for i in sess.get_inputs()],
      "outputs": [(o.name, o.shape, o.type) for o in sess.get_outputs()]}
print("IO:", io)

# ---- independent full roll, my own loop ----
h = np.zeros((1, 1, 128), dtype=np.float32)
my_act = np.zeros((n, 3), dtype=np.float64)
for i in range(n):
    a, h = sess.run(["actions", "h_out"], {"obs": obs[i][None, :], "h_in": h})
    my_act[i] = a[0].astype(np.float64)

# prior agent's saved actions
prior_act = np.array([json.loads(l)["action"] for l in open(f"{PRIOR}/actions_v2_cpu.jsonl")])
assert prior_act.shape == (n, 3)
dev = np.abs(my_act - prior_act)
check_ticks = [0, 1, 450, 900, 1350, 1798]
print("per-tick max dev at checks:", {k: float(dev[k].max()) for k in check_ticks})
print("overall max abs dev vs prior saved:", float(dev.max()))

# ---- re-zeroed-h counterexample probe: would a re-zeroed state differ? ----
rez = np.zeros((n, 3))
h0 = np.zeros((1, 1, 128), dtype=np.float32)
for i in [0, 1, 450, 900, 1350, 1798]:
    a, _ = sess.run(["actions", "h_out"], {"obs": obs[i][None, :], "h_in": h0})
    rez[i] = a[0]
print("rolled vs re-zeroed max diff at mid ticks:",
      {k: float(np.abs(my_act[k] - rez[k]).max()) for k in [450, 900, 1350, 1798]})

# ---- signature stats (independent) ----
first8 = t <= t[0] + 8.0
sig = dict(
    vx_mean_first8=float(my_act[first8, 0].mean()),
    vy_mean_first8=float(my_act[first8, 1].mean()),
    vy_duty_pos_first8=float((my_act[first8, 1] > 0).mean()),
    vx_mean_all=float(my_act[:, 0].mean()),
    vy_mean_all=float(my_act[:, 1].mean()),
    vy_duty_all=float((my_act[:, 1] > 0).mean()),
)
s = np.sign(my_act[first8, 1]); s = s[s != 0]
sig["vy_sign_changes_first8"] = int((s[1:] != s[:-1]).sum())

# goal bearing
sx, sy, hd = -0.499, -0.451, 2.22
gx, gy = -2.0, 2.25
bmap = math.atan2(gy - sy, gx - sx)
bbody = math.degrees(math.atan2(math.sin(bmap - hd), math.cos(bmap - hd)))
mv = my_act[first8, :2].mean(axis=0)
cdir = math.degrees(math.atan2(mv[1], mv[0]))
off = (cdir - bbody + 180) % 360 - 180
sig.update(goal_bearing_body_deg=bbody, cmd_dir_first8_deg=cdir, off_goal_deg=off)
print("SIG:", json.dumps(sig, indent=1))

# ---- agreement vs embedded TRT ground truth ----
gt = obs[1:, 16:19].astype(np.float64)
pred = my_act[:-1]
d = np.abs(pred - gt)
agree = dict(mean_abs=d.mean(axis=0).tolist(), max_abs=d.max(axis=0).tolist(),
             corr=[float(np.corrcoef(pred[:, i], gt[:, i])[0, 1]) for i in range(3)])
print("AGREE:", json.dumps(agree, indent=1))

# gt-side signature (for the SURPRISE caveat)
gs = np.sign(gt[t[1:] <= t[0] + 8.0, 1]); gs = gs[gs != 0]
print("GT: vy max", float(gt[:, 1].max()), "vy duty", float((gt[:, 1] > 0).mean()),
      "vy sign changes first8", int((gs[1:] != gs[:-1]).sum()),
      "vx mean", float(gt[:, 0].mean()), "vy mean", float(gt[:, 1].mean()))

# sustained windows on GT (command time of gt row k is t[k], since gt row k = action at tick k)
gt_t = t[:-1]
def sustained_onset(mask, W=30):
    for i in range(len(mask) - W + 1):
        if mask[i:i + W].all():
            return i
    return None
oi = sustained_onset(gt[:, 1] > 0)
ri = sustained_onset(gt[:, 0] < 0)
# wrong-direction: |angle(cmd) - goal bearing| > 90 deg sustained 1 s
cmd_ang = np.degrees(np.arctan2(gt[:, 1], gt[:, 0]))
wrong = np.abs(((cmd_ang - bbody) + 180) % 360 - 180) > 90
wi = sustained_onset(wrong)
print("ONSETS: sustained vy>0 t_sim", float(gt_t[oi]), "rel", float(gt_t[oi]-t[0]),
      "| sustained vx<0 t_sim", float(gt_t[ri]), "rel", float(gt_t[ri]-t[0]),
      "| sustained wrong-dir t_sim", float(gt_t[wi]), "rel", float(gt_t[wi]-t[0]),
      "| first vy>0 t_sim", float(gt_t[int(np.argmax(gt[:, 1] > 0))]))

# window 3-15s vy stats (report claims duty 1.000, mean +0.0247)
w = (gt_t >= t[0] + 3.0) & (gt_t <= t[0] + 15.0)
print("GT 3-15s window: vy mean", float(gt[w, 1].mean()), "duty", float((gt[w, 1] > 0).mean()))
# CPU-side same claims
wc = (t >= t[0] + 3.0) & (t <= t[0] + 15.0)
print("CPU 3-15s window: vy mean", float(my_act[wc, 1].mean()), "duty", float((my_act[wc, 1] > 0).mean()))

# ---- referent timeline (independent) ----
refs = [(r["referent"]["x"], r["referent"]["y"]) for r in recs]
ch = [i for i in range(1, n) if refs[i] != refs[i-1]]
print("REF: distinct", len(set(refs)), "changes", len(ch),
      "first change t", float(t[ch[0]]), "last change t", float(t[ch[-1]]),
      "longest identical run", max(np.diff([0]+ch+[n])) if ch else n)
print("change times:", [round(float(t[i]), 3) for i in ch])

# ---- v1 probe (independent roll) ----
sess1 = ort.InferenceSession(V1, providers=["CPUExecutionProvider"])
h = np.zeros((1, 1, 128), dtype=np.float32)
v1_act = np.zeros((n, 3))
for i in range(n):
    a, h = sess1.run(["actions", "h_out"], {"obs": obs[i][None, :], "h_in": h})
    v1_act[i] = a[0]
prior_v1 = np.array([json.loads(l)["action"] for l in open(f"{PRIOR}/actions_v1_cpu.jsonl")])
print("V1: max dev vs prior saved", float(np.abs(v1_act - prior_v1).max()),
      "vx mean", float(v1_act[:, 0].mean()), "vy mean", float(v1_act[:, 1].mean()),
      "vx first8", float(v1_act[first8, 0].mean()),
      "vx duty>0", float((v1_act[:, 0] > 0).mean()))
