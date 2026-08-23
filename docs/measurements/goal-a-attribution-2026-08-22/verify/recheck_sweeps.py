import json, math, os
import numpy as np, onnxruntime as ort
SP="docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
# Run from the repo root.
#   ONNX  — untracked repo content; sha256 digests in ../provenance.md
#   NODE  — machine-local, 131 MB, not in git; location and sha256 in ../provenance.md
ONNX = "models/strafer_depth_subgoal_v2_998.onnx"
NODE = os.path.expanduser("~/arm3_obs_capture_20260822/node_obs.jsonl")
B,D0=-8.1,19
def rows(p,n=None,dh=False):
    o=[]
    for l in open(p):
        r=json.loads(l)
        if dh and r.get("header"): continue
        o.append(r["obs"])
        if n and len(o)>=n: break
    return np.asarray(o,dtype=np.float64)
node=rows(NODE,n=30); gym=rows(f"{SP}/gym_obs.jsonl"); env=rows(f"{SP}/env_obsbuf.jsonl",dh=True)
s=ort.InferenceSession(ONNX,providers=["CPUExecutionProvider"])
def off(o):
    h=np.zeros((1,1,128),dtype=np.float32)
    a,_=s.run(["actions","h_out"],{"obs":o.astype(np.float32).reshape(1,-1),"h_in":h})
    a=a.reshape(-1)
    return ((math.degrees(math.atan2(a[1],a[0]))-B)+180.0)%360.0-180.0
pub={r["rec"]:r["off"] for r in json.load(open(f"{SP}/patch_replays_consolidated.json"))["sweeps"]["gymclean_sweep"]}
w=0.0; flips=0
for i in range(30):
    o=gym[i].copy(); o[10:14]=node[0,10:14]; o[16:19]=node[0,16:19]
    d=off(o); w=max(w,abs(d-pub[i])); flips += abs(d)<=45.0
print("clean-sim rows (node bookkeeping) toward goal:", f"{flips}/30", " worst |delta| vs published:", f"{w:.3e}")
# stricter single-field variant: node obs, depth <- clean gym row i
f2=0
for i in range(30):
    o=node[0].copy(); o[D0:]=gym[i,D0:]
    f2 += abs(off(o))<=45.0
print("node obs with clean depth substituted, toward goal:", f"{f2}/30")
