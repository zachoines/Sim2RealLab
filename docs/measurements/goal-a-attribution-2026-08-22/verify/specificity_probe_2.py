import json, math
import numpy as np, onnxruntime as ort
SP="docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
NODE="/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"
ONNX="/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v2_998.onnx"
B,D0=-8.1,19; NF,FAR=0.2/6.0,1.0
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
    a=a.reshape(-1); return ((math.degrees(math.atan2(a[1],a[0]))-B)+180.0)%360.0-180.0, a
def show(tag,o):
    d,a=off(o); print("%-62s off %+8.3f  toward=%-5s"%(tag,d,abs(d)<=45.0)); return d

rng=np.random.default_rng(20260822)
m=np.isclose(node[0,D0:],NF,atol=1e-5); n_nf=int(m.sum())
show("node tick-0 unmodified", node[0])
o=node[0].copy(); o[D0:][m]=FAR
show("A: node nearfield pixels -> far clamp (training convention)", o)

# control 1: random non-nearfield pixels, same count
o=node[0].copy(); idx=rng.choice(np.flatnonzero(~m), size=n_nf, replace=False); o[D0:][idx]=FAR
show("C1: random NON-nearfield pixels, same count -> far clamp", o)
# control 2: random pixels anywhere, same count
o=node[0].copy(); idx=rng.choice(3600, size=n_nf, replace=False); o[D0:][idx]=FAR
show("C2: random pixels anywhere, same count -> far clamp", o)
# control 3: nearfield pixels -> a mid depth instead of the far clamp
for mid in (0.5, 1.0, 2.0, 4.0):
    o=node[0].copy(); o[D0:][m]=mid/6.0
    show("C3: node nearfield pixels -> %.1f m (not the far clamp)"%mid, o)
# control 4: scale whole depth image so its mean matches variant A, no pixel-class structure
oA=node[0].copy(); oA[D0:][m]=FAR
o=node[0].copy(); o[D0:]=np.clip(o[D0:]*(oA[D0:].mean()/o[D0:].mean()),0,1)
show("C4: whole image scaled to A's mean depth (no class structure)", o)

print()
# precise inverse: on env rec-15 restore ONLY the slam-created pixels back to the fill
slam=np.isclose(env[15,D0:],FAR,atol=1e-6) & np.isclose(gym[15,D0:],NF,atol=1e-5)
print("slam-created pixels in env rec-15: %d (%.4f)"%(slam.sum(), slam.mean()))
show("env rec-15 unmodified (corrupted)", env[15])
o=env[15].copy(); o[D0:][slam]=NF
show("D: env rec-15 with ONLY the slam-created pixels undone", o)
o=env[15].copy(); o[D0:]=gym[15,D0:]
show("D': env rec-15 with depth <- clean gym rec-15 (published)", o)
# control: undo an equal count of far-clamp pixels that the slam did NOT create
other=np.isclose(env[15,D0:],FAR,atol=1e-6) & ~slam
print("non-slam far-clamp pixels available: %d"%other.sum())
if other.sum():
    o=env[15].copy(); o[D0:][np.flatnonzero(other)]=NF
    show("D-ctrl: env rec-15 with the NON-slam far-clamp pixels set to fill", o)
