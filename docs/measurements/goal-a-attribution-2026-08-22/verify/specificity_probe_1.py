"""Does the nearfield slam ALONE control the command? Apply the training
convention to the robot's own captured depth and re-run v2."""
import json, math, os
import numpy as np, onnxruntime as ort
SP="docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
# Run from the repo root.
#   ONNX  — untracked repo content; sha256 digests in ../provenance.md
#   NODE  — machine-local, 131 MB, not in git; location and sha256 in ../provenance.md
ONNX = "models/strafer_depth_subgoal_v2_998.onnx"
NODE = os.path.expanduser("~/arm3_obs_capture_20260822/node_obs.jsonl")
B,D0=-8.1,19
NF,FAR=0.2/6.0,1.0
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
def cmd(o):
    h=np.zeros((1,1,128),dtype=np.float32)
    a,_=s.run(["actions","h_out"],{"obs":o.astype(np.float32).reshape(1,-1),"h_in":h})
    return a.reshape(-1).astype(np.float64)
def off(a): return ((math.degrees(math.atan2(a[1],a[0]))-B)+180.0)%360.0-180.0
def show(tag,o):
    a=cmd(o); d=off(a)
    print("%-58s off %+8.3f deg  toward_goal=%-5s  [%+.5f %+.5f %+.5f]"%(tag,d,abs(d)<=45.0,*a))
    return d

nd=node[:, D0:]
print("node depth: share at nearfield fill (0.2 m) = %.4f ; at far clamp = %.4f"%(
    np.isclose(nd[0],NF,atol=1e-5).mean(), np.isclose(nd[0],FAR,atol=1e-6).mean()))
print()
show("node tick-0, unmodified (= the rig's own command)", node[0])

# A: apply the training slam to the node's OWN frame: every nearfield-fill pixel -> far clamp
o=node[0].copy(); m=np.isclose(o[D0:],NF,atol=1e-5); o[D0:][m]=FAR
print("   (pixels slammed: %d of 3600, %.4f)"%(m.sum(), m.mean()))
show("node tick-0, training convention applied to its own depth", o)

# B: half of them, deterministic checkerboard -- the ~50% the dither actually sends over
o=node[0].copy(); m=np.isclose(o[D0:],NF,atol=1e-5).reshape(45,80)
chk=np.zeros_like(m); chk[::2,::2]=True; chk[1::2,1::2]=True
mm=(m&chk); o[D0:]=o[D0:].reshape(45,80).copy().ravel(); o[D0:][mm.ravel()]=FAR
print("   (pixels slammed: %d, %.4f)"%(mm.sum(), mm.mean()))
show("node tick-0, half the nearfield pixels slammed (checkerboard)", o)

# C: control -- slam an equal number of NON-nearfield pixels instead
o=node[0].copy(); nonm=(~np.isclose(o[D0:],NF,atol=1e-5))
idx=np.flatnonzero(nonm)[:int(m.sum())]
o[D0:][idx]=FAR
print("   (control: %d non-nearfield pixels set to far clamp)"%len(idx))
show("node tick-0, equal-count far-clamp on NON-nearfield pixels (control)", o)

# D: inverse -- take the env's corrupted depth and undo the slam (far clamp -> nearfield fill)
o=env[15].copy(); m2=np.isclose(o[D0:],FAR,atol=1e-6)
o[D0:][m2]=NF
print("   (un-slammed: %d of 3600, %.4f)"%(m2.sum(), m2.mean()))
show("env rec-15 corrupted depth with the slam UNDONE (inverse)", o)

# E: across all 30 node ticks, convention applied
f=0
for i in range(30):
    o=node[i].copy(); mi=np.isclose(o[D0:],NF,atol=1e-5); o[D0:][mi]=FAR
    f += abs(off(cmd(o)))<=45.0
print("\nnode ticks 0-29 with the training convention applied: %d/30 toward goal"%f)
f=0
for i in range(30):
    f += abs(off(cmd(node[i])))<=45.0
print("node ticks 0-29 unmodified:                            %d/30 toward goal"%f)
