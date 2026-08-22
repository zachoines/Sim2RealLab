"""Independent re-derivation of the decisive patch replays and the two sweeps.
Written from the published method description, not by re-executing the original
script. Compares against patch_replays_consolidated.json."""
import json, math, hashlib
import numpy as np
import onnxruntime as ort

SP   = "docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
NODE = "/home/zachoines/arm3_obs_capture_20260822/node_obs.jsonl"
ONNX = "/home/zachoines/Workspace/Sim2RealLab/models/strafer_depth_subgoal_v2_998.onnx"
BEARING, D0 = -8.1, 19

print("onnx sha256:", hashlib.sha256(open(ONNX,'rb').read()).hexdigest())
print("ort:", ort.__version__)

def rows(p, n=None, drop_header=False):
    out=[]
    for l in open(p):
        r=json.loads(l)
        if drop_header and r.get("header"): continue
        out.append(r["obs"])
        if n and len(out)>=n: break
    return np.asarray(out, dtype=np.float64)

node = rows(NODE, n=30)
gym  = rows(f"{SP}/gym_obs.jsonl")
env  = rows(f"{SP}/env_obsbuf.jsonl", drop_header=True)
print("shapes", node.shape, gym.shape, env.shape)

sess = ort.InferenceSession(ONNX, providers=["CPUExecutionProvider"])
def act(o):
    h = np.zeros((1,1,128), dtype=np.float32)
    a,_ = sess.run(["actions","h_out"], {"obs": o.astype(np.float32).reshape(1,-1), "h_in": h})
    return a.reshape(-1).astype(np.float64)
def off(a):
    return ((math.degrees(math.atan2(a[1], a[0])) - BEARING) + 180.0) % 360.0 - 180.0

cases = {}
cases["a_node0_unmodified"]        = node[0].copy()
o = node[0].copy(); o[D0:] = gym[:,D0:].mean(axis=0)
cases["b_node0_gym_mean_depth"]    = o
o = node[0].copy(); o[D0:] = env[15, D0:]
cases["t_node0_envnoise_depth"]    = o
cases["t_envbuf_rec0"]             = env[0].copy()
cases["t_envbuf_rec15"]            = env[15].copy()
o = env[15].copy(); o[D0:] = gym[15, D0:]
cases["t_envbuf15_clean_depth"]    = o
o = env[15].copy(); o[D0:] = node[0, D0:]
cases["t_envbuf15_node_depth"]     = o
o = gym[15].copy(); o[10:14] = node[0,10:14]; o[16:19] = node[0,16:19]; o[D0:] = env[15,D0:]
cases["t_cleancell_envnoise_depth"]= o

pub = {p["patch"]: p for p in json.load(open(f"{SP}/patch_replays_consolidated.json"))["patch_replays"]}
print(f"\n{'case':34s} {'off_goal_deg':>13s} {'published':>11s} {'|delta|':>10s}  cmd")
worst = 0.0
for k, o in cases.items():
    a = act(o); d = off(a); p = pub[k]["off_goal_deg"]
    worst = max(worst, abs(d-p), *(abs(x-y) for x,y in zip(a, pub[k]["command"])))
    print(f"{k:34s} {d:13.4f} {p:11.4f} {abs(d-p):10.2e}  [{a[0]:+.5f} {a[1]:+.5f} {a[2]:+.5f}]")

sw = json.load(open(f"{SP}/patch_replays_consolidated.json"))["sweeps"]
env_flip = sum(1 for i in range(30) if abs(off(act(env[i]))) <= 45.0)
gym_flip = 0
for i in range(30):
    o = node[0].copy(); o[D0:] = gym[i, D0:]
    if abs(off(act(o))) <= 45.0: gym_flip += 1
print(f"\nenv noise-bearing rows toward goal : {env_flip}/30   published {sw['env_noise_rows_toward_goal']}")
print(f"clean-depth rows toward goal       : {gym_flip}/30   published {sw['gymclean_rows_toward_goal']}")
print(f"\nworst absolute deviation across all re-derived quantities: {worst:.3e}")
