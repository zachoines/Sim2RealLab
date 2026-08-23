"""Re-derives the depth convention divergence from the copied artifacts alone.

Run from the repo root. Needs only numpy: both inputs are in this record.
  gym_obs.jsonl    clean sim observation term output (nearfield fill applied)
  env_obsbuf.jsonl the same ticks after the ObservationManager's noise models
Depth occupies dims 19: of each 3619-vector, scaled by 1/DEPTH_MAX, 45x80.
"""
import json
import numpy as np

SP = "docs/measurements/goal-a-attribution-2026-08-22/same-pose-probe"
D0, H, W = 19, 45, 80
FILL, FAR = 0.2 / 6.0, 6.0 / 6.0

gym = np.array([json.loads(l)["obs"] for l in open(f"{SP}/gym_obs.jsonl")])
env = np.array([r["obs"] for r in map(json.loads, open(f"{SP}/env_obsbuf.jsonl"))
                if not r.get("header")])
G = gym[:, D0:].reshape(-1, H, W)
E = env[:, D0:].reshape(-1, H, W)

clean_fill = np.isclose(G, FILL, atol=1e-5)
env_far = np.isclose(E, FAR, atol=1e-6)
slammed = clean_fill & env_far

print(f"records paired                        {len(G)}")
print(f"clean nearfield-fill share of frame   {clean_fill.mean():.4f}")
print(f"clean far-clamp share of frame        {np.isclose(G, FAR, atol=1e-6).mean():.4f}")
print(f"noisy far-clamp share of frame        {env_far.mean():.4f}")
print(f"slammed (clean 0.2 m -> noisy 6.0 m)  {slammed.mean():.4f} of frame")
print(f"  as a share of the nearfield class   {slammed.sum() / clean_fill.sum():.4f}")
rows = slammed.mean(axis=(0, 2))
print(f"rows 0-21   mean {rows[:22].mean():.4f}   max {rows[:22].max():.4f}")
print(f"rows 22-44  mean {rows[22:].mean():.4f}   min {rows[22:].min():.4f}"
      f"   max {rows[22:].max():.4f}")
