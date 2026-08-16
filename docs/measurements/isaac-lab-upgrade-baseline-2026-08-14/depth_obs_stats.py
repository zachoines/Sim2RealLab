#!/usr/bin/env python3
"""Depth-observation statistics — the renderer's fingerprint on the policy input.

The config-hash goldens cover the depth term's *declaration*, never the tensor
it produces. A renderer that starts honoring the authored vertical aperture, a
denoiser or antialiasing default that moves, or a camera-mount change all leave
every golden green while feeding a deployed checkpoint a different picture. The
only pre/post evidence for that class is the distribution of the depth block
itself.

Actions are held at zero: the observation is sampled from the reset pose, so
the numbers move when the *renderer* moves and are not dragged around by
whatever the physics determinism floor turns out to be. Domain randomization
still spreads the envs over distinct spawn poses, so the sample is not one
viewpoint repeated.

A per-row mean profile is recorded alongside the scalar moments because a
vertical-FOV change is a *shape* change in that profile — floor rows brighten
and ceiling rows darken — which a single mean can hide.

Usage (from the repo root, after ``source env_setup.sh``):

    D=docs/measurements/isaac-lab-upgrade-baseline-2026-08-14
    $ISAACLAB -p $D/depth_obs_stats.py \
        --headless --enable_cameras --num_envs 8 --seed 42 --frames 30 \
        --env Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0 \
        --out $D/render/depth_obs_enriched_robust_play.json
"""

import argparse

DEPTH_WIDTH = 80
DEPTH_HEIGHT = 45
N_SCALARS = 19


def main():
    parser = argparse.ArgumentParser(description="Depth-observation statistics probe")
    parser.add_argument(
        "--env", type=str, default="Isaac-Strafer-Nav-RLDepth-Subgoal-Enriched-Robust-Play-v0"
    )
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--out", type=str, required=True)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import hashlib
    import json
    import pathlib

    import gymnasium as gym
    import numpy as np
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import strafer_lab  # noqa: F401  (registers the envs)

    device = getattr(args, "device", "cuda:0")
    env_cfg = parse_env_cfg(args.env, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.env, cfg=env_cfg)

    obs, _ = env.reset(seed=args.seed)
    action_dim = env.unwrapped.action_manager.total_action_dim
    zero = torch.zeros((args.num_envs, action_dim), device=device, dtype=torch.float32)

    def policy_obs(o):
        t = o["policy"] if isinstance(o, dict) else o
        return t.detach().cpu().numpy().astype(np.float64)

    frames = []
    for step in range(args.frames):
        arr = policy_obs(obs)
        depth = arr[:, N_SCALARS:]
        assert depth.shape[1] == DEPTH_WIDTH * DEPTH_HEIGHT, depth.shape
        frames.append(depth.reshape(args.num_envs, DEPTH_HEIGHT, DEPTH_WIDTH))
        obs, _, _, _, _ = env.step(zero)

    stack = np.stack(frames, axis=0)  # (frames, envs, H, W)

    def moments(a):
        return {
            "mean": float(a.mean()),
            "std": float(a.std()),
            "min": float(a.min()),
            "max": float(a.max()),
            "p01": float(np.percentile(a, 1)),
            "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99)),
            "frac_at_min": float(np.mean(a <= a.min() + 1e-9)),
            "frac_at_max": float(np.mean(a >= a.max() - 1e-9)),
        }

    payload = {
        "env": args.env,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "frames": args.frames,
        "device": device,
        "depth_shape": [DEPTH_HEIGHT, DEPTH_WIDTH],
        "actions": "zero",
        "aggregate": moments(stack),
        "per_frame_mean": [float(x) for x in stack.mean(axis=(1, 2, 3))],
        "per_frame_std": [float(x) for x in stack.std(axis=(1, 2, 3))],
        "row_band_mean": [float(x) for x in stack.mean(axis=(0, 1, 3))],
        "col_band_mean": [float(x) for x in stack.mean(axis=(0, 1, 2))],
        "first_frame_per_env_mean": [float(x) for x in stack[0].mean(axis=(1, 2))],
        "sha256_float64": hashlib.sha256(
            np.ascontiguousarray(stack).tobytes()
        ).hexdigest(),
        "sha256_float32": hashlib.sha256(
            np.ascontiguousarray(stack.astype(np.float32)).tobytes()
        ).hexdigest(),
    }

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(out.with_suffix(".npz"), depth=stack.astype(np.float32))

    scalars = {k: v for k, v in payload.items() if not isinstance(v, list)}
    print(json.dumps(scalars, indent=2, sort_keys=True))
    print("row_band_mean[0:5] =", payload["row_band_mean"][:5])
    print("row_band_mean[-5:] =", payload["row_band_mean"][-5:])

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
