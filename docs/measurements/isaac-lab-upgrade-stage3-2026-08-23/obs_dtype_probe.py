#!/usr/bin/env python3
"""Observation-group dtype census.

rsl-rl 5.4.0 changed the dtype its rollout storage allocates: it now follows the
dtype of the observation tensors it is handed instead of forcing float32. That
change is a no-op only while every observation group the policy consumes is
already float32 — a float64 or float16 group would silently change what the
storage holds, and no config-hash golden or test count would move.

The check has no pre-bump anchor to diff against because it was promoted from a
static reading of the upstream diff, so it is recorded as an absolute statement
about the new pin: every group, its dtype, its shape, and a pass/fail on
"all float32".

Usage (new-pin shell):

    $NEWLAB -p docs/measurements/isaac-lab-upgrade-stage3-2026-08-23/obs_dtype_probe.py \
        --headless --env Isaac-Strafer-Nav-RLNoCam-v0 --num_envs 4 --seed 42 \
        --out <dir>/physics/obs-dtypes.json
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Observation-group dtype census")
    parser.add_argument("--env", type=str, default="Isaac-Strafer-Nav-RLNoCam-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, required=True)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import json
    import pathlib
    import sys

    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import parse_env_cfg

    import isaaclab
    import strafer_lab  # noqa: F401  (registers the envs)

    print(f"BINDING sys.executable = {sys.executable}")
    print(f"BINDING isaaclab.__file__ = {isaaclab.__file__}")

    device = getattr(args, "device", "cuda:0")
    env_cfg = parse_env_cfg(args.env, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.env, cfg=env_cfg)

    obs, _ = env.reset(seed=args.seed)

    groups = {}

    def record(prefix, value):
        if isinstance(value, dict):
            for key, sub in value.items():
                record(f"{prefix}/{key}" if prefix else str(key), sub)
        elif isinstance(value, torch.Tensor):
            groups[prefix] = {
                "dtype": str(value.dtype),
                "shape": list(value.shape),
                "device": str(value.device),
                "is_float32": value.dtype == torch.float32,
            }
        else:
            groups[prefix] = {"dtype": f"<non-tensor {type(value).__name__}>",
                              "is_float32": False}

    record("", obs)

    all_f32 = all(g.get("is_float32") for g in groups.values()) and bool(groups)
    meta = {
        "env": args.env,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "device": device,
        "torch": torch.__version__,
        "isaaclab": isaaclab.__file__,
        "groups": groups,
        "all_groups_float32": all_f32,
        "verdict": "PASS — rsl-rl storage dtype change is a no-op"
        if all_f32
        else "FAIL — a non-float32 group changes rsl-rl >=5.4.0 storage dtype",
    }
    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
