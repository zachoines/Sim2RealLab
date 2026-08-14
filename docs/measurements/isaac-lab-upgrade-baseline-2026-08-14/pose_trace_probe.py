#!/usr/bin/env python3
"""Seeded replayed-action pose trace — the determinism floor (D0) for physics.

No same-pin same-seed rerun-identity evidence exists for this stack, so a
strict pre/post physics comparison has no calibration: a post-bump difference
cannot be read as a regression until it is known what two identical runs of the
*same* build produce. This probe answers that. Two back-to-back runs feed one
env the same seeded action sequence and dump the chassis pose per step; if the
two dumps hash equal, physics gates after the bump can be hash gates, and if
they do not, the observed spread is the noise floor every gate must clear.

Two hashes are emitted, deliberately separated:

  ``dr_hash``    — the post-reset draw: root state, joint state, and (where the
                   physics view exposes them) per-body masses and material
                   properties, captured immediately after the first reset.
  ``trace_hash`` — the per-step root position and orientation over the rollout.

An RNG-order change upstream moves the first and drags the second with it; a
solver or contact change moves only the second. Collapsed into one hash the two
causes are indistinguishable, and the attribution is the whole point.

Usage (from the repo root, after ``source env_setup.sh``):

    $ISAACLAB -p docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/pose_trace_probe.py \
        --headless --num_envs 16 --seed 42 --steps 300 \
        --out docs/measurements/isaac-lab-upgrade-baseline-2026-08-14/physics/pose_trace_run1.npz
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Seeded pose-trace determinism probe")
    parser.add_argument("--env", type=str, default="Isaac-Strafer-Nav-RLNoCam-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--out", type=str, required=True)

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import hashlib
    import json
    import pathlib

    import gymnasium as gym
    import numpy as np
    import torch
    import warp as wp
    from isaaclab_tasks.utils import parse_env_cfg

    import strafer_lab  # noqa: F401  (registers the envs)

    device = getattr(args, "device", "cuda:0")
    env_cfg = parse_env_cfg(args.env, device=device, num_envs=args.num_envs)
    env_cfg.seed = args.seed
    env = gym.make(args.env, cfg=env_cfg)

    def _np(t):
        """float64 numpy from a torch tensor, a warp array, or plain numpy.

        Asset-data fields on this stack are already warp arrays for some
        properties and torch tensors for others, and the physx view getters
        return numpy — the probe reads all three, so it converts all three
        rather than assuming one.
        """
        if isinstance(t, wp.array):
            t = wp.to_torch(t)
        if isinstance(t, torch.Tensor):
            return t.detach().cpu().numpy().astype(np.float64)
        return np.asarray(t, dtype=np.float64)

    def _hash(arrays):
        h = hashlib.sha256()
        for key in sorted(arrays):
            h.update(key.encode())
            h.update(np.ascontiguousarray(arrays[key]).tobytes())
        return h.hexdigest()

    env.reset(seed=args.seed)
    robot = env.unwrapped.scene["robot"]

    # --- the post-reset draw ------------------------------------------------
    dr = {
        "root_state_w": _np(robot.data.root_state_w),
        "joint_pos": _np(robot.data.joint_pos),
        "joint_vel": _np(robot.data.joint_vel),
    }
    # Randomized physical properties live on the physx view, not on .data, and
    # not every build exposes every getter — absence is recorded, never faked.
    dr_absent = []
    view = getattr(robot, "root_physx_view", None)
    for label, getter in (
        ("masses", "get_masses"),
        ("inertias", "get_inertias"),
        ("material_properties", "get_material_properties"),
    ):
        fn = getattr(view, getter, None) if view is not None else None
        if fn is None:
            dr_absent.append(label)
            continue
        try:
            dr[label] = _np(fn())
        except Exception as exc:  # noqa: BLE001 — record, don't guess
            dr_absent.append(f"{label} ({type(exc).__name__})")

    # --- the rollout --------------------------------------------------------
    # A generator on the compute device, so the action stream is reproducible
    # without depending on global RNG state any env reset may also consume.
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)
    action_dim = env.unwrapped.action_manager.total_action_dim

    positions = np.empty((args.steps, args.num_envs, 3), dtype=np.float64)
    quats = np.empty((args.steps, args.num_envs, 4), dtype=np.float64)
    actions_log = np.empty((args.steps, args.num_envs, action_dim), dtype=np.float64)

    for step in range(args.steps):
        actions = (
            torch.rand(
                (args.num_envs, action_dim),
                generator=gen,
                device=device,
                dtype=torch.float32,
            )
            * 2.0
            - 1.0
        )
        env.step(actions)
        positions[step] = _np(robot.data.root_pos_w)
        quats[step] = _np(robot.data.root_quat_w)
        actions_log[step] = _np(actions)

    trace = {"positions": positions, "quats": quats}
    dr_hash = _hash(dr)
    trace_hash = _hash(trace)
    action_hash = _hash({"actions": actions_log})

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **dr, **trace, actions=actions_log)

    meta = {
        "env": args.env,
        "num_envs": args.num_envs,
        "seed": args.seed,
        "steps": args.steps,
        "device": device,
        "action_dim": action_dim,
        "dr_hash": dr_hash,
        "trace_hash": trace_hash,
        "action_hash": action_hash,
        "dr_fields": sorted(dr),
        "dr_fields_unavailable": dr_absent,
        "npz": out.name,
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    print(json.dumps(meta, indent=2, sort_keys=True))

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
