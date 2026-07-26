#!/usr/bin/env python3
"""Measure whether the enriched rooms' ceiling is one-way. (Kit/GPU)

The enriched ceiling is a single open quad whose one normal faces down, drawn
one-sided by RTX back-face culling: the robot's camera underneath must see a
closed ceiling, an overhead camera above it must see straight through. Both
halves are renderer-build-specific — RTX reads a *custom* ``singleSided``
attribute and ignores the ``doubleSided`` schema one, and it culls at all only
while ``/rtx/hydra/faceCulling/enabled`` is set — so this is a measurement to
re-run on a renderer or Isaac Sim upgrade, not a settled fact.

One run records one arm; ``--compare`` reads two recordings back and prints the
deltas, so the two Kit boots never share a process (the culling switch is
global). The arms are orthogonal on purpose: a null result with geometry and
the render switch moving together is unattributable.

  --ceiling surface --face-culling on    the shipped configuration
  --ceiling slab    --face-culling off   an opaque closed box, the control
  --ceiling slab    --face-culling on    the switch alone, on a closed box

Reported per arm: the policy camera's ``distance_to_image_plane`` and RGB from
inside the enclosure, an overhead camera's share of pixels at the ceiling's
range, and the robot's root state. Compared across arms: the policy depth
delta (which must be exactly zero — the depth path has no frame-to-frame noise
floor), the overhead ceiling share, and the RGB means, since a surface that
stops occluding a light also stops shading the room.

Boots Kit and renders. Run only in a cleared GPU window.

Usage (from repo root, after `source env_setup.sh`):
    $ISAACLAB -p source/strafer_lab/scripts/ceiling_sidedness_probe.py \
        --enable_cameras --out shipped.npz
    $ISAACLAB -p source/strafer_lab/scripts/ceiling_sidedness_probe.py \
        --enable_cameras --ceiling slab --face-culling off --out opaque.npz
    python3 source/strafer_lab/scripts/ceiling_sidedness_probe.py \
        --compare opaque.npz shipped.npz
"""
from __future__ import annotations

import argparse
import json

import numpy as np

_FACE_CULLING_KEY = "rtx.hydra.faceCulling.enabled"
_FACE_CULLING_PATH = "/rtx/hydra/faceCulling/enabled"

# Half-width of the range window that counts a pixel as "on the ceiling". Wide
# enough for the surface's own tilt-free spread, far below the drop to the floor.
_RANGE_BAND_M = 0.15


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--compare", nargs=2, metavar=("BEFORE", "AFTER"), default=None,
        help="read two recordings and print the deltas; skips Kit entirely",
    )
    parser.add_argument("--task", default="Isaac-Strafer-Nav-RLDepth-Enriched-Real-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260101)
    parser.add_argument(
        "--settle_steps", type=int, default=10,
        help="zero-action steps between the two recorded renders",
    )
    parser.add_argument(
        "--ceiling", choices=["surface", "slab"], default="surface",
        help="'slab' rebuilds the ceiling as a closed box of the same span, "
             "its underside on the same plane — the opaque control",
    )
    parser.add_argument(
        "--face-culling", choices=["on", "off"], default="on",
        help="the global RTX back-face-culling switch (default on, as the "
             "enriched cfgs ship it)",
    )
    parser.add_argument(
        "--ceiling_z", type=float, default=2.5,
        help="pose height every env's ceiling is pinned to, metres — inside the "
             "shipped band so the range window is readable",
    )
    parser.add_argument(
        "--overhead_alt", type=float, default=6.0,
        help="overhead camera height above the env origin, metres",
    )
    parser.add_argument(
        "--env_spacing", type=float, default=0.0,
        help="override the grid spacing, metres — envs are not visually "
             "isolated, so widening it attributes a delta to what the camera "
             "sees of its neighbours (0 keeps the task's own spacing)",
    )
    parser.add_argument("--out", default="ceiling_probe.npz")
    return parser


def compare(before_path: str, after_path: str) -> None:
    """Print the cross-arm deltas from two recordings."""
    before, after = np.load(before_path), np.load(after_path)
    label_b, label_a = str(before["label"]), str(after["label"])
    print(f"[probe] before = {label_b}  ({before_path})")
    print(f"[probe] after  = {label_a}  ({after_path})")

    for phase in ("reset", "settled"):
        delta = float(
            np.nanmax(np.abs(after[f"policy_depth_{phase}"] - before[f"policy_depth_{phase}"]))
        )
        verdict = "IDENTICAL" if delta == 0.0 else "MOVED"
        print(f"[probe] policy depth max|delta| ({phase:7s}): {delta:.9f} m  {verdict}")

    state = float(np.nanmax(np.abs(after["robot_state_settled"] - before["robot_state_settled"])))
    print(f"[probe] robot root state max|delta|      : {state:.9f}")

    for key, unit in (
        ("overhead_ceiling_fraction", ""),
        ("overhead_rgb_mean", ""),
        ("policy_rgb_mean", ""),
    ):
        print(
            f"[probe] {key:32s}: {float(before[key]):.6f}{unit} -> "
            f"{float(after[key]):.6f}{unit}"
        )
    print(
        f"[probe] face culling live                : "
        f"{bool(before['face_culling_live'])} -> {bool(after['face_culling_live'])}"
    )


def main() -> None:
    parser = build_parser()

    # --compare is numpy over recorded arrays; booting Kit for it would cost a
    # GPU window for arithmetic.
    known_args, _ = parser.parse_known_args()
    if known_args.compare is not None:
        compare(*known_args.compare)
        return

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    simulation_app = AppLauncher(args).app

    import carb
    import gymnasium as gym
    import torch

    import isaaclab.sim as sim_utils
    from isaaclab.sensors import TiledCameraCfg
    from isaaclab_tasks.utils import parse_env_cfg

    import strafer_lab.tasks  # noqa: F401 — triggers gym.register
    from strafer_lab.tasks.navigation.d555_cfg import (
        D555_FOCAL_LENGTH_MM,
        D555_HORIZONTAL_APERTURE_MM,
    )

    def overhead_camera_cfg(alt: float) -> TiledCameraCfg:
        """Nadir camera above each env origin, RGB and depth in one grab."""
        return TiledCameraCfg(
            prim_path="{ENV_REGEX_NS}/ProbeOverheadCam",
            update_period=0.0,
            height=180,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=D555_FOCAL_LENGTH_MM,
                horizontal_aperture=D555_HORIZONTAL_APERTURE_MM,
                clipping_range=(0.01, 100.0),
            ),
            offset=TiledCameraCfg.OffsetCfg(
                pos=(0.0, 0.0, alt),
                rot=(0.0, 0.70710678, 0.0, 0.70710678),  # +90 deg about Y: look down
                convention="world",
            ),
        )

    def closed_slab_cfg(surface_cfg) -> sim_utils.CuboidCfg:
        """A box whose underside sits on the surface's plane, so the two arms
        differ in what the geometry does above that plane and nothing else."""
        thickness = 2.0 * abs(surface_cfg.surface_offset)
        return sim_utils.CuboidCfg(
            size=(surface_cfg.size[0], surface_cfg.size[1], thickness),
            rigid_props=surface_cfg.rigid_props,
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=surface_cfg.visual_material,
        )

    def to_numpy(field) -> np.ndarray:
        """Data fields are torch or warp arrays depending on the backend."""
        if hasattr(field, "detach"):
            return field.detach().float().cpu().numpy()
        return np.asarray(field.numpy())

    def grab(env, name: str) -> dict:
        camera = env.scene.sensors[name]
        env.sim.render()
        camera.update(dt=camera.cfg.update_period, force_recompute=True)
        return {key: to_numpy(plane) for key, plane in camera.data.output.items()}

    def to_bytes(rgb: np.ndarray) -> np.ndarray:
        """Byte image from an annotator that may hand back 0-1 floats."""
        rgb = np.asarray(rgb)[..., :3]
        if np.issubdtype(rgb.dtype, np.floating) and float(np.nanmax(rgb)) <= 1.0:
            rgb = rgb * 255.0
        return np.clip(rgb, 0.0, 255.0).astype(np.uint8)

    def range_band_fraction(depth: np.ndarray, centre: float) -> float:
        """Share of finite pixels ranging within the band — one surface's share."""
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            return 0.0
        inside = (finite > centre - _RANGE_BAND_M) & (finite < centre + _RANGE_BAND_M)
        return float(inside.mean())

    cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
    # Every env enclosed at one known height, so the overhead range band counts
    # ceiling pixels and nothing else.
    cfg.events.generate_room.params["p_ceil"] = 1.0
    cfg.events.generate_room.params["ceiling_height_range"] = (args.ceiling_z, args.ceiling_z)
    setattr(cfg.scene, "probe_overhead_camera", overhead_camera_cfg(args.overhead_alt))

    if args.env_spacing > 0.0:
        cfg.scene.env_spacing = args.env_spacing
    if args.ceiling == "slab":
        cfg.scene.ceiling.spawn = closed_slab_cfg(cfg.scene.ceiling.spawn)
    carb_settings = dict(cfg.sim.render.carb_settings or {})
    carb_settings[_FACE_CULLING_KEY] = args.face_culling == "on"
    cfg.sim.render.carb_settings = carb_settings

    label = f"{args.ceiling}+culling-{args.face_culling}"
    print(f"[probe] arm: {label}, task {args.task}, {args.num_envs} envs", flush=True)

    env = gym.make(args.task, cfg=cfg).unwrapped
    env.reset(seed=args.seed)

    culling_live = carb.settings.get_settings().get(_FACE_CULLING_PATH)
    print(f"[probe] {_FACE_CULLING_PATH} = {culling_live}", flush=True)

    policy_reset = grab(env, "d555_camera")
    overhead = grab(env, "probe_overhead_camera")

    zero = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)
    for _ in range(args.settle_steps):
        env.step(zero)
    policy_settled = grab(env, "d555_camera")
    robot_state = to_numpy(env.scene["robot"].data.root_link_pose_w)

    ceiling_range = args.overhead_alt - args.ceiling_z
    record = {
        "label": label,
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "face_culling_live": bool(culling_live),
        "ceiling_range_m": ceiling_range,
        "policy_depth_reset": policy_reset["distance_to_image_plane"],
        "policy_depth_settled": policy_settled["distance_to_image_plane"],
        "policy_rgb_mean": float(np.asarray(policy_reset["rgb"], dtype=np.float64).mean()),
        "overhead_depth": overhead["distance_to_image_plane"],
        "overhead_rgb": to_bytes(overhead["rgb"]),
        "overhead_rgb_mean": float(np.asarray(overhead["rgb"], dtype=np.float64).mean()),
        "overhead_ceiling_fraction": range_band_fraction(
            overhead["distance_to_image_plane"], ceiling_range
        ),
        "robot_state_settled": robot_state,
    }
    np.savez_compressed(args.out, **record)

    print("\n========== SUMMARY ==========", flush=True)
    print(f"{'arm':<26} {label}")
    print(f"{'face culling live':<26} {bool(culling_live)}")
    print(f"{'overhead ceiling pixels':<26} {record['overhead_ceiling_fraction']:.6f}")
    print(f"{'overhead rgb mean':<26} {record['overhead_rgb_mean']:.4f}")
    print(f"{'policy rgb mean':<26} {record['policy_rgb_mean']:.4f}")
    print(
        "Interpretation: a one-way ceiling reads a near-zero overhead ceiling "
        "share here while a --compare against the opaque control reads a policy "
        "depth delta of exactly 0.0 m. A high overhead share means the surface "
        "still faces the camera above; a non-zero depth delta means the robot's "
        "own view of the enclosure moved and the arm is not distribution-safe."
    )
    print(f"[probe] wrote {args.out}", flush=True)
    print(json.dumps({
        k: v for k, v in record.items() if not isinstance(v, np.ndarray)
    }, indent=2), flush=True)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
