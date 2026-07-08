#!/usr/bin/env python3
"""Kit gate for the D555 policy-camera vertical-FOV parity fix (80x45).

Background: Isaac Sim / RTX derives a camera's vertical FOV from its render
RESOLUTION aspect ratio (square pixels) and IGNORES the authored
``vertical_aperture`` — confirmed directly by an earlier run of this probe (the
prim's verticalAperture changed but the rendered depth did not). So the only
lever on the sim policy camera's vertical FOV is its resolution. The fix makes
the policy camera 80x45 (16:9), whose square-pixel-derived VFOV is 56.4 deg —
matching the real D555 / perception camera — instead of the 4:3 80x60's ~71 deg.

This probe renders one static ground-plane scene from three cameras at one
identical pose (level, forward) and compares row-wise depth profiles:

  ref      - perception camera (640x360, 16:9), block-averaged 8x8 -> 80x45.
             What deployment feeds the policy: the real sensor's vertical FOV
             downsampled to the policy grid.
  fixed    - policy camera via make_d555_camera_cfg() -> now 80x45 (the fix).
  control  - an 80x60 (4:3) camera: the pre-fix resolution, rendered as
             before/after evidence that the geometry actually changed.

Acceptance (exit 0):
  1. ``fixed`` matches ``ref`` row-for-row at sensor-noise level -- the 80x45
     policy render spans the real sensor's vertical FOV, so the deploy 8x8
     block-average maps onto the identical angular grid.
  2. The effective vertical FOV (measured from the rendered floor profile) is
     ~56 deg for ``fixed`` and ~71 deg for ``control`` -- proof the resolution
     change actually moved the geometry (and that ``fixed`` is NOT still the old
     4:3 FOV).

This protects the ~16 h retrain investment: run it and confirm PASS BEFORE
retraining DEPTH_SUBGOAL on the 80x45 camera.

Usage (from repo root, after ``source env_setup.sh``)::

    $STRAFER_ISAACLAB_PYTHON source/strafer_lab/scripts/depth_camera_vfov_probe.py \
        --headless --report /tmp/vfov_probe.txt
"""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(description="D555 policy-camera VFOV parity probe (80x45)")
    parser.add_argument(
        "--probe-height", type=float, default=1.0,
        help="Camera height (m) above the ground plane; the level forward "
             "camera sees the floor in its lower FOV, giving a row-wise depth "
             "gradient set purely by the vertical FOV.",
    )
    parser.add_argument(
        "--row-tol", type=float, default=0.10,
        help="Max allowed per-row |depth delta| (m) between the fixed policy "
             "camera and the block-averaged perception reference for PASS. "
             "0.10 m is well within the real D555's depth noise at range (~2%% "
             "at 6 m) and ~10x tighter than a real VFOV mismatch would produce "
             "(meter-scale across many rows); the residual max sits on the 6 m "
             "clip-boundary row, where the perception block-average straddles "
             "the saturation kink (a sampling artifact, not geometry).",
    )
    parser.add_argument(
        "--row-mean-tol", type=float, default=0.02,
        help="Max allowed MEAN per-row delta (m). Catches a systematic VFOV "
             "mismatch (which shifts every row) even if no single row trips "
             "the max tolerance.",
    )
    parser.add_argument(
        "--settle-frames", type=int, default=12,
        help="Render frames to step before reading depth (RTX convergence).",
    )
    parser.add_argument(
        "--report", type=str, default="depth_vfov_probe_report.txt",
        help="File to write the full report + verdict to. Isaac Sim's os._exit "
             "on app close truncates stdout, so the file is the source of truth.",
    )

    from isaaclab.app import AppLauncher

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import copy
    import math
    import warnings

    import numpy as np
    import torch
    import warp as wp

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext
    from isaaclab.sensors import TiledCamera, TiledCameraCfg

    from strafer_lab.tasks.navigation.d555_cfg import (
        D555_CAMERA_ROT_ROS,
        make_d555_camera_cfg,
        make_d555_perception_camera_cfg,
    )
    from strafer_shared.constants import (
        DEPTH_HEIGHT,
        DEPTH_MAX,
        DEPTH_WIDTH,
        PERCEPTION_HEIGHT,
        PERCEPTION_WIDTH,
    )

    def to_t(x):
        return wp.to_torch(x) if isinstance(x, wp.array) else x

    def _place(cfg, prim_path):
        cfg = copy.deepcopy(cfg)
        cfg.prim_path = prim_path
        cfg.data_types = ["distance_to_image_plane"]
        # All cameras share ONE pose: level, forward-facing, at probe_height.
        # The floor fills the lower FOV; the row profile below the image center
        # is set purely by the vertical FOV.
        cfg.offset = TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, args.probe_height),
            rot=D555_CAMERA_ROT_ROS,
            convention="ros",
        )
        return cfg

    ref_cfg = _place(make_d555_perception_camera_cfg(), "/World/cam_ref")
    fixed_cfg = _place(
        make_d555_camera_cfg(data_types=("distance_to_image_plane",)), "/World/cam_fixed"
    )
    assert fixed_cfg.height == DEPTH_HEIGHT and fixed_cfg.width == DEPTH_WIDTH  # 45 x 80

    # The pre-fix 4:3 camera: same as the policy cam but at the old 80x60.
    control_cfg = _place(
        make_d555_camera_cfg(data_types=("distance_to_image_plane",)), "/World/cam_control"
    )
    control_cfg.height = 60  # the 4:3 resolution the fix replaces

    # --- Scene: ground plane + light. distance_to_image_plane over the floor (a
    # surface PARALLEL to the optical axis) varies row-to-row purely with the
    # vertical FOV, so the row profile is the VFOV fingerprint. (A frontal wall
    # would not work: a plane perpendicular to the optical axis has one constant
    # image-plane distance, so its profile is flat regardless of VFOV.)
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    ground = sim_utils.GroundPlaneCfg()
    ground.func("/World/ground", ground)
    light = sim_utils.DomeLightCfg(intensity=1000.0)
    light.func("/World/Light", light)

    ref_cam = TiledCamera(ref_cfg)
    fixed_cam = TiledCamera(fixed_cfg)
    control_cam = TiledCamera(control_cfg)

    sim.reset()
    for _ in range(args.settle_frames):
        sim.step()
        ref_cam.update(sim.get_physics_dt())
        fixed_cam.update(sim.get_physics_dt())
        control_cam.update(sim.get_physics_dt())

    def depth_hw(cam):
        d = to_t(cam.data.output["distance_to_image_plane"])[0].float()
        if d.ndim == 3:  # (H, W, 1)
            d = d[..., 0]
        d = torch.where(torch.isfinite(d), d, torch.full_like(d, float("nan")))
        out = d.detach().cpu().numpy().astype(np.float64)
        # Clip to the D555 saturation range the policy actually sees: the obs
        # pipeline caps depth at DEPTH_MAX (6 m). Beyond that both cameras read
        # 6 m identically, so the near-horizon rows (depth -> inf) — where any
        # sub-pixel sampling difference explodes into tens of metres — collapse
        # to the meaningful, in-range comparison. NaN (sky) is preserved.
        return np.minimum(out, DEPTH_MAX)

    ref_full = depth_hw(ref_cam)        # (360, 640)
    fixed_d = depth_hw(fixed_cam)       # (45, 80)
    control_d = depth_hw(control_cam)   # (60, 80)

    def finite_frac(d):
        return float(np.isfinite(d).mean())

    def nanmean(a, axis):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmean(a, axis=axis)

    # Block-average the perception depth to the policy grid, exactly as the
    # deploy depth downsampler does (mean over each 8x8 pixel block; 640/80 and
    # 360/45 both = 8 because both are 16:9).
    bh, bw = PERCEPTION_HEIGHT // DEPTH_HEIGHT, PERCEPTION_WIDTH // DEPTH_WIDTH  # 8, 8
    ref_blocks = ref_full.reshape(DEPTH_HEIGHT, bh, DEPTH_WIDTH, bw)
    ref_d = nanmean(ref_blocks, axis=(1, 3))  # (45, 80)

    ref_rows = nanmean(ref_d, axis=1)        # (45,)
    fixed_rows = nanmean(fixed_d, axis=1)     # (45,)
    control_rows = nanmean(control_d, axis=1)  # (60,)

    # fixed vs ref: both 45 rows, same grid -> row-for-row comparison.
    valid = np.isfinite(ref_rows) & np.isfinite(fixed_rows)
    fixed_delta = np.abs(fixed_rows[valid] - ref_rows[valid])
    fixed_max = float(np.max(fixed_delta)) if fixed_delta.size else float("nan")
    fixed_mean = float(np.mean(fixed_delta)) if fixed_delta.size else float("nan")

    # Effective vertical FOV measured from the rendered floor: for a level
    # camera at height h, the bottom row (r = H-1, y_ndc = (H-1)/H) reads floor
    # depth d = h / (y_ndc * tan(vfov/2)), so tan(vfov/2) = h / (d * y_ndc).
    def measured_vfov_deg(rows, H):
        finite = np.where(np.isfinite(rows))[0]
        if finite.size == 0:
            return float("nan")
        r = int(finite.max())  # bottom-most floor row
        y_ndc = (2.0 * r + 1.0 - H) / H
        d = float(rows[r])
        if d <= 0 or y_ndc <= 0:
            return float("nan")
        return 2.0 * math.degrees(math.atan(args.probe_height / (d * y_ndc)))

    vfov_fixed = measured_vfov_deg(fixed_rows, DEPTH_HEIGHT)
    vfov_control = measured_vfov_deg(control_rows, 60)

    # --- Report (buffered to a list, written to file AND stdout; the file is
    #     authoritative because Isaac Sim's os._exit truncates stdout). ---
    lines: list[str] = []

    def emit(s: str = "") -> None:
        lines.append(s)
        print(s, flush=True)

    emit("\n=== D555 policy-camera VFOV parity probe (80x45) ===")
    emit(f"camera pose: level forward, height {args.probe_height:.2f} m")
    emit(f"finite-pixel fraction  ref={finite_frac(ref_full):.2f}  "
         f"fixed={finite_frac(fixed_d):.2f}  control={finite_frac(control_d):.2f}")
    emit(f"resolutions            ref=640x360  fixed={DEPTH_WIDTH}x{DEPTH_HEIGHT}  control=80x60")
    emit(f"measured vertical FOV  fixed={vfov_fixed:.2f} deg (target 56.4)  "
         f"control={vfov_control:.2f} deg (old 4:3, ~71.1)")
    emit(f"floor rows compared (fixed vs ref): {int(valid.sum())} of {DEPTH_HEIGHT}")
    emit("")
    emit("row  |   ref(m)  fixed(m) | |fixed-ref|")
    idx = np.where(valid)[0]
    for r in idx[:: max(1, len(idx) // 20)]:
        emit(f"{r:3d}  | {ref_rows[r]:8.3f} {fixed_rows[r]:8.3f} | {abs(fixed_rows[r]-ref_rows[r]):9.4f}")
    emit("")
    emit(f"fixed-vs-ref      max={fixed_max:.4f} m  mean={fixed_mean:.4f} m  (tol {args.row_tol} m)")

    # --- Gates ---
    ok = True

    if not (finite_frac(fixed_d) > 0.15 and valid.sum() >= 5):
        emit(f"FAIL: the fixed camera sees too little floor "
             f"(finite {finite_frac(fixed_d):.2f}, {int(valid.sum())} rows) -- cannot certify. "
             "Check camera orientation / probe height.")
        ok = False

    if not fixed_max <= args.row_tol:
        emit(f"FAIL: fixed-vs-ref max row delta {fixed_max:.4f} m exceeds tol {args.row_tol} m "
             "-- the 80x45 policy render does NOT match the deploy reference.")
        ok = False

    if not fixed_mean <= args.row_mean_tol:
        emit(f"FAIL: fixed-vs-ref MEAN row delta {fixed_mean:.4f} m exceeds tol "
             f"{args.row_mean_tol} m -- a systematic FOV mismatch across rows.")
        ok = False

    # Geometry-changed evidence: fixed must render the real ~56 deg (NOT the old
    # ~71 deg), and control must still render ~71 deg.
    if not (abs(vfov_fixed - 56.4) < 2.0):
        emit(f"FAIL: measured fixed VFOV {vfov_fixed:.2f} deg is not the real 56.4 deg "
             "-- the 80x45 render did not land the target FOV.")
        ok = False
    if not (vfov_control - vfov_fixed > 10.0):
        emit(f"FAIL: fixed ({vfov_fixed:.2f}) and control ({vfov_control:.2f}) VFOVs did not "
             "diverge -- the resolution change did not move the geometry as expected.")
        ok = False

    emit("")
    emit(f"RESULT: {'PASS' if ok else 'FAIL'}")

    with open(args.report, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    sys.stdout.flush()

    simulation_app.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
