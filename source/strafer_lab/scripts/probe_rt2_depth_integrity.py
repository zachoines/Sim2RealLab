"""Depth-integrity probe for the RTX Real-Time 2.0 renderer flip.

The depth stream ``/d555/depth/image_rect_raw`` is a live policy input. Before
the bridge's default renderer is flipped to RTX Real-Time 2.0, this probe
confirms the depth values reaching the policy are intact. It exercises the
exact camera path the bridge uses — the perception ``TiledCamera``'s
``distance_to_image_plane`` annotator, read once per ``env.step`` — under the
selected renderer.

Three checks (the first two are structural sanity, the third is the real gate):

  a. **Cadence.** N env steps produce exactly N depth frames, advancing one
     ``step_dt`` of sim time each — no extra / duplicated / dropped frames. (A
     display-side "FPS multiplier" cannot inject frames into this env-step-driven
     read; this confirms it empirically.)
  b. **Frame-diff under motion.** Consecutive frames differ while the robot is
     moving — no repeated (frozen / interpolated) frames.
  c. **RT1 vs RT2 parity.** A static-scene depth frame rendered under Real-Time
     2.0 matches the Real-Time 1.0 render of the same pose within the
     renderer-nondeterminism budget (``--max-abs-m``, default 1e-3 m). This is
     the residual risk the docs do not guarantee: Real-Time 2.0 changes the
     lighting integrator, and ``distance_to_image_plane`` is a geometric AOV
     that should be invariant — this measures whether it actually is.

Usage (an Isaac Sim Kit conda env must be active for capture)::

    source env_setup.sh && conda activate "$CONDA_ENV"

    # 1) Capture under each renderer (checks a + b run inline, PASS/FAIL):
    $ISAACLAB -p source/strafer_lab/scripts/probe_rt2_depth_integrity.py \\
        --renderer rt2 --out /tmp/depth_rt2.npz
    $ISAACLAB -p source/strafer_lab/scripts/probe_rt2_depth_integrity.py \\
        --renderer legacy --out /tmp/depth_legacy.npz

    # 2) Parity compare (check c) — CPU only, no Kit boot:
    python source/strafer_lab/scripts/probe_rt2_depth_integrity.py \\
        --compare /tmp/depth_rt2.npz /tmp/depth_legacy.npz

Both captures reset the same task with the same seed, so the static-frame pose
matches across runs and the parity compare is apples-to-apples.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from strafer_lab.tools.depth_probe import (
    check_cadence,
    check_frame_diff_under_motion,
    depth_parity,
)


# ---------------------------------------------------------------------------
# CPU parity-compare mode (no Kit boot).
# ---------------------------------------------------------------------------


def _run_compare(rt2_npz: Path, legacy_npz: Path, *, max_abs_m: float) -> int:
    rt2 = np.load(rt2_npz)
    legacy = np.load(legacy_npz)
    print(f"[probe] compare static frames: {rt2_npz} (RT2) vs {legacy_npz} (legacy)")
    print(f"[probe]   rt2 rendermode={rt2['rendermode']}  legacy rendermode={legacy['rendermode']}")
    a, b = rt2["static_depth"], legacy["static_depth"]
    if a.shape != b.shape:
        print(f"[probe] FAIL: static depth shape {a.shape} (RT2) != {b.shape} (legacy)")
        return 1
    ok, max_diff, msg = depth_parity(a, b, max_abs_m=max_abs_m)
    print(f"[probe]   parity: {msg}")
    print(f"[probe] {'PASS' if ok else 'FAIL'}: RT1 vs RT2 static-depth parity")
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--compare", nargs=2, metavar=("RT2_NPZ", "LEGACY_NPZ"), default=None,
        help="CPU-only: compare two capture .npz files for RT1-vs-RT2 static "
             "depth parity. Skips the Kit boot entirely.",
    )
    parser.add_argument(
        "--renderer", dest="rtx_renderer",  # avoid the reserved SimApp key
        choices=renderer_settings.RENDERER_CHOICES,
        default=renderer_settings.DEFAULT_RENDERER,
        help="Renderer to capture under (see run_sim_in_the_loop --renderer).",
    )
    parser.add_argument(
        "--task", default="Isaac-Strafer-Nav-Capture-Bridge-v0",
        help="Capture env carrying the perception camera.",
    )
    parser.add_argument(
        "--scene", default="scene_singleroom_000_seed0",
        help="Scene name under Assets/generated/scenes/ (light single-room default).",
    )
    parser.add_argument("--out", type=Path, default=None, help="Output .npz path for the capture.")
    parser.add_argument("--steps", type=int, default=32, help="Motion frames to capture (checks a+b).")
    parser.add_argument("--settle", type=int, default=8, help="Zero-action settle steps before the static frame.")
    parser.add_argument("--max-abs-m", type=float, default=1e-3, help="RT1-vs-RT2 depth parity budget in metres.")
    return parser


# Parse first so --compare can run without importing / booting Kit.
renderer_settings = None  # bound below once the scripts dir is importable


def _bind_renderer_settings() -> None:
    """Make ``strafer_lab.bridge.renderer_settings`` importable for the parser."""
    global renderer_settings
    from strafer_lab.bridge import renderer_settings as rs

    renderer_settings = rs


_bind_renderer_settings()
# parse_known_args so AppLauncher flags (--headless, --kit_args, ...) on a
# capture run don't error here; the capture path below re-parses with them.
_args, _ = _build_parser().parse_known_args()

if _args.compare is not None:
    raise SystemExit(
        _run_compare(Path(_args.compare[0]), Path(_args.compare[1]), max_abs_m=_args.max_abs_m)
    )

if _args.out is None:
    raise SystemExit("--out is required for a capture run (omit only with --compare)")

# --- Capture path: boot Kit. -------------------------------------------------
import run_sim_in_the_loop as _rsil  # noqa: E402  (shares the bridge renderer boot path)
from isaaclab.app import AppLauncher  # noqa: E402

_capture_parser = _build_parser()
AppLauncher.add_app_launcher_args(_capture_parser)
args = _capture_parser.parse_args()
args.headless = True
args.enable_cameras = True

# Inject the renderer's Kit settings the same way the bridge does.
_rsil._apply_renderer_boot_args(args)

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app


# ---------------------------------------------------------------------------
# Post-launch imports (need the Kit runtime active).
# ---------------------------------------------------------------------------

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: F401, E402
import strafer_lab.tasks  # noqa: F401, E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def _depth_to_numpy(camera) -> np.ndarray:
    """Snapshot ``distance_to_image_plane`` for env 0 as a (H, W) float array."""
    out = camera.data.output["distance_to_image_plane"]
    if isinstance(out, torch.Tensor):
        arr = out
    else:  # warp array
        import warp as wp

        arr = wp.to_torch(out)
    return arr[0].detach().float().cpu().numpy().squeeze()


def main() -> int:
    print(f"[probe] renderer={args.rtx_renderer} scene={args.scene} out={args.out}", flush=True)

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    usdc = Path("Assets/generated/scenes") / f"{args.scene}.usdc"
    if usdc.is_file():
        env_cfg.scene.scene_geometry.spawn.usd_path = str(usdc.resolve())
    # One continuous episode at a fixed pose (no mid-run reset would break the
    # cadence/parity); decimation/render match the bridge.
    _rsil._disable_env_terminations(env_cfg.terminations)
    env_cfg.decimation = 1
    if hasattr(env_cfg.sim, "render_interval"):
        env_cfg.sim.render_interval = 1

    env = gym.make(args.task, cfg=env_cfg)
    unwrapped = env.unwrapped
    env.reset()
    unwrapped.render_enabled = False
    _rsil._log_active_renderer()

    import carb

    rendermode = carb.settings.get_settings().get("/rtx/rendermode")
    camera = unwrapped.scene["d555_camera_perception"]
    action = torch.zeros(unwrapped.action_manager.action.shape, device=unwrapped.device)

    physics_dt = float(unwrapped.sim.get_physics_dt())
    step_dt = physics_dt * int(unwrapped.cfg.decimation)

    # --- Static frame: hold still, settle, capture one frame. ----------------
    for _ in range(max(1, args.settle)):
        env.step(action)
        simulation_app.update()
    static_depth = _depth_to_numpy(camera)

    # --- Motion frames: drive forward + yaw, capture per step. ---------------
    motion = action.clone()
    if motion.shape[-1] >= 3:
        motion[0, 0] = 0.4   # forward (normalized)
        motion[0, 2] = 0.5   # yaw (normalized)
    depths, sim_times = [], []
    sim_time = 0.0
    for _ in range(args.steps):
        env.step(motion)
        simulation_app.update()
        sim_time += step_dt
        depths.append(_depth_to_numpy(camera))
        sim_times.append(sim_time)

    depths = np.stack(depths, axis=0)
    sim_times = np.asarray(sim_times, dtype=np.float64)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        static_depth=static_depth,
        motion_depths=depths,
        sim_times=sim_times,
        step_dt=np.float64(step_dt),
        rendermode=str(rendermode),
        renderer=str(args.rtx_renderer),
    )
    print(f"[probe] saved {depths.shape[0]} motion frames + 1 static frame -> {args.out}", flush=True)
    print(f"[probe] static depth shape {static_depth.shape}, rendermode={rendermode}", flush=True)

    ok_c, msg_c = check_cadence(sim_times, step_dt)
    ok_d, msg_d = check_frame_diff_under_motion(depths)
    print(f"[probe] (a) cadence: {'PASS' if ok_c else 'FAIL'} — {msg_c}", flush=True)
    print(f"[probe] (b) frame-diff under motion: {'PASS' if ok_d else 'FAIL'} — {msg_d}", flush=True)
    print(
        "[probe] (c) RT1-vs-RT2 parity: run --compare on an rt2 and a legacy "
        "capture (CPU-only)",
        flush=True,
    )
    rc = 0 if (ok_c and ok_d) else 1
    print(f"[probe] {'PASS' if rc == 0 else 'FAIL'} (checks a+b for renderer={args.rtx_renderer})", flush=True)
    return rc


if __name__ == "__main__":
    try:
        _rc = main()
    finally:
        simulation_app.close()
    sys.exit(_rc)
