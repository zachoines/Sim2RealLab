#!/usr/bin/env python3
"""Measure the throughput cost of raising the physics rate (sim.dt).

The high-yaw-rate chassis bounce is removed by halving the timestep
(120 -> 240 Hz). Doing that on the shared contract means decimation 4 -> 8
(control rate fixed at 30 Hz, dataset unchanged), which doubles the PhysX
substeps per control step while leaving render untouched. This script
measures what that does to end-to-end control-step throughput in the real
env, and tracks chassis-z excursion so the bounce (and its removal at the
higher rate) is visible in the same run.

Run the same env at --sim-hz 120 then 240 and compare. RLNoCam isolates the
physics-doubling hit (negligible render); RLDepth shows the render-diluted
hit closer to capture.

Usage (from repo root, after `source env_setup.sh`):
    $ISAACLAB -p Scripts/sim_rate_profile.py --env Isaac-Strafer-Nav-RLNoCam-v0 --sim-hz 120
    $ISAACLAB -p Scripts/sim_rate_profile.py --env Isaac-Strafer-Nav-RLNoCam-v0 --sim-hz 240
    $ISAACLAB -p Scripts/sim_rate_profile.py --env Isaac-Strafer-Nav-RLDepth-Real-v0 --sim-hz 120 --video
    $ISAACLAB -p Scripts/sim_rate_profile.py --env Isaac-Strafer-Nav-RLDepth-Real-v0 --sim-hz 240 --video
"""

import argparse


def main():
    parser = argparse.ArgumentParser(description="Physics-rate throughput profile")
    parser.add_argument("--env", type=str, default="Isaac-Strafer-Nav-RLNoCam-v0")
    parser.add_argument("--sim-hz", type=float, default=120.0,
                        help="Physics substep rate; decimation = sim_hz/control_hz")
    parser.add_argument("--control-hz", type=float, default=30.0)
    parser.add_argument("--omega", type=float, default=1.0,
                        help="Yaw command (normalized stick) to drive max spin")
    parser.add_argument("--warmup", type=int, default=60,
                        help="Untimed control steps before measuring")
    parser.add_argument("--steps", type=int, default=200,
                        help="Timed control steps")
    parser.add_argument("--video", action="store_true", default=False)
    parser.add_argument("--video-dir", type=str, default="/tmp/sim_rate_videos")

    from isaaclab.app import AppLauncher
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if "NoCam" not in args.env or args.video:
        args.enable_cameras = True

    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import os
    import time
    import gymnasium as gym
    import torch
    import warp as wp
    from isaaclab_tasks.utils import parse_env_cfg
    import strafer_lab  # noqa: F401  (registers envs)

    def to_t(x):
        return wp.to_torch(x) if isinstance(x, wp.array) else x

    device = args.device if hasattr(args, "device") else "cuda:0"
    env_cfg = parse_env_cfg(args.env, device=device, num_envs=1)

    # Override the physics rate, holding the control rate (and therefore the
    # dataset) fixed: decimation = sim_hz / control_hz, render once per control
    # step (render_interval = decimation) so render cost per step is unchanged.
    decim = int(round(args.sim_hz / args.control_hz))
    env_cfg.sim.dt = 1.0 / args.sim_hz
    env_cfg.decimation = decim
    if hasattr(env_cfg.sim, "render_interval"):
        env_cfg.sim.render_interval = decim
    print(f"[rate] env={args.env} sim_hz={args.sim_hz:.0f} dt={env_cfg.sim.dt:.6f} "
          f"decimation={decim} control_hz={args.control_hz:.0f}")

    render_mode = "rgb_array" if args.video else None
    if args.video:
        from isaaclab.envs.common import ViewerCfg
        env_cfg.viewer = ViewerCfg(eye=(1.2, 1.2, 0.8), lookat=(0.0, 0.0, 0.05),
                                   origin_type="env", env_index=0,
                                   resolution=(960, 720))
    env = gym.make(args.env, cfg=env_cfg, render_mode=render_mode)
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        tag = f"{args.env.split('-')[-2]}_{int(args.sim_hz)}hz"
        env = gym.wrappers.RecordVideo(
            env, video_folder=args.video_dir, name_prefix=tag,
            step_trigger=lambda s: s == 0,
            video_length=args.warmup + args.steps, disable_logger=True)

    obs, _ = env.reset()
    if args.video:
        import numpy as np
        from isaacsim.core.utils.viewports import set_camera_view as scv
        u = env.unwrapped
        o = u.scene.env_origins[0].cpu().numpy()
        scv(eye=(o + np.array(env_cfg.viewer.eye)).tolist(),
            target=(o + np.array(env_cfg.viewer.lookat)).tolist(),
            camera_prim_path=env_cfg.viewer.cam_prim_path)

    robot = env.unwrapped.scene["robot"]
    action = torch.tensor([[0.0, 0.0, args.omega]], device=device)

    def chassis_z():
        return float(to_t(robot.data.root_pos_w)[0, 2])

    # Warmup (spin up to steady state; untimed).
    for _ in range(args.warmup):
        env.step(action)

    zmin = zmax = chassis_z()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        env.step(action)
        z = chassis_z()
        zmin = min(zmin, z); zmax = max(zmax, z)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0

    ms = dt / args.steps * 1000.0
    fps = args.steps / dt
    print("\n========== RATE PROFILE ==========")
    print(f"env             {args.env}")
    print(f"physics rate    {args.sim_hz:.0f} Hz   (decimation {decim}, "
          f"{decim} substeps/control step)")
    print(f"control steps   {args.steps} timed ({args.warmup} warmup)")
    print(f"per control step {ms:.1f} ms")
    print(f"throughput      {fps:.2f} control-steps/s")
    print(f"chassis-z p2p   {(zmax - zmin) * 1000:.1f} mm  (bounce indicator)")
    print("==================================")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
