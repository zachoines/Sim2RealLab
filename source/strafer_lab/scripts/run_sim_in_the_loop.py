"""Boot the Strafer perception env with the ROS2 bridge wired up.

This is the DGX-side entry point for sim-in-the-loop runs. It launches
Isaac Sim via ``AppLauncher``, instantiates a perception-camera task at
``num_envs=1``, enables the bundled ``isaacsim.ros2.bridge`` extension,
builds the Strafer OmniGraph (cameras, odom, TF, /cmd_vel subscribe),
and then runs in one of two modes:

  - ``--mode bridge``: drive the env step loop reading /cmd_vel from the
    bridge and injecting it into the action tensor. Used for manual ops
    where the Jetson side publishes /cmd_vel directly (rqt teleop, the
    real Nav2 stack on the LAN, ``ros2 topic pub`` smoke checks).

  - ``--mode harness``: also instantiate the sim-in-the-loop harness,
    walk every mission emitted by ``MissionGenerator`` for the configured
    scene, submit each one to the Jetson autonomy executor over the
    ``execute_mission`` action, and capture reachability-labeled frames
    into a ``frames.jsonl`` dataset.

The Jetson at ``STRAFER_JETSON_HOST`` consumes the same real-robot
topics in either mode:

    /d555/color/image_raw          sensor_msgs/Image
    /d555/color/camera_info        sensor_msgs/CameraInfo
    /d555/depth/image_rect_raw     sensor_msgs/Image
    /d555/depth/camera_info        sensor_msgs/CameraInfo
    /strafer/odom                  nav_msgs/Odometry
    tf                             odom→base_link, base_link→d555_link
    /cmd_vel                       geometry_msgs/Twist  (SUBSCRIBE)

Usage:

    source env_setup.sh   # loads ROS_DOMAIN_ID, RMW_IMPLEMENTATION, LD_PRELOAD

    # Manual / Nav2 mode:
    isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py

    # Reachability-labeled dataset capture:
    isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py \\
        --mode harness \\
        --scene-metadata Assets/generated/scenes/kitchen_01/scene_metadata.json \\
        --scene-usd Assets/generated/scenes/kitchen_01/scene.usdc \\
        --output data/sim_in_the_loop/kitchen_01

Verify from another shell (Jetson or DGX):

    ros2 topic list
    ros2 topic hz /d555/color/image_raw
    ros2 topic pub /cmd_vel geometry_msgs/msg/Twist '{linear: {x: 0.3}}'
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path
from typing import Any

from isaaclab.app import AppLauncher
from strafer_shared.constants import MAX_ANGULAR_VEL, MAX_LINEAR_VEL


def _clamp_unit(value: float) -> float:
    """Clamp ``value`` to ``[-1, 1]`` (the action term's normalized contract)."""
    return max(-1.0, min(1.0, value))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("bridge", "harness"),
        default="bridge",
        help="Run mode: 'bridge' drives /cmd_vel into the env (default); "
             "'harness' walks scene_metadata.json missions through the "
             "Jetson's execute_mission action and writes a reachability dataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Strafer-Nav-Real-InfinigenPerception-Play-v0",
        help="Isaac Lab task carrying the 640x360 perception camera. "
        "num_envs is forced to 1 for sim-in-the-loop.",
    )
    parser.add_argument(
        "--graph-path",
        type=str,
        default="/World/ROS2Bridge",
        help="USD path for the OmniGraph prim the bridge is built into.",
    )
    parser.add_argument(
        "--cmd-vel-timeout",
        type=float,
        default=0.5,
        help="Bridge mode only: seconds with no /cmd_vel before holding still.",
    )
    parser.add_argument(
        "--scene-usd",
        type=Path,
        default=None,
        help="Override the env's default scene USD with this path.",
    )

    # Harness-mode args
    parser.add_argument(
        "--scene-metadata",
        type=Path,
        default=None,
        help="Harness mode: path to scene_metadata.json describing the targets.",
    )
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Harness mode: scene_name for frames.jsonl. Defaults to the "
             "parent dir name of --scene-metadata.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Harness mode: output dir for the reachability dataset (one "
             "episode_NNNN/ per mission).",
    )
    parser.add_argument(
        "--max-missions",
        type=int,
        default=None,
        help="Harness mode: cap on missions to run from the generator.",
    )
    parser.add_argument(
        "--allowed-labels",
        nargs="*",
        default=None,
        help="Harness mode: only target objects whose label is in this list.",
    )
    parser.add_argument(
        "--blocked-labels",
        nargs="*",
        default=("wall", "floor", "ceiling"),
        help="Harness mode: skip targets whose label is in this list.",
    )
    parser.add_argument(
        "--mission-timeout-s",
        type=float,
        default=60.0,
        help="Harness mode: seconds before the harness cancels a stuck mission.",
    )
    parser.add_argument(
        "--capture-every-n-steps",
        type=int,
        default=5,
        help="Harness mode: env-step interval for frame capture during nav.",
    )

    # Bridge / harness overrides on the RL env cfg. Defaults match the
    # integration-test workflow: fixed spawn yaw, no episode-timeout
    # auto-reset mid-run. The RL training wrapper (train_strafer_navigation.py)
    # does not call these, so training behavior is unchanged.
    parser.add_argument(
        "--pin-yaw",
        type=float,
        default=0.0,
        help="Spawn yaw in radians. 0.0 (default) faces +X. Pass None is "
             "not supported; to keep the env's random yaw range, pass the "
             "env cfg directly (no sim-in-the-loop override).",
    )
    parser.add_argument(
        "--enable-episode-timeout",
        action="store_true",
        help="Keep the env's default episode-length termination. By default "
             "sim-in-the-loop disables it so the robot is not teleported "
             "mid-mission (or mid-bridge-run) by the training-scale timeout.",
    )
    parser.add_argument(
        "--decimation",
        type=int,
        default=1,
        help="Physics ticks per env.step. RL training defaults to 4 to "
             "compress sim-time (motor dynamics / command delay are tuned "
             "for that), but for sim-in-the-loop we want high wall-clock "
             "publish rate, not sim-time compression. Decimation=1 runs "
             "one physics tick per env.step, which on DGX Spark gets the "
             "perception env to ~29 steps/s vs ~8 at the training default. "
             "Raise back to 4 only if you need training-equivalent dynamics.",
    )
    parser.add_argument(
        "--render-interval",
        type=int,
        default=1,
        help="Override env_cfg.sim.render_interval (physics ticks per render). "
             "Default 1 (render every tick): on DGX Spark, env.step is "
             "render-paced, and lower render_interval means higher tick rate "
             "(counterintuitive but measured — per-render wall time grows "
             "with render_interval, so fewer-but-slower renders is a net loss). "
             "Training defaults to 4 and is untouched.",
    )
    parser.add_argument(
        "--no-camera-bridge",
        action="store_true",
        help="Skip wiring camera streams into the bridge OmniGraph. "
             "Useful for a cameras-off perf baseline (paired with "
             "--task Isaac-Strafer-Nav-Real-NoCam-Play-v0). Default off; "
             "the standard bridge publishes color + depth.",
    )
    parser.add_argument(
        "--camera-frame-skip",
        type=int,
        default=3,
        help="Number of simulation frames ROS2CameraHelper skips between "
             "each Image/CameraInfo publish. 0 = publish every tick "
             "(matches pre-optimization behavior). Default 3 = publish "
             "once every 4 physics ticks, matching sim.render_interval so "
             "we don't serialize + push duplicate frames when the "
             "underlying render has not advanced.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Phase-level timing harness. Wraps sim.step / sim.render plus "
             "the bridge loop's outer phases (cmd_vel read, env.step, "
             "publish, kit.update) and prints rolling p50/p99 every "
             "--profile-interval seconds. Use to attribute the per-step "
             "wall budget across PhysX vs Kit vs bridge.",
    )
    parser.add_argument(
        "--profile-interval",
        type=float,
        default=10.0,
        help="Seconds between --profile reports (default 10).",
    )
    parser.add_argument(
        "--profile-window",
        type=int,
        default=200,
        help="Rolling sample window for p50/p99 in --profile (default 200 steps).",
    )

    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Phase-level profiling harness (--profile)
# ---------------------------------------------------------------------------


class _PhaseProfiler:
    """Roll p50/p99 wall-time samples for named phases, print periodically.

    Designed for the bridge mainloop. Outer-loop phases are recorded with
    :meth:`time` as a context manager; ``env.step`` internals (PhysX vs
    Kit-render) are captured by monkey-patching ``sim.step`` /
    ``sim.render`` once at install time so the per-decimation-tick timing
    is attributed correctly.
    """

    def __init__(self, *, window: int, period_s: float) -> None:
        import collections
        self._collections = collections
        self._samples: dict[str, "collections.deque[float]"] = {}
        self._window = max(1, int(window))
        self._period_s = float(period_s)
        self._last_print = time.monotonic()
        self._step_count = 0

    def record(self, phase: str, dt_s: float) -> None:
        bucket = self._samples.get(phase)
        if bucket is None:
            bucket = self._collections.deque(maxlen=self._window)
            self._samples[phase] = bucket
        bucket.append(dt_s * 1000.0)  # store as milliseconds

    @property
    def time(self):  # context manager factory
        import contextlib

        @contextlib.contextmanager
        def _ctx(phase: str):
            t0 = time.perf_counter()
            try:
                yield
            finally:
                self.record(phase, time.perf_counter() - t0)
        return _ctx

    def install_env_step_hooks(self, sim_context: Any) -> None:
        """Wrap sim.step / sim.render to attribute their per-tick wall time."""
        orig_step = sim_context.step
        orig_render = sim_context.render

        def _timed_step(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig_step(*args, **kwargs)
            finally:
                self.record("env.step :: sim.step (PhysX)", time.perf_counter() - t0)

        def _timed_render(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return orig_render(*args, **kwargs)
            finally:
                self.record("env.step :: sim.render (Kit)", time.perf_counter() - t0)

        sim_context.step = _timed_step
        sim_context.render = _timed_render

    def tick(self) -> None:
        """Bump the step counter and emit a report if the period has elapsed."""
        self._step_count += 1
        now = time.monotonic()
        if now - self._last_print < self._period_s:
            return
        self._last_print = now
        self._report(now)

    def _report(self, now: float) -> None:
        import statistics
        if not self._samples:
            return
        # Sort phases by p50 desc so the dominant cost is always at the top.
        rows: list[tuple[str, float, float, int]] = []
        for phase, bucket in self._samples.items():
            if not bucket:
                continue
            vals = sorted(bucket)
            n = len(vals)
            p50 = vals[n // 2]
            p99 = vals[min(n - 1, max(0, int(round(0.99 * (n - 1)))))]
            rows.append((phase, p50, p99, n))
        rows.sort(key=lambda r: r[1], reverse=True)

        print(f"\n[profile] step={self._step_count}  window={self._window}  unit=ms")
        print(f"  {'phase':<42s} {'p50':>8s} {'p99':>8s} {'n':>5s}")
        for phase, p50, p99, n in rows:
            print(f"  {phase:<42s} {p50:8.2f} {p99:8.2f} {n:5d}")
        # Hint at where to look next.
        top = rows[0][0] if rows else "?"
        print(f"  -> dominant: {top}")


def _apply_sim_in_the_loop_overrides(
    env_cfg,
    *,
    pin_yaw: float,
    disable_timeout: bool,
    decimation: int,
    render_interval: int | None,
) -> None:
    """Adapt an RL env cfg for sim-in-the-loop use.

    Three overrides, all safe to apply to any Strafer nav env cfg:

    - **Pin spawn yaw** to a fixed value. The reset event either exposes
      ``yaw_range`` (Infinigen floor-spawn variant) or
      ``pose_range["yaw"]`` (non-Infinigen variant); we patch whichever
      is present.
    - **Disable the episode-length termination** so the env does not
      auto-reset on a training-scale timeout. Collision and flipped
      terminations remain active — we still want those to fire if the
      simulation goes off the rails.
    - **Override physics decimation.** RL training runs with
      ``decimation=4`` so motor-dynamics and command-delay are tuned
      against a 33 ms env-step. Bridge mode needs high wall-clock
      publish rate, not sim-time compression; running at ``decimation=1``
      cuts wall-time per step by ~3×.
    """
    reset_term = getattr(env_cfg.events, "reset_robot", None)
    if reset_term is not None and getattr(reset_term, "params", None):
        params = reset_term.params
        if "yaw_range" in params:
            params["yaw_range"] = (pin_yaw, pin_yaw)
            print(f"[sim_in_the_loop] reset yaw pinned to {pin_yaw:.3f} rad (yaw_range)")
        elif "pose_range" in params and isinstance(params["pose_range"], dict):
            params["pose_range"]["yaw"] = (pin_yaw, pin_yaw)
            print(f"[sim_in_the_loop] reset yaw pinned to {pin_yaw:.3f} rad (pose_range)")

    if disable_timeout:
        if getattr(env_cfg.terminations, "time_out", None) is not None:
            env_cfg.terminations.time_out = None
            print("[sim_in_the_loop] episode-length time_out termination disabled")

    if decimation > 0 and decimation != env_cfg.decimation:
        prev = env_cfg.decimation
        env_cfg.decimation = decimation
        print(f"[sim_in_the_loop] decimation {prev} -> {decimation}")

    if render_interval is not None and render_interval > 0 and render_interval != env_cfg.sim.render_interval:
        prev = env_cfg.sim.render_interval
        env_cfg.sim.render_interval = render_interval
        print(f"[sim_in_the_loop] sim.render_interval {prev} -> {render_interval}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()

    if args.mode == "harness":
        _validate_harness_args(args)

    # AppLauncher must boot Isaac Sim before any omni.* / strafer_lab.tasks imports.
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym

    import isaaclab_tasks  # noqa: F401
    import strafer_lab.tasks  # noqa: F401

    from isaacsim.core.utils.extensions import enable_extension
    from isaaclab_tasks.utils import parse_env_cfg

    from strafer_lab.bridge.config import build_default_bridge_config
    from strafer_lab.bridge.graph import build_bridge_graph

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    # Enable the bridge extension before any OmniGraph nodes are created.
    # ``isaacsim.ros2.bridge`` registers its node types with OmniGraph at
    # extension-enable time; attempting ``og.Controller.edit(...)`` with
    # ``isaacsim.ros2.bridge.*`` before the extension is hot throws a
    # "node type not registered" error.
    enable_extension("isaacsim.ros2.bridge")
    simulation_app.update()

    env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=1)
    if args.scene_usd is not None:
        env_cfg.scene.scene_geometry.spawn.usd_path = str(args.scene_usd.resolve())
        print(f"[sim_in_the_loop] scene USD override → {env_cfg.scene.scene_geometry.spawn.usd_path}")

    _apply_sim_in_the_loop_overrides(
        env_cfg,
        pin_yaw=args.pin_yaw,
        disable_timeout=not args.enable_episode_timeout,
        decimation=args.decimation,
        render_interval=args.render_interval,
    )

    env = gym.make(args.task, cfg=env_cfg)

    config = build_default_bridge_config(
        graph_path=args.graph_path,
        camera_frame_skip=args.camera_frame_skip,
    )
    # Telemetry (/clock, /odom, TF, /cmd_vel) is handled by a Python
    # rclpy thread; only the cameras stay on the OmniGraph because they
    # are bound to Isaac Sim render products.
    build_bridge_graph(config, skip_cameras=args.no_camera_bridge, skip_telemetry=True)
    print(f"[sim_in_the_loop] bridge graph built at {args.graph_path}")
    if args.no_camera_bridge:
        print("[sim_in_the_loop] camera streams skipped (--no-camera-bridge)")
    else:
        print(f"[sim_in_the_loop] camera_frame_skip = {config.camera_frame_skip}")
        print(f"[sim_in_the_loop] color camera prim={config.color_camera.camera_prim_path}")
    print(f"[sim_in_the_loop] chassis_prim={config.chassis_prim_path}")

    if args.mode == "bridge":
        _run_bridge_mode(simulation_app, env, args, config)
    else:
        _run_harness_mode(simulation_app, env, args)

    env.close()
    simulation_app.close()


# ---------------------------------------------------------------------------
# Mode: bridge (drive /cmd_vel into the env)
# ---------------------------------------------------------------------------


def _run_bridge_mode(simulation_app, env, args, config) -> None:
    import torch

    from strafer_lab.bridge.async_publisher import StraferAsyncPublisher

    unwrapped = env.unwrapped
    env.reset()

    # Eliminate the redundant Kit pump under --viz kit. Without this, env.step
    # calls sim.render() → KitVisualizer.step → app.update(playSimulations=False)
    # to refresh the editor viewport, then this loop calls simulation_app.update()
    # below with playSimulations=True to fire OnPlaybackTick and drive the camera
    # OmniGraph publish chain. Both pumps RTX-render the editor viewport, paying
    # ~80 ms of duplicated work per loop.
    #
    # Setting render_enabled=False makes env.step() invoke sim.render() with
    # skip_app_pumping=True, which short-circuits KitVisualizer (no app.update).
    # The simulation_app.update() below then becomes the SOLE Kit pump per loop
    # and drives BOTH the viewport refresh AND the camera OmniGraph in one pass.
    # No-op in headless mode (no Kit visualizer registered).
    unwrapped.render_enabled = False

    action_shape = unwrapped.action_manager.action.shape
    print(f"[sim_in_the_loop] action tensor shape = {tuple(action_shape)}")

    device = unwrapped.device
    zero_action = torch.zeros(action_shape, device=device)

    publisher = StraferAsyncPublisher(
        robot=unwrapped.scene["robot"],
        clock_topic=config.clock_topic,
        odom_topic=config.odom_topic,
        cmd_vel_topic=config.cmd_vel_topic,
        odom_frame_id=config.odom_frame_id,
        base_frame_id=config.base_frame_id,
    )
    print("[sim_in_the_loop] async publisher up: /clock, /odom, TF, /cmd_vel")

    # Track sim time in the bridge loop so /clock advances in lock-step
    # with env.step. Matches IsaacReadSimulationTime's source: seconds
    # elapsed since play-start = physics_dt * decimation * iter_count.
    physics_dt = unwrapped.sim.get_physics_dt()
    step_dt = physics_dt * unwrapped.cfg.decimation
    sim_time_s = 0.0

    profiler: _PhaseProfiler | None = None
    if args.profile:
        profiler = _PhaseProfiler(
            window=args.profile_window, period_s=args.profile_interval
        )
        profiler.install_env_step_hooks(unwrapped.sim)
        print(
            f"[sim_in_the_loop] --profile on: window={args.profile_window} steps, "
            f"reporting every {args.profile_interval:.1f}s"
        )

    last_cmd_time = time.monotonic()
    try:
        while simulation_app.is_running():
            if profiler is not None:
                with profiler.time("loop :: cmd_vel read"):
                    linear, angular = publisher.get_cmd_vel()
            else:
                linear, angular = publisher.get_cmd_vel()
            vx, vy = linear[0], linear[1]
            wz = angular[2]

            now = time.monotonic()
            if any(abs(v) > 1e-6 for v in (vx, vy, wz)):
                last_cmd_time = now
                action = zero_action.clone()
                if action.shape[-1] >= 3:
                    # /cmd_vel arrives in physical units (m/s, rad/s);
                    # MecanumWheelAction.process_actions clamps to [-1, 1]
                    # and scales by [MAX_LINEAR_VEL, MAX_LINEAR_VEL,
                    # MAX_ANGULAR_VEL]. Match that contract here so a
                    # Nav2 command of 0.5 m/s produces 0.5 m/s of body
                    # velocity, not 0.78 m/s. Mirrors the normalization
                    # in IsaacLabEnvAdapter._build_action used by the
                    # harness path.
                    action[0, 0] = _clamp_unit(float(vx) / MAX_LINEAR_VEL)
                    action[0, 1] = _clamp_unit(float(vy) / MAX_LINEAR_VEL)
                    action[0, 2] = _clamp_unit(float(wz) / MAX_ANGULAR_VEL)
            elif now - last_cmd_time > args.cmd_vel_timeout:
                action = zero_action
            else:
                action = zero_action

            if profiler is not None:
                with profiler.time("loop :: env.step (total)"):
                    env.step(action)
                sim_time_s += step_dt
                with profiler.time("loop :: publish_state"):
                    publisher.publish_state(sim_time_s)
                with profiler.time("loop :: simulation_app.update"):
                    simulation_app.update()
                profiler.tick()
            else:
                env.step(action)
                sim_time_s += step_dt
                publisher.publish_state(sim_time_s)
                # Sole Kit pump per loop iteration (render_enabled=False above
                # disables KitVisualizer's pump inside env.step). One app.update
                # per env.step refreshes the editor viewport, fires OnPlaybackTick,
                # and drives the camera OmniGraph publish chain in one pass.
                simulation_app.update()
    finally:
        publisher.shutdown()


# ---------------------------------------------------------------------------
# Mode: harness (walk MissionGenerator, capture reachability dataset)
# ---------------------------------------------------------------------------


def _run_harness_mode(simulation_app, env, args) -> None:
    from strafer_lab.sim_in_the_loop import (
        HarnessConfig,
        MissionGenerator,
        SimInTheLoopHarness,
    )
    from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter
    from strafer_lab.sim_in_the_loop.runtime_mission import Ros2MissionApi
    from strafer_lab.tools.perception_writer import PerceptionFrameWriter

    scene_name = args.scene_name or args.scene_metadata.parent.name

    generator = MissionGenerator.from_metadata_path(
        scene_metadata_path=args.scene_metadata,
        scene_name=scene_name,
        max_missions=args.max_missions,
        allowed_labels=args.allowed_labels,
        blocked_labels=tuple(args.blocked_labels or ()),
    )
    missions = list(generator)
    if not missions:
        print(f"[sim_in_the_loop] no missions to run for {scene_name}; exiting")
        return
    print(f"[sim_in_the_loop] {len(missions)} missions queued for {scene_name}")

    args.output.mkdir(parents=True, exist_ok=True)
    writer = PerceptionFrameWriter(output_root=args.output)

    env_adapter = IsaacLabEnvAdapter(
        env=env,
        graph_path=args.graph_path,
        scene_name=scene_name,
    )

    cfg = HarnessConfig(
        mission_timeout_s=args.mission_timeout_s,
        capture_every_n_steps=args.capture_every_n_steps,
        scene_type="infinigen_sim_in_the_loop",
    )

    with Ros2MissionApi() as mission_api:
        harness = SimInTheLoopHarness(
            env_adapter=env_adapter,
            mission_api=mission_api,
            writer=writer,
            config=cfg,
        )

        for spec in missions:
            if not simulation_app.is_running():
                print("[sim_in_the_loop] simulation app shut down; aborting run")
                break
            outcome = harness.run_one_mission(spec)
            print(
                f"[sim_in_the_loop] {spec.mission_id} reachable={outcome.reachability} "
                f"state={outcome.final_status.state} frames={outcome.frames_written} "
                f"elapsed={outcome.elapsed_s:.1f}s"
            )

    print(f"[sim_in_the_loop] writer stats: {writer.stats.to_dict()}")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_harness_args(args: argparse.Namespace) -> None:
    missing = []
    if args.scene_metadata is None:
        missing.append("--scene-metadata")
    if args.output is None:
        missing.append("--output")
    if missing:
        raise SystemExit(
            f"--mode harness requires {' and '.join(missing)} to be set"
        )
    if not args.scene_metadata.is_file():
        raise SystemExit(f"scene metadata not found: {args.scene_metadata}")


if __name__ == "__main__":
    main()
