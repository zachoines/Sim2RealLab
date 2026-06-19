"""Boot the Strafer perception env with the ROS2 bridge wired up.

This is the DGX-side entry point for sim-in-the-loop runs. It launches
Isaac Sim via ``AppLauncher``, instantiates a perception-camera task at
``num_envs=1``, enables the bundled ``isaacsim.ros2.bridge`` extension,
builds the residual bridge OmniGraph scaffolding (``OnPlaybackTick`` +
``ROS2Context``; the chassis telemetry and camera publishers all run on
Python rclpy threads now — see
``strafer_lab.bridge.async_publisher`` and
``strafer_lab.bridge.async_camera_publisher``), and then runs in one of
two modes:

  - ``--mode bridge``: drive the env step loop reading /cmd_vel from the
    bridge and injecting it into the action tensor. Used for manual ops
    where the Jetson side publishes /cmd_vel directly (rqt teleop, the
    real Nav2 stack on the LAN, ``ros2 topic pub`` smoke checks).

  - ``--mode harness``: also instantiate the sim-in-the-loop harness,
    walk a mission stream (every embedded scene target by
    default, or a curated ``mission_queue.yaml`` via ``--mission-queue``),
    submit each mission to the Jetson autonomy executor over the
    ``execute_mission`` action, and record one LeRobot v3 episode per
    mission via ``strafer_lab.tools.lerobot_writer.StraferLeRobotWriter``
    — frames carry the perception camera RGB + depth sidecars, the
    normalized ``/cmd_vel`` action, and (by default) the Replicator
    ``bbox_2d_tight`` detections columns.

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

    # LeRobot v3 dataset capture through the autonomy stack (preferred
    # entry point is capture.py --driver bridge, which dispatches here):
    isaaclab -p source/strafer_lab/scripts/run_sim_in_the_loop.py \\
        --mode harness --headless \\
        --scene-name kitchen_01 \\
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


# Consecutive whole-mission crashes in harness mode before the run aborts
# rather than burning the remaining queue against a dead executor. Not a
# CLI knob: it's a circuit-breaker for an unrecoverable peer, not a tuning
# parameter — a single executor crash should not abort, three in a row is
# a downed-stack signal.
_MAX_CONSECUTIVE_MISSION_FAILURES = 3


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
             "'harness' walks the scene's embedded missions through the "
             "Jetson's execute_mission action and writes a reachability dataset.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-Strafer-Nav-Capture-Bridge-v0",
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
        help="Override the env's default scene USD with this path. Its "
             "customData also supplies the harness mission targets.",
    )

    # Harness-mode args
    parser.add_argument(
        "--scene-name",
        type=str,
        default=None,
        help="Harness mode: scene_id recorded in per-episode metadata + the "
             "scene whose targets to walk. Resolves the scene USD at "
             "Assets/generated/scenes/<scene-name>.usdc (whose customData "
             "carries the targets) unless --scene-usd overrides it.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Harness mode: LeRobot dataset root. Auto-suffixed with a "
             "timestamp if it already exists (LeRobotDataset.create "
             "refuses to overwrite).",
    )
    parser.add_argument(
        "--mission-queue",
        type=Path,
        default=None,
        help="Harness mode: mission_queue.yaml to walk instead of "
             "enumerating the scene's embedded targets. The bridge "
             "dispatches each row's mission_text through the autonomy "
             "stack; planned_path is ignored (the Jetson planner emits "
             "its own).",
    )
    parser.add_argument(
        "--max-missions",
        type=int,
        default=None,
        help="Harness mode: cap on missions to run from the generator "
             "or queue.",
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
        help="Harness mode: env-step interval for frame capture during nav. "
             "The writer's fps is derived as 1 / (env step dt × this).",
    )
    parser.add_argument(
        "--sensors",
        type=str,
        default=None,
        help="Harness mode: per-session sensor stack as a comma-separated "
             "token list over rgb_full,depth_full,rgb_policy,depth_policy. "
             "The env renders and the writer records exactly this stack. "
             "Defaults to the task's own stack (the bridge capture env "
             "ships rgb_full,depth_full,depth_policy). rgb_full + "
             "depth_full are mandatory unless --no-camera-bridge: the "
             "Jetson stack navigates on those streams.",
    )
    parser.add_argument(
        "--detections",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Harness mode: record per-frame 2D detections from the "
             "perception camera's Replicator bounding-box annotator as "
             "first-class dataset columns. On by default; "
             "--no-detections shrinks the schema.",
    )
    parser.add_argument(
        "--detections-max",
        type=int,
        default=None,
        help="Harness mode: padded detection slots per frame. Defaults to "
             "the writer's standard slot count.",
    )
    parser.add_argument(
        "--inject-bad-grounding",
        choices=("off", "wrong_room", "wrong_instance", "wrong_object"),
        default="off",
        help="Harness mode: hard-negative goal perturbation. With "
             "--inject-bad-grounding-prob, a mission's dispatched goal is "
             "swapped to a wrong object while the recorded mission text "
             "keeps naming the original target. Modes: wrong_room (different "
             "room), wrong_instance (same label, same room), wrong_object "
             "(different label, same room). Per-episode metadata records "
             "requested vs actual mode; downstream filters must key off the "
             "actual mode.",
    )
    parser.add_argument(
        "--inject-bad-grounding-prob",
        type=float,
        default=0.3,
        help="Harness mode: per-mission perturbation probability when "
             "--inject-bad-grounding is enabled.",
    )
    parser.add_argument(
        "--injection-seed",
        type=int,
        default=0,
        help="Harness mode: RNG seed for the injection coin flips + "
             "candidate picks, so a capture run's injection sequence is "
             "reproducible.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help="Harness mode: LeRobot repo_id. Defaults to 'strafer/<scene>'.",
    )
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Harness mode: session identifier saved on every episode; "
             "defaults to a wall-clock timestamp.",
    )
    parser.add_argument(
        "--vcodec",
        type=str,
        default="h264",
        help="Harness mode: LeRobot v3 video codec. h264 is the "
             "broadest-availability choice on ARM64.",
    )
    parser.add_argument(
        "--cmd-vel-grace",
        type=float,
        default=30.0,
        help="Harness mode: once the executor has started driving, this "
             "many seconds of /cmd_vel silence discards the episode "
             "(connectivity loss / executor crash). Silence before the "
             "first command of a mission is governed by "
             "--mission-timeout-s instead. 0 disables.",
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
        help="Skip constructing the async camera publisher. Useful for a "
             "cameras-off perf baseline (paired with "
             "--task Isaac-Strafer-Nav-RLNoCam-Play-v0). Default off; "
             "the standard bridge publishes color + depth.",
    )
    parser.add_argument(
        "--camera-frame-skip",
        type=int,
        default=3,
        help="Number of bridge ticks the async camera publisher drops "
             "between each Image/CameraInfo publish. 0 = publish every "
             "tick (matches pre-optimization behavior). Default 3 = "
             "publish once every 4 physics ticks, matching "
             "sim.render_interval so we don't serialize + push duplicate "
             "frames when the underlying render has not advanced.",
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

    cameras_required: tuple[str, ...] = ()
    if args.mode == "harness":
        cameras_required = _resolve_harness_sensor_stack(env_cfg, args)

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
    # Both telemetry (/clock, /odom, TF, /cmd_vel) and cameras
    # (/d555/color/..., /d555/depth/...) run on Python rclpy threads now.
    # The OmniGraph this builds is just the OnPlaybackTick + ROS2Context
    # scaffolding that other Kit-bound bridge nodes may attach to later.
    build_bridge_graph(config)
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
        _run_harness_mode(simulation_app, env, args, config, cameras_required)

    env.close()
    simulation_app.close()


def _resolve_harness_sensor_stack(env_cfg, args) -> tuple[str, ...]:
    """Resolve + apply the harness capture stack to the env cfg.

    Without ``--sensors`` the env keeps its registered stack and the
    writer mirrors it, so the rendered cameras and the recorded columns
    cannot drift. An explicit ``--sensors`` recomposes the env around
    the requested stack (same mechanism as the teleop driver).
    """
    if getattr(env_cfg, "sensors", None) is None:
        raise SystemExit(
            f"--mode harness requires a composed capture task exposing a "
            f"sensor stack; {args.task} has none.",
        )
    if args.sensors:
        from strafer_lab.tasks.navigation.composed_env_cfg import SensorStackCfg

        tokens = tuple(t.strip() for t in args.sensors.split(",") if t.strip())
        env_cfg.sensors = SensorStackCfg(cameras_required=tokens)
        env_cfg.__post_init__()
        # Re-pin what parse_env_cfg already applied, in case recomposition
        # rebuilt the scene cfg.
        env_cfg.scene.num_envs = 1
        print(f"[sim_in_the_loop] sensor stack override → {tokens}")
    stack = tuple(env_cfg.sensors.cameras_required)
    if not args.no_camera_bridge and not {"rgb_full", "depth_full"}.issubset(stack):
        raise SystemExit(
            f"--mode harness needs rgb_full + depth_full in the sensor stack "
            f"(got {stack}): the Jetson stack navigates on the bridged "
            f"perception camera streams. Pass --no-camera-bridge only for "
            f"publisher-less debugging.",
        )
    return stack


# ---------------------------------------------------------------------------
# Mode: bridge (drive /cmd_vel into the env)
# ---------------------------------------------------------------------------


def _run_bridge_mode(simulation_app, env, args, config) -> None:
    import torch

    from strafer_lab.bridge.async_camera_publisher import StraferCameraAsyncPublisher
    from strafer_lab.bridge.async_publisher import StraferAsyncPublisher
    from strafer_shared.constants import D555_FOCAL_LENGTH_MM, D555_HORIZONTAL_APERTURE_MM

    unwrapped = env.unwrapped
    env.reset()

    # Eliminate the redundant Kit pump under --viz kit. Without this, env.step
    # calls sim.render() → KitVisualizer.step → app.update(playSimulations=False)
    # to refresh the editor viewport, then this loop calls simulation_app.update()
    # below with playSimulations=True to fire OnPlaybackTick. Both pumps RTX-
    # render the editor viewport, paying ~80 ms of duplicated work per loop.
    #
    # Setting render_enabled=False makes env.step() invoke sim.render() with
    # skip_app_pumping=True, which short-circuits KitVisualizer (no app.update).
    # The simulation_app.update() below then becomes the SOLE Kit pump per loop
    # and refreshes the viewport / fires OnPlaybackTick in one pass.
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

    camera_publisher: StraferCameraAsyncPublisher | None = None
    if not args.no_camera_bridge:
        # Camera-thread phase metrics are recorded by the worker thread
        # itself (post-step); the profiler reads them on the bridge thread
        # when reporting. Threading-safe because collections.deque.append
        # is atomic in CPython.
        if profiler is not None:
            readback_cb = lambda ms, _p=profiler: _p.record(  # noqa: E731
                "camera :: GPU→CPU readback", ms / 1000.0,
            )
            publish_cb = lambda ms, _p=profiler: _p.record(  # noqa: E731
                "camera :: rclpy publish", ms / 1000.0,
            )
        else:
            readback_cb = None
            publish_cb = None
        camera_publisher = StraferCameraAsyncPublisher(
            camera_sensor=unwrapped.scene["d555_camera_perception"],
            color_stream=config.color_camera,
            depth_stream=config.depth_camera,
            focal_length_mm=D555_FOCAL_LENGTH_MM,
            horizontal_aperture_mm=D555_HORIZONTAL_APERTURE_MM,
            frame_skip=config.camera_frame_skip,
            on_readback_ms=readback_cb,
            on_publish_ms=publish_cb,
        )
        print(
            "[sim_in_the_loop] async camera publisher up: "
            f"{config.color_camera.image_topic}, "
            f"{config.color_camera.camera_info_topic}, "
            f"{config.depth_camera.image_topic}, "
            f"{config.depth_camera.camera_info_topic}"
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
                if camera_publisher is not None:
                    with profiler.time("loop :: camera notify_frame"):
                        camera_publisher.notify_frame(sim_time_s)
                with profiler.time("loop :: simulation_app.update"):
                    simulation_app.update()
                profiler.tick()
            else:
                env.step(action)
                sim_time_s += step_dt
                publisher.publish_state(sim_time_s)
                if camera_publisher is not None:
                    camera_publisher.notify_frame(sim_time_s)
                # Sole Kit pump per loop iteration (render_enabled=False above
                # disables KitVisualizer's pump inside env.step). One app.update
                # per env.step refreshes the editor viewport and fires
                # OnPlaybackTick for whatever residual nodes the bridge graph
                # still hosts.
                simulation_app.update()
    finally:
        if camera_publisher is not None:
            camera_publisher.shutdown()
        publisher.shutdown()


# ---------------------------------------------------------------------------
# Mode: harness (walk a mission stream, record a LeRobot v3 dataset)
# ---------------------------------------------------------------------------


def _git_rev_parse_head(repo_root: Path) -> str:
    """Return ``git rev-parse HEAD`` for the repo, or an empty string."""
    import subprocess

    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode("utf-8").strip()
    except Exception:
        return ""


def _build_harness_missions(args, scene_name: str, metadata: dict) -> list:
    """Build the (spec, meta) stream: queue rows or scene-metadata targets,
    each run through the hard-negative injection planner."""
    import random

    from strafer_lab.sim_in_the_loop import EpisodeMeta, MissionGenerator
    from strafer_lab.tools.grounding_injection import (
        plan_injection,
        resolve_target_room_idx,
    )
    from strafer_lab.tools.mission_queue import (
        load_mission_queue,
        queue_row_to_mission_spec,
    )

    rng = random.Random(args.injection_seed)
    scene_objects = list(metadata.get("objects") or [])

    sourced: list[tuple] = []  # (spec, mission_text, room_idx, paraphrases, generator_metadata, source)
    if args.mission_queue is not None:
        rows = load_mission_queue(args.mission_queue)
        if args.max_missions is not None:
            rows = rows[: args.max_missions]
        for row in rows:
            spec = queue_row_to_mission_spec(row, scene_name=scene_name)
            room_idx = resolve_target_room_idx(
                target_label=row.target_label,
                target_position_3d=row.target_position_3d,
                objects=scene_objects,
            )
            sourced.append(
                (spec, row.mission_text, room_idx, row.paraphrases,
                 row.generator_metadata, "queue"),
            )
    else:
        generator = MissionGenerator.from_metadata(
            metadata,
            scene_name=scene_name,
            max_missions=args.max_missions,
            allowed_labels=args.allowed_labels,
            blocked_labels=tuple(args.blocked_labels or ()),
        )
        for spec in generator:
            sourced.append(
                (spec, spec.raw_command, spec.target_room_idx, (), {},
                 "scene-metadata"),
            )

    missions: list = []
    for spec, mission_text, room_idx, paraphrases, generator_metadata, source in sourced:
        plan = plan_injection(
            mode=args.inject_bad_grounding,
            probability=args.inject_bad_grounding_prob,
            rng=rng,
            target_label=spec.target_label,
            target_instance_id=spec.target_instance_id,
            target_position_3d=spec.target_position_3d,
            target_room_idx=room_idx,
            objects=scene_objects,
        )
        # Four distinct text axes, easy to conflate:
        #   - mission_text: the RECORDED language. Always names the ORIGINAL
        #     target, even under injection — that mismatch against where the
        #     robot actually drove is the whole point of a grounding negative.
        #   - dispatch_command: what the EXECUTOR is told to drive to. When
        #     injected we hand it a simplified "go to the {label}" imperative
        #     so it reliably grounds the *wrong* goal; honest missions pass
        #     spec.raw_command through unchanged.
        #   - spec.raw_command: the original rich natural-language mission.
        #   - paraphrases: augmentations of the recorded text — a separate
        #     axis, not a substitute for any of the above.
        # Mild confound to be aware of downstream: injected episodes dispatch
        # the simplified imperative while honest ones dispatch the richer
        # raw_command, so dispatch phrasing is not i.i.d. across the two
        # classes. The recorded mission_text is not affected.
        dispatch_command = (
            f"go to the {plan.target_label.strip().lower()}"
            if plan.injected else spec.raw_command
        )
        meta = EpisodeMeta(
            mission_text=mission_text,
            dispatch_command=dispatch_command,
            source_mission_source=source,
            target_label=plan.target_label,
            target_object_id=str(plan.target_instance_id),
            target_position_3d=plan.target_position_3d,
            injection_mode=plan.injection_mode,
            injection_mode_actual=plan.injection_mode_actual,
            original_target_position_3d=plan.original_target_position_3d,
            paraphrases=tuple(paraphrases),
            generator_metadata=dict(generator_metadata),
        )
        missions.append((spec, meta))
    return missions


def _run_harness_mode(simulation_app, env, args, config, cameras_required) -> None:
    from strafer_lab.bridge.async_camera_publisher import StraferCameraAsyncPublisher
    from strafer_lab.bridge.async_publisher import StraferAsyncPublisher
    from strafer_lab.sim_in_the_loop import (
        BridgeLeRobotRecorder,
        CmdVelGraceWatch,
        HarnessConfig,
        SimInTheLoopHarness,
    )
    from strafer_lab.sim_in_the_loop.runtime_env import IsaacLabEnvAdapter
    from strafer_lab.sim_in_the_loop.runtime_mission import Ros2MissionApi
    from strafer_lab.tools.bbox_extractor import (
        ReplicatorBboxExtractor,
        resolve_render_product_path,
    )
    from strafer_lab.tools import scene_metadata_reader
    from strafer_lab.tools.lerobot_detections import DETECTIONS_MAX_DEFAULT
    from strafer_lab.tools.lerobot_writer import StraferLeRobotWriter, hash_scene_metadata
    from strafer_lab.tools.scene_paths import resolve_scene_usd_path
    from strafer_shared.constants import D555_FOCAL_LENGTH_MM, D555_HORIZONTAL_APERTURE_MM

    # --- mission stream -------------------------------------------------
    scene_usd_path = resolve_scene_usd_path(
        scene=args.scene_name,
        usd_override=args.scene_usd,
    )
    scene_metadata = scene_metadata_reader.load(scene_usd_path)
    scene_name = args.scene_name or scene_usd_path.stem

    missions = _build_harness_missions(args, scene_name, scene_metadata)
    if not missions:
        print(f"[sim_in_the_loop] no missions to run for {scene_name}; exiting")
        return
    injected_count = sum(1 for _, meta in missions if meta.injection_mode_actual)
    print(
        f"[sim_in_the_loop] {len(missions)} missions queued for {scene_name} "
        f"({injected_count} with injected bad grounding)"
    )

    # --- env + publishers (same per-tick protocol as bridge mode) --------
    unwrapped = env.unwrapped
    env.reset()
    # Same Kit-pump dedup as bridge mode: env.step skips KitVisualizer and
    # the adapter's per-step simulation_app.update() is the sole pump.
    unwrapped.render_enabled = False

    publisher = StraferAsyncPublisher(
        robot=unwrapped.scene["robot"],
        clock_topic=config.clock_topic,
        odom_topic=config.odom_topic,
        cmd_vel_topic=config.cmd_vel_topic,
        odom_frame_id=config.odom_frame_id,
        base_frame_id=config.base_frame_id,
    )
    print("[sim_in_the_loop] async publisher up: /clock, /odom, TF, /cmd_vel")

    camera_publisher: StraferCameraAsyncPublisher | None = None
    if not args.no_camera_bridge:
        camera_publisher = StraferCameraAsyncPublisher(
            camera_sensor=unwrapped.scene["d555_camera_perception"],
            color_stream=config.color_camera,
            depth_stream=config.depth_camera,
            focal_length_mm=D555_FOCAL_LENGTH_MM,
            horizontal_aperture_mm=D555_HORIZONTAL_APERTURE_MM,
            frame_skip=config.camera_frame_skip,
        )
        print(
            "[sim_in_the_loop] async camera publisher up: "
            f"{config.color_camera.image_topic}, {config.depth_camera.image_topic}"
        )

    def _on_stepped(sim_time_s: float) -> None:
        publisher.publish_state(sim_time_s)
        if camera_publisher is not None:
            camera_publisher.notify_frame(sim_time_s)
        simulation_app.update()

    detections_source = None
    if args.detections:
        extractor = ReplicatorBboxExtractor(
            camera_render_product_path=resolve_render_product_path(
                unwrapped.scene["d555_camera_perception"],
            ),
        )
        detections_source = extractor.extract
        print("[sim_in_the_loop] detections annotator attached (perception camera)")

    env_adapter = IsaacLabEnvAdapter(
        env=env,
        scene_name=scene_name,
        cmd_vel_source=publisher.get_cmd_vel,
        cameras_required=cameras_required,
        detections_source=detections_source,
        on_stepped=_on_stepped,
    )

    # --- writer -----------------------------------------------------------
    output_root = Path(args.output).resolve()
    if output_root.exists():
        suffix = time.strftime("_%Y%m%dT%H%M%S")
        output_root = output_root.with_name(output_root.name + suffix)
        print(
            f"[sim_in_the_loop] requested --output already exists; using "
            f"auto-suffixed path → {output_root}"
        )

    step_dt = float(unwrapped.sim.get_physics_dt()) * int(unwrapped.cfg.decimation)
    writer_fps = max(1, round(1.0 / (step_dt * max(1, args.capture_every_n_steps))))
    repo_root = Path(__file__).resolve().parents[3]
    writer = StraferLeRobotWriter(
        root=output_root,
        repo_id=args.repo_id or f"strafer/{scene_name}",
        fps=writer_fps,
        capture_git_sha=_git_rev_parse_head(repo_root),
        scene_metadata_hash=hash_scene_metadata(scene_metadata),
        cameras_required=cameras_required,
        detections_max=(
            (args.detections_max or DETECTIONS_MAX_DEFAULT) if args.detections else None
        ),
        vcodec=args.vcodec,
        session_id=args.session_id or time.strftime("%Y%m%dT%H%M%S"),
    )
    print(
        f"[sim_in_the_loop] writer ready → {output_root} "
        f"(fps={writer_fps}, cameras={cameras_required}, "
        f"detections={'on' if args.detections else 'off'})"
    )

    grace_watch = CmdVelGraceWatch(
        last_cmd_time=publisher.last_cmd_monotonic,
        grace_s=args.cmd_vel_grace,
        now=time.monotonic,
    )
    cfg = HarnessConfig(
        mission_timeout_s=args.mission_timeout_s,
        capture_every_n_steps=args.capture_every_n_steps,
    )

    recorder = BridgeLeRobotRecorder(writer=writer, scene_id=scene_name)
    consecutive_failures = 0
    try:
        with writer, Ros2MissionApi() as mission_api:
            harness = SimInTheLoopHarness(
                env_adapter=env_adapter,
                mission_api=mission_api,
                recorder=recorder,
                config=cfg,
                abort_signal=grace_watch,
            )

            for spec, meta in missions:
                if not simulation_app.is_running():
                    print("[sim_in_the_loop] simulation app shut down; aborting run")
                    break
                try:
                    outcome = harness.run_one_mission(spec, meta=meta)
                except Exception as exc:
                    consecutive_failures += 1
                    print(
                        f"[sim_in_the_loop] {spec.mission_id} crashed "
                        f"({type(exc).__name__}: {exc}); episode discarded "
                        f"({consecutive_failures} consecutive failure(s))"
                    )
                    if consecutive_failures >= _MAX_CONSECUTIVE_MISSION_FAILURES:
                        print(
                            f"[sim_in_the_loop] {_MAX_CONSECUTIVE_MISSION_FAILURES} "
                            "consecutive mission failures — executor looks down; "
                            "aborting run"
                        )
                        break
                    continue
                consecutive_failures = 0
                tag = "DISCARDED" if outcome.discarded else (
                    "ok" if outcome.reachability else "failed"
                )
                print(
                    f"[sim_in_the_loop] {spec.mission_id} [{tag}] "
                    f"state={outcome.final_status.state} "
                    f"frames={outcome.frames_written} "
                    f"elapsed={outcome.elapsed_s:.1f}s"
                )
    finally:
        if camera_publisher is not None:
            camera_publisher.shutdown()
        publisher.shutdown()

    print(
        f"[sim_in_the_loop] capture done: {recorder.episodes_kept} episode(s) "
        f"kept, {recorder.episodes_discarded} discarded → {output_root}"
    )


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _validate_harness_args(args: argparse.Namespace) -> None:
    if args.output is None:
        raise SystemExit("--mode harness requires --output to be set")
    if args.scene_name is None and args.scene_usd is None:
        raise SystemExit(
            "--mode harness requires --scene-name or --scene-usd to locate "
            "the scene USD whose customData carries the mission targets"
        )
    if args.mission_queue is not None and not args.mission_queue.is_file():
        raise SystemExit(f"mission queue not found: {args.mission_queue}")
    # Resolve the scene USD before the multi-minute Kit boot so a typo'd
    # scene name fails in milliseconds. strafer_lab.tools imports without Kit.
    from strafer_lab.tools.scene_paths import resolve_scene_usd_path

    try:
        resolve_scene_usd_path(scene=args.scene_name, usd_override=args.scene_usd)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
