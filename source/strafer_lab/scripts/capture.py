"""Strafer harness data-capture entry point.

Unified entry point for the harness data-capture pipeline per
``docs/tasks/active/harness/harness-architecture.md``. Selects one
``(driver, mission-source)`` cell of the cross-product matrix and runs
a capture session that writes a LeRobot v3 dataset.

::

    source/strafer_lab/scripts/capture.py
      --driver        {bridge, teleop, scripted}      ← who provides actions
      --mission-source {queue, captioner, coverage, scene-metadata}
      --scene         <scene_name>
      --output        <dataset_root>
      [--num-envs N]                                  ← scripted only
      [--mission-queue <yaml>]                        ← queue mission-source
      [--n-trajectories N]                            ← captioner / coverage
      [--inject-bad-grounding {off, wrong_room, wrong_instance}]
      [--paraphrase-missions N]                       ← post-capture pass
      [--sensors rgb_full[,depth_full,...] | <preset>] ← per-session stack
      [--capture-policy-cam / --no-capture-policy-cam]  (deprecated)

Implementation status:

- ``--driver teleop --mission-source scene-metadata`` → in-process
  Isaac Lab + gamepad (``teleop_capture.py``).
- ``--driver bridge`` (``scene-metadata`` or ``queue``) → the Jetson
  autonomy stack drives via /cmd_vel
  (``run_sim_in_the_loop.py --mode harness``).
- The ``scripted`` driver and the ``captioner`` / ``coverage`` mission
  sources are not implemented yet and raise ``NotImplementedError``.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


# Valid (driver, mission_source) combinations. Mapping value = status:
# "wired" cells dispatch to a driver script; "pending" cells raise
# NotImplementedError until their driver ships.
VALID_COMBINATIONS: dict[tuple[str, str], str] = {
    ("bridge", "scene-metadata"): "wired",
    ("bridge", "queue"): "wired",
    ("teleop", "queue"): "pending",
    ("teleop", "scene-metadata"): "wired",
    ("scripted", "queue"): "pending",
    ("scripted", "captioner"): "pending",
    ("scripted", "coverage"): "pending",
}


# Driver script lookup. Both drivers are siblings in this scripts/ dir —
# each boots its own AppLauncher so capture.py stays Isaac-Sim-free and
# importable without Isaac Sim for unit tests.
_TELEOP_DRIVER_SCRIPT = Path(__file__).resolve().parent / "teleop_capture.py"
_BRIDGE_DRIVER_SCRIPT = Path(__file__).resolve().parent / "run_sim_in_the_loop.py"


VALID_DRIVERS = sorted({d for d, _ in VALID_COMBINATIONS})
VALID_MISSION_SOURCES = sorted({m for _, m in VALID_COMBINATIONS})


# Sensor-stack tokens shared with the env composition's SensorStackCfg and the
# LeRobot writer's build_features — one ``cameras_required`` vocabulary so the
# rendered cameras and the recorded columns cannot drift. ``*_full`` ride the
# 640x360 perception camera, ``*_policy`` the 80x60 policy camera.
SENSOR_TOKENS: tuple[str, ...] = ("rgb_full", "depth_full", "rgb_policy", "depth_policy")

# Named presets that resolve to a cameras_required tuple. RGB-only is the
# common teleop case (CLIP mid-mission-validation + grounding-VLM finetune all
# need just one full-res egocentric RGB); the wider presets are for VLA capture
# and the bridge's full data product.
SENSOR_PRESETS: dict[str, tuple[str, ...]] = {
    "teleop": ("rgb_full",),
    "vla": ("rgb_full", "depth_full"),
    "coverage": ("rgb_full", "depth_full"),
    "bridge": ("rgb_full", "depth_full", "depth_policy"),
}


def resolve_sensor_stack(
    spec: str | None,
    *,
    capture_policy_cam: bool,
) -> tuple[str, ...]:
    """Resolve a ``--sensors`` spec to a canonical cameras_required tuple.

    ``spec`` is either a named preset (``teleop`` / ``vla`` / ``coverage`` /
    ``bridge``) or a comma-separated token list (``rgb_full,depth_full``). When
    ``spec`` is ``None`` the stack falls back to the deprecated
    ``--capture-policy-cam`` bool so existing invocations keep working.
    """
    if spec is None:
        return ("rgb_full", "rgb_policy") if capture_policy_cam else ("rgb_full",)
    spec = spec.strip()
    if spec in SENSOR_PRESETS:
        tokens: tuple[str, ...] = SENSOR_PRESETS[spec]
    else:
        tokens = tuple(t.strip() for t in spec.split(",") if t.strip())
    if not tokens:
        raise SystemExit("--sensors: empty sensor stack")
    unknown = set(tokens) - set(SENSOR_TOKENS)
    if unknown:
        raise SystemExit(
            f"--sensors: unknown token(s) {sorted(unknown)}. "
            f"Valid tokens: {SENSOR_TOKENS}; presets: {sorted(SENSOR_PRESETS)}",
        )
    return tuple(t for t in SENSOR_TOKENS if t in set(tokens))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--driver",
        required=True,
        choices=VALID_DRIVERS,
        help="Action source: bridge (Jetson ROS), teleop (gamepad in-process), "
             "scripted (RL/proportional in-process).",
    )
    parser.add_argument(
        "--mission-source",
        required=True,
        choices=VALID_MISSION_SOURCES,
        help="Where missions come from: queue (forward-generated YAML), "
             "captioner (random A→B + post-hoc), coverage (visit-every-room), "
             "scene-metadata (walk scene_metadata.json targets).",
    )
    parser.add_argument(
        "--scene",
        required=True,
        help="Scene name (used as the LeRobot dataset's repo_id suffix and "
             "as scene_id in per-episode metadata).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Dataset root directory. Must not exist (LeRobotDataset.create "
             "refuses to overwrite). One scene = one dataset.",
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Parallel env count (scripted driver only). Bounded by "
             "harness-throughput-measurement; default 1.",
    )
    parser.add_argument(
        "--mission-queue",
        default=None,
        help="Path to mission_queue.yaml (queue mission-source only).",
    )
    parser.add_argument(
        "--n-trajectories",
        type=int,
        default=None,
        help="Number of trajectories to capture (captioner / coverage modes).",
    )
    parser.add_argument(
        "--inject-bad-grounding",
        choices=("off", "wrong_room", "wrong_instance", "wrong_object"),
        default="off",
        help="Hard-negative injection mode (bridge / scripted only): "
             "wrong_room (different room), wrong_instance (same label, same "
             "room), wrong_object (different label, same room).",
    )
    parser.add_argument(
        "--inject-bad-grounding-prob",
        type=float,
        default=0.3,
        help="Per-mission probability of perturbation when "
             "--inject-bad-grounding is enabled.",
    )
    parser.add_argument(
        "--paraphrase-missions",
        type=int,
        default=0,
        help="Run a post-capture Qwen2.5-VL paraphrase pass with N "
             "paraphrases per episode. Default 0 (off).",
    )
    parser.add_argument(
        "--detections",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Record per-frame 2D detections (Replicator ground truth) as "
             "first-class dataset columns. Default: on for the bridge "
             "driver, unavailable for teleop (the gamepad driver has no "
             "annotator and prioritizes operator frame rate).",
    )
    parser.add_argument(
        "--sensors",
        default=None,
        help="Per-session sensor stack: a preset (teleop / vla / coverage / "
             "bridge) or a comma-separated token list over "
             f"{','.join(SENSOR_TOKENS)} (e.g. 'rgb_full,depth_full'). The env "
             "renders and the writer records exactly this stack. Defaults to "
             "the deprecated --capture-policy-cam behavior when omitted.",
    )
    parser.add_argument(
        "--capture-policy-cam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Deprecated — prefer --sensors. Capture the 80×60 policy camera "
             "alongside the 640×360 perception camera. Used only when "
             "--sensors is omitted.",
    )
    parser.add_argument(
        "--vcodec",
        default="h264",
        help="LeRobot v3 video codec. h264 is the broadest-availability "
             "choice on ARM64; libsvtav1 is the lerobot default. See "
             "lerobot.datasets.video_utils.VALID_VIDEO_CODECS for the full set.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=8,
        help="Capture rate in Hz (teleop driver). The bridge driver "
             "derives its rate from the env step dt × its "
             "--capture-every-n-steps instead.",
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    """Validate the (driver, mission-source) combination + flag dependencies."""
    cell = (args.driver, args.mission_source)
    if cell not in VALID_COMBINATIONS:
        raise SystemExit(
            f"Invalid combination: --driver {args.driver} --mission-source "
            f"{args.mission_source}.\n"
            f"Valid combinations: {sorted(VALID_COMBINATIONS)}",
        )
    if args.mission_source == "queue" and args.mission_queue is None:
        raise SystemExit(
            "--mission-source queue requires --mission-queue <yaml>",
        )
    if args.driver != "scripted" and args.num_envs > 1:
        raise SystemExit(
            f"--num-envs {args.num_envs} is only valid with --driver scripted "
            "(teleop and bridge run single-env).",
        )
    if args.inject_bad_grounding != "off" and args.driver == "teleop":
        raise SystemExit(
            "--inject-bad-grounding is not valid for --driver teleop. "
            "Teleop hard negatives come from the X / SELECT buttons.",
        )
    if args.detections and args.driver == "teleop":
        raise SystemExit(
            "--detections is not available for --driver teleop; the teleop "
            "driver captures without the detections annotator.",
        )


def _build_teleop_subprocess_argv(
    args: argparse.Namespace,
    extra_args: Sequence[str],
) -> list[str]:
    """Translate ``capture.py`` args into the teleop driver's argv.

    The teleop driver script (``source/strafer_lab/scripts/teleop_capture.py``)
    boots its own AppLauncher and parses its own CLI; we just forward
    the cross-product-validated args plus any unknown extras (e.g.
    ``--headless`` / AppLauncher flags) the operator passed through.
    """
    stack = resolve_sensor_stack(
        args.sensors, capture_policy_cam=args.capture_policy_cam,
    )
    argv: list[str] = [
        sys.executable, str(_TELEOP_DRIVER_SCRIPT),
        "--scene", args.scene,
        "--output", args.output,
        "--fps", str(args.fps),
        "--vcodec", args.vcodec,
        "--sensors", ",".join(stack),
    ]
    if extra_args:
        argv.extend(extra_args)
    return argv


def _build_bridge_subprocess_argv(
    args: argparse.Namespace,
    extra_args: Sequence[str],
) -> list[str]:
    """Translate ``capture.py`` args into the bridge driver's argv.

    The bridge driver is ``run_sim_in_the_loop.py --mode harness`` — it
    boots its own AppLauncher, brings up the ROS 2 bridge publishers,
    and walks the mission stream while the Jetson autonomy stack drives.
    Unknown extras (``--headless``, ``--max-missions``, AppLauncher
    flags, ...) pass through to the child.
    """
    stack = resolve_sensor_stack(
        args.sensors if args.sensors is not None else "bridge",
        capture_policy_cam=args.capture_policy_cam,
    )
    argv: list[str] = [
        sys.executable, str(_BRIDGE_DRIVER_SCRIPT),
        "--mode", "harness",
        "--scene-name", args.scene,
        "--output", args.output,
        "--vcodec", args.vcodec,
        "--sensors", ",".join(stack),
    ]
    if args.mission_source == "queue":
        argv.extend(["--mission-queue", args.mission_queue])
    if args.detections is False:
        argv.append("--no-detections")
    if args.inject_bad_grounding != "off":
        argv.extend([
            "--inject-bad-grounding", args.inject_bad_grounding,
            "--inject-bad-grounding-prob", str(args.inject_bad_grounding_prob),
        ])
    if extra_args:
        argv.extend(extra_args)
    return argv


def _dispatch(
    args: argparse.Namespace,
    *,
    extra_args: Sequence[str] = (),
    runner=subprocess.run,
) -> int:
    """Route to the driver implementation for this (driver, mission-source).

    Returns process exit code. ``runner`` is injected so unit tests can
    swap in a mock that captures argv without spawning Isaac Sim.
    """
    cell = (args.driver, args.mission_source)
    status = VALID_COMBINATIONS[cell]

    if status != "wired":
        if cell == ("teleop", "queue"):
            raise NotImplementedError(
                "--driver teleop --mission-source queue is not wired yet: the "
                "teleop driver reads targets from scene metadata, not a mission "
                "queue, and the curated-queue producer it would consume has not "
                "shipped. Use --mission-source scene-metadata for teleop today.",
            )
        raise NotImplementedError(
            f"--driver {args.driver} --mission-source {args.mission_source} "
            "is not implemented yet; this cell ships with the scripted driver.",
        )

    if args.driver == "teleop":
        driver_script = _TELEOP_DRIVER_SCRIPT
        argv = _build_teleop_subprocess_argv(args, extra_args)
    else:  # bridge
        driver_script = _BRIDGE_DRIVER_SCRIPT
        argv = _build_bridge_subprocess_argv(args, extra_args)

    print(
        f"[capture] selected: driver={args.driver}, "
        f"mission_source={args.mission_source} ({status})",
        file=sys.stderr,
    )
    if not driver_script.is_file():
        raise SystemExit(
            f"driver script missing at {driver_script}. "
            "This is a packaging bug.",
        )
    print(f"[capture] dispatching → {' '.join(argv)}", file=sys.stderr)
    result = runner(argv, check=False)
    return int(getattr(result, "returncode", 0) or 0)


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)
    _validate(args)
    return _dispatch(args, extra_args=tuple(extra))


if __name__ == "__main__":
    sys.exit(main())
