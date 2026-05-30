"""Strafer harness data-capture entry point.

Unified entry point for the harness data-capture pipeline per
``docs/tasks/active/harness/harness-architecture.md``. Selects one
``(driver, mission-source)`` cell of the cross-product matrix and runs
a capture session that writes a LeRobot v3 dataset.

::

    Scripts/capture.py
      --driver        {bridge, teleop, scripted}      ← who provides actions
      --mission-source {queue, captioner, coverage, scene-metadata}
      --scene         <scene_name>
      --output        <dataset_root>
      [--num-envs N]                                  ← scripted only
      [--mission-queue <yaml>]                        ← queue mission-source
      [--n-trajectories N]                            ← captioner / coverage
      [--inject-bad-grounding {off, wrong_room, wrong_instance}]
      [--paraphrase-missions N]                       ← post-capture pass
      [--capture-policy-cam / --no-capture-policy-cam]

Implementation status (this is Tier 1 / PR B scaffolding):

- ``--driver teleop --mission-source scene-metadata`` → the only
  combination Tier 1 wires; in-process Isaac Lab + gamepad. The
  driver itself lands as the next commit on this branch.
- All other ``(driver, mission-source)`` cells raise
  ``NotImplementedError`` with a pointer to the tier that ships them.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence


# Valid (driver, mission_source) combinations from
# harness-architecture.md. Mapping value = (tier_label, status).
VALID_COMBINATIONS: dict[tuple[str, str], tuple[str, str]] = {
    ("bridge", "scene-metadata"): ("Tier 2", "pending"),
    ("bridge", "queue"): ("Tier 2", "pending"),
    ("teleop", "queue"): ("Tier 1 follow-up", "pending"),
    ("teleop", "scene-metadata"): ("Tier 1", "wired"),
    ("scripted", "queue"): ("Tier 3", "pending"),
    ("scripted", "captioner"): ("Tier 3", "pending"),
    ("scripted", "coverage"): ("Tier 3", "pending"),
}


# Driver script lookup. The teleop driver lives alongside collect_demos.py
# in source/strafer_lab/scripts/ — it boots its own AppLauncher so capture.py
# stays Isaac-Sim-free and importable from .venv_harness for unit tests.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_TELEOP_DRIVER_SCRIPT = (
    _REPO_ROOT / "source" / "strafer_lab" / "scripts" / "teleop_capture.py"
)


VALID_DRIVERS = sorted({d for d, _ in VALID_COMBINATIONS})
VALID_MISSION_SOURCES = sorted({m for _, m in VALID_COMBINATIONS})


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
        choices=("off", "wrong_room", "wrong_instance"),
        default="off",
        help="Hard-negative injection mode (bridge / scripted only).",
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
        "--capture-policy-cam",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Capture the 80×60 policy camera alongside the 640×360 "
             "perception camera. Default on for v1 training corpora.",
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
        help="Capture rate in Hz. Default 8 matches the bridge mainloop "
             "post-decimation rate.",
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
            "Teleop hard negatives come from the X / SELECT buttons. See "
            "harness-architecture.md → Driver: teleop.",
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
    argv: list[str] = [
        sys.executable, str(_TELEOP_DRIVER_SCRIPT),
        "--scene", args.scene,
        "--output", args.output,
        "--fps", str(args.fps),
        "--vcodec", args.vcodec,
    ]
    if args.capture_policy_cam:
        argv.append("--capture-policy-cam")
    else:
        argv.append("--no-capture-policy-cam")
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
    tier_label, status = VALID_COMBINATIONS[cell]

    if cell == ("teleop", "scene-metadata"):
        print(
            f"[capture] selected: driver={args.driver}, "
            f"mission_source={args.mission_source} ({tier_label}, {status})",
            file=sys.stderr,
        )
        if not _TELEOP_DRIVER_SCRIPT.is_file():
            raise SystemExit(
                f"teleop driver script missing at {_TELEOP_DRIVER_SCRIPT}. "
                "This is a packaging bug.",
            )
        argv = _build_teleop_subprocess_argv(args, extra_args)
        print(f"[capture] dispatching → {' '.join(argv)}", file=sys.stderr)
        result = runner(argv, check=False)
        return int(getattr(result, "returncode", 0) or 0)

    raise NotImplementedError(
        f"--driver {args.driver} --mission-source {args.mission_source} "
        f"is gated on {tier_label}. See docs/tasks/active/harness/"
        f"harness-architecture.md#implementation-tiers for the schedule.",
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = _build_parser()
    args, extra = parser.parse_known_args(argv)
    _validate(args)
    return _dispatch(args, extra_args=tuple(extra))


if __name__ == "__main__":
    sys.exit(main())
