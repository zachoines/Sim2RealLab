"""LeRobot v3 dataset writer for strafer harness captures.

Wraps :class:`lerobot.datasets.lerobot_dataset.LeRobotDataset` with a
per-episode API that matches the harness driver lifecycle (begin →
add_frame → end). Native LeRobot v3 columns (state, action, video) ride
the upstream writer; strafer extensions (per-episode metadata + 16UC1
PNG depth) ship as sidecar files.

Sidecars produced alongside the LeRobot v3 dataset root:

- ``<root>/videos/observation.depth.perception/episode-NNNNNN/NNNNNN.png``
  — 16UC1 depth, per-frame, see :mod:`lerobot_depth`.
- ``<root>/meta/strafer_episodes.parquet`` — per-episode extension
  columns (outcome, scene_id, capture_git_sha, scene_metadata_hash,
  injection_mode_actual, episode_split, ...). Joined to LeRobot's own
  ``meta/episodes/`` chunked Parquet via ``episode_index``.
- ``<root>/meta/detection_labels.json`` — id↔string vocab for the
  first-class ``observation.detections.*`` padded columns (present only
  when the writer is constructed with ``detections_max``); see
  :mod:`lerobot_detections`.

Lifecycle::

    writer = StraferLeRobotWriter(
        root="data/sim_in_the_loop/<scene>",
        repo_id="strafer/<scene>",
        fps=8,
        cameras_required=("rgb_full", "rgb_policy", "depth_full"),
        capture_git_sha="abc123",
        scene_metadata_hash="sha256...",
    )
    writer.begin_episode(
        mission_text="go to the chair",
        scene_id="scene_001",
        target_label="chair",
        target_position_3d=[4.2, 1.8, 0.0],
        start_pose=[0.5, 0.5, 0.0],
        source_driver="teleop",
        source_mission_source="scene-metadata",
    )
    while collecting:
        writer.add_frame(
            sim_time=t,
            pose=[x, y, z, qx, qy, qz, qw],
            achieved_vel=[vx, vy, omega_z],
            action=[vx_cmd, vy_cmd, omega_z_cmd],
            rgb_perception=rgb_640x360,
            rgb_policy=rgb_80x60,     # optional
            depth_m=depth_640x360,    # optional float32 meters
        )
    writer.end_episode(
        outcome="succeeded",
        outcome_category="on_course",
        hard_negative_category=None,
    )
    writer.finalize()  # call on exit; or use as context manager

The class is pure-Python aside from the lerobot import — it does not
touch Isaac Sim, ROS, or any sim-only state. Unit-testable without Isaac Sim.
"""

from __future__ import annotations

import hashlib
import json
import queue
import threading
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .bbox_extractor import DetectedBbox
from .lerobot_depth import (
    PERCEPTION_DEPTH,
    POLICY_DEPTH,
    frame_path as depth_frame_path,
    write_depth_png,
)
from .lerobot_detections import (
    DetectionLabelVocab,
    detections_features,
    pack_detections,
)


# Resolution constants for the InfinigenPerception scene. Kept here so
# the writer's features dict is deterministic without importing the
# Isaac Lab scene config.
_PERCEPTION_RES = (360, 640)  # (H, W)
_POLICY_RES = (60, 80)

# Sensor-stack tokens shared with the env composition's SensorStackCfg. The
# env render set and this writer's schema are both driven from one
# ``cameras_required`` tuple so the rendered cameras and the recorded columns
# cannot drift. ``*_full`` ride the 640x360 perception camera, ``*_policy``
# the 80x60 policy camera; RGB tokens are LeRobot video columns, depth tokens
# ride as 16UC1 PNG sidecars.
CAMERA_TOKENS: tuple[str, ...] = ("rgb_full", "depth_full", "rgb_policy", "depth_policy")


def _normalize_cameras_required(
    cameras_required: Sequence[str] | None,
    capture_policy_cam: bool | None,
) -> tuple[str, ...]:
    """Resolve + validate the requested sensor stack.

    ``capture_policy_cam`` is the deprecated bool that only ever toggled the
    policy RGB video column; it maps to a ``cameras_required`` tuple when no
    explicit stack is given.
    """
    if capture_policy_cam is not None and cameras_required is None:
        cameras_required = ("rgb_full", "rgb_policy") if capture_policy_cam else ("rgb_full",)
    if cameras_required is None:
        cameras_required = ("rgb_full",)
    requested = set(cameras_required)
    unknown = requested - set(CAMERA_TOKENS)
    if unknown:
        raise ValueError(
            f"Unknown camera token(s) {sorted(unknown)}; valid: {CAMERA_TOKENS}",
        )
    return tuple(t for t in CAMERA_TOKENS if t in requested)


# ---------------------------------------------------------------------------
# Per-episode strafer extension columns
# ---------------------------------------------------------------------------


@dataclass
class StraferEpisodeExtensions:
    """Per-episode strafer metadata appended to a sidecar parquet.

    Columns mirror the harness-architecture brief's per-episode metadata
    table. None-valued fields land as null in the parquet.
    """

    episode_index: int
    scene_id: str
    target_label: str | None = None
    target_object_id: str | None = None
    target_position_3d: tuple[float, float, float] | None = None
    start_pose: tuple[float, float, float] | None = None  # (x, y, yaw)
    outcome: str = "succeeded"
    outcome_category: str = "on_course"
    paraphrases: tuple[str, ...] = ()
    source_driver: str = "teleop"
    source_mission_source: str = "scene-metadata"
    hard_negative_category: str | None = None
    injection_mode: str | None = None
    injection_mode_actual: str | None = None
    original_target_position_3d: tuple[float, float, float] | None = None
    operator_handle: str | None = None
    session_id: str | None = None
    generator_metadata: dict[str, Any] = field(default_factory=dict)
    leg_initial_distance_m: float | None = None
    episode_split: str = "train"
    capture_git_sha: str = ""
    scene_metadata_hash: str = ""
    # Realized D555 mount orientation (4 opaque floats, identity default); the
    # ±2deg mount jitter is logged so the realized camera stays derivable. A
    # realized_d555_mount_translation column is a future extension — not added.
    realized_d555_mount_quat: tuple[float, ...] | None = None

    def as_record(self) -> dict[str, Any]:
        """Return a dict suitable for pyarrow Table append."""
        return {
            "episode_index": int(self.episode_index),
            "scene_id": str(self.scene_id),
            "target_label": self.target_label,
            "target_object_id": self.target_object_id,
            "target_position_3d": (
                list(self.target_position_3d)
                if self.target_position_3d is not None else None
            ),
            "start_pose": (
                list(self.start_pose) if self.start_pose is not None else None
            ),
            "outcome": self.outcome,
            "outcome_category": self.outcome_category,
            "paraphrases": list(self.paraphrases),
            "source_driver": self.source_driver,
            "source_mission_source": self.source_mission_source,
            "hard_negative_category": self.hard_negative_category,
            "injection_mode": self.injection_mode,
            "injection_mode_actual": self.injection_mode_actual,
            "original_target_position_3d": (
                list(self.original_target_position_3d)
                if self.original_target_position_3d is not None else None
            ),
            "operator_handle": self.operator_handle,
            "session_id": self.session_id,
            "generator_metadata": json.dumps(self.generator_metadata),
            "leg_initial_distance_m": self.leg_initial_distance_m,
            "episode_split": self.episode_split,
            "capture_git_sha": self.capture_git_sha,
            "scene_metadata_hash": self.scene_metadata_hash,
            "realized_d555_mount_quat": (
                list(self.realized_d555_mount_quat)
                if self.realized_d555_mount_quat is not None else None
            ),
        }


# ---------------------------------------------------------------------------
# Depth sidecar worker pool
# ---------------------------------------------------------------------------


class _DepthWriterPool:
    """Background workers that PNG-encode depth frames off the hot path.

    Mirrors lerobot's :class:`AsyncImageWriter` semantics so the rest of
    the writer can stay synchronous-looking: unbounded queue (no
    backpressure on the env step — the operator-perceived smoothness of
    teleop matters more than tight queue bounds, and a few hundred MB of
    queued depth at <=2MB/frame is harmless on the DGX), drain-at-
    episode-boundary via :meth:`wait_until_done`.

    **Episode-boundary semantics**:
    :meth:`StraferLeRobotWriter.end_episode` calls :meth:`wait_until_done`
    *before* either ``save_episode`` (the kept path) or
    ``clear_episode_buffer`` (the discarded path). The next
    ``begin_episode`` therefore can never see queued frames from the
    previous episode, so an episode-index reuse on discard is safe.

    **Ctrl+C semantics**: workers are daemon threads. A normal Ctrl+C
    unwinds through the writer's context manager, which calls
    :meth:`StraferLeRobotWriter.finalize` and drains the queue cleanly.
    A *second* Ctrl+C during that drain kills the daemon workers
    mid-flush and a handful of in-flight frames are lost — which is
    the "abort everything now" intent. The in-flight episode is
    discarded in either case, so partial data never reaches disk.

    Worker exceptions are captured and re-raised on the next drain so a
    bad disk doesn't get silently swallowed.
    """

    def __init__(self, num_workers: int) -> None:
        self._write_depth_png = write_depth_png
        self._queue: queue.Queue[tuple[Path, np.ndarray] | None] = queue.Queue()
        self._workers: list[threading.Thread] = []
        self._exc_lock = threading.Lock()
        self._first_exc: BaseException | None = None
        for i in range(max(1, int(num_workers))):
            t = threading.Thread(
                target=self._worker_loop,
                name=f"strafer-depth-writer-{i}",
                daemon=True,
            )
            t.start()
            self._workers.append(t)

    def _worker_loop(self) -> None:
        while True:
            item = self._queue.get()
            try:
                if item is None:
                    return
                path, arr = item
                try:
                    self._write_depth_png(path, arr)
                except BaseException as exc:
                    with self._exc_lock:
                        if self._first_exc is None:
                            self._first_exc = exc
                    traceback.print_exc()
            finally:
                self._queue.task_done()

    def submit(self, path: Path, depth_m: np.ndarray) -> None:
        # Copy the array — the caller's buffer may be reused by Isaac Sim
        # on the next step. np.array(..., copy=True) is the cheap path.
        self._queue.put((path, np.array(depth_m, copy=True)))

    def wait_until_done(self) -> None:
        """Block until the queue is drained; re-raise the first worker error."""
        self._queue.join()
        with self._exc_lock:
            exc, self._first_exc = self._first_exc, None
        if exc is not None:
            raise exc

    def stop(self) -> None:
        """Signal all workers to exit and wait for them."""
        for _ in self._workers:
            self._queue.put(None)
        for t in self._workers:
            t.join(timeout=10.0)


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def build_features(
    cameras_required: Sequence[str] | None = None,
    state_dim: int = 10,  # pose (7) + achieved_vel (3)
    action_dim: int = 3,
    *,
    capture_policy_cam: bool | None = None,
    detections_max: int | None = None,
) -> dict[str, dict[str, Any]]:
    """Return the LeRobot v3 features dict for the selected sensor stack.

    Declares **exactly** the columns the requested ``cameras_required`` stack
    needs — an absent RGB modality produces no column at all (never zero-
    filled). Depth modalities ride as PNG sidecars outside the LeRobot schema,
    so they add no feature column here; they are still validated per frame
    against the same stack. Centralized so tests + writer + future consumers
    agree on the schema.

    ``detections_max`` declares the four ``observation.detections.*`` padded
    columns at that slot count; ``None`` (default) declares none — the
    schema shrinks like the camera stack.

    ``capture_policy_cam`` is the deprecated bool form (RGB video columns only);
    prefer ``cameras_required``.
    """
    stack = _normalize_cameras_required(cameras_required, capture_policy_cam)
    features: dict[str, dict[str, Any]] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": [
                "pose_x", "pose_y", "pose_z",
                "pose_qx", "pose_qy", "pose_qz", "pose_qw",
                "achieved_vx", "achieved_vy", "achieved_omega_z",
            ],
        },
        "action": {
            "dtype": "float32",
            "shape": (action_dim,),
            "names": ["vx_cmd", "vy_cmd", "omega_z_cmd"],
        },
    }
    if "rgb_full" in stack:
        features["observation.images.perception"] = {
            "dtype": "video",
            "shape": (*_PERCEPTION_RES, 3),
            "names": ["height", "width", "channels"],
        }
    if "rgb_policy" in stack:
        features["observation.images.policy"] = {
            "dtype": "video",
            "shape": (*_POLICY_RES, 3),
            "names": ["height", "width", "channels"],
        }
    if detections_max is not None:
        features.update(detections_features(detections_max))
    return features


class StraferLeRobotWriter:
    """Writer wrapping LeRobotDataset for the strafer harness.

    Construction creates the dataset on disk; the writer is invalid
    after :meth:`finalize`. Use as a context manager to guarantee
    ``finalize()`` runs on exceptional exit::

        with StraferLeRobotWriter(...) as writer:
            ...

    Required arguments:
      root:                    output dataset root (must not exist yet).
      repo_id:                  LeRobot repo_id (used in metadata).
      fps:                     capture rate in Hz.
      cameras_required:        the sensor stack the on-disk schema declares —
                               a tuple over ``rgb_full`` / ``depth_full`` /
                               ``rgb_policy`` / ``depth_policy``. The schema
                               grows or shrinks to exactly this stack; absent
                               modalities produce no columns. Defaults to
                               ``("rgb_full",)`` (RGB-only). ``capture_policy_cam``
                               is the deprecated bool alias (RGB columns only).
      detections_max:          when set, declares the four
                               ``observation.detections.*`` padded columns at
                               this slot count and requires ``detections`` on
                               every ``add_frame``; ``None`` (default) declares
                               none. ``lerobot_detections.DETECTIONS_MAX_DEFAULT``
                               is the recommended count for drivers that
                               capture detections.
      capture_git_sha:         git rev-parse HEAD of the repo at capture
                               time; recorded in per-episode metadata.
      scene_metadata_hash:     sha256 of the scene's canonical embedded
                               metadata dict (see ``hash_scene_metadata``).
      scene_metadata:          the scene metadata dict itself; when supplied it
                               is embedded at finalize as
                               ``meta/scenes/<scene_id>/scene_metadata.json`` so
                               pose-derived labels resolve offline. ``None``
                               travels only the hash.

    Async writers (defaults tuned for in-process teleop capture):

      image_writer_threads:    Threads for lerobot's :class:`AsyncImageWriter`.
                               Default 4 (≈2 per camera at our two-camera
                               configuration) so per-step PIL PNG encodes
                               run off the env-step thread. ``save_episode``
                               drains the queue, preserving episode bounds.
      depth_writer_threads:    Threads for the strafer depth-sidecar pool.
                               The depth PNG is authored outside the lerobot
                               dataset features, so lerobot's pool can't see
                               it — we run our own. Default 2 (depth is one
                               write per step, single-camera).
                               Set both to 0 for fully synchronous behaviour
                               (useful when debugging frame ordering).
    """

    def __init__(
        self,
        *,
        root: Path | str,
        repo_id: str,
        fps: int,
        capture_git_sha: str,
        scene_metadata_hash: str,
        scene_metadata: dict[str, Any] | None = None,
        cameras_required: Sequence[str] | None = None,
        capture_policy_cam: bool | None = None,
        detections_max: int | None = None,
        vcodec: str = "h264",
        operator_handle: str | None = None,
        session_id: str | None = None,
        image_writer_threads: int = 4,
        depth_writer_threads: int = 2,
    ) -> None:
        # Lazy import — keeps the module importable in environments
        # without lerobot installed (tests that only exercise the depth
        # sidecar, for example).
        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "lerobot is required to construct StraferLeRobotWriter. "
                "Install with `pip install --no-deps 'lerobot==0.5.1'` plus "
                "the matching `datasets`, `av`, and `jsonlines` pins, "
                "inside the Isaac Lab Python env. See "
                "docs/HARNESS_DATA_CAPTURE.md → One-time env setup for "
                "the full instructions.",
            ) from exc

        self._root = Path(root)
        self._cameras_required = _normalize_cameras_required(
            cameras_required, capture_policy_cam,
        )
        self._detections_max = (
            int(detections_max) if detections_max is not None else None
        )
        self._detection_vocab = (
            DetectionLabelVocab() if self._detections_max is not None else None
        )
        self._features = build_features(
            cameras_required=self._cameras_required,
            detections_max=self._detections_max,
        )
        self._capture_git_sha = str(capture_git_sha)
        self._scene_metadata_hash = str(scene_metadata_hash)
        self._scene_metadata = scene_metadata
        # Captured from the first episode; one scene per dataset is the invariant.
        self._sidecar_scene_id: str | None = None
        self._operator_handle = operator_handle
        self._session_id = session_id
        self._fps = int(fps)

        self._dataset = LeRobotDataset.create(
            repo_id=str(repo_id),
            fps=self._fps,
            features=self._features,
            root=self._root,
            robot_type="strafer",
            use_videos=True,
            vcodec=vcodec,
            image_writer_threads=int(image_writer_threads),
        )
        self._depth_pool: _DepthWriterPool | None = (
            _DepthWriterPool(int(depth_writer_threads))
            if depth_writer_threads and depth_writer_threads > 0 else None
        )
        self._open_episode: dict[str, Any] | None = None
        self._frame_count_this_episode = 0
        self._episode_extensions: list[StraferEpisodeExtensions] = []
        self._next_episode_index = 0
        self._finalized = False

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "StraferLeRobotWriter":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.finalize()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(
        self,
        *,
        mission_text: str,
        scene_id: str,
        target_label: str | None = None,
        target_object_id: str | None = None,
        target_position_3d: Sequence[float] | None = None,
        start_pose: Sequence[float] | None = None,
        source_driver: str = "teleop",
        source_mission_source: str = "scene-metadata",
        paraphrases: Sequence[str] | None = None,
        leg_initial_distance_m: float | None = None,
        episode_split: str | None = None,
        injection_mode: str | None = None,
        generator_metadata: dict[str, Any] | None = None,
        realized_d555_mount_quat: Sequence[float] | None = None,
    ) -> int:
        """Open a new episode buffer. Returns the assigned ``episode_index``."""
        if self._finalized:
            raise RuntimeError("begin_episode called after finalize()")
        if self._open_episode is not None:
            raise RuntimeError(
                "begin_episode called while another episode is open; "
                "call end_episode first",
            )
        if self._sidecar_scene_id is None:
            self._sidecar_scene_id = scene_id
        episode_index = self._next_episode_index
        split = episode_split or ("val" if episode_index % 10 == 0 else "train")
        self._open_episode = {
            "mission_text": mission_text,
            "scene_id": scene_id,
            "target_label": target_label,
            "target_object_id": target_object_id,
            "target_position_3d": (
                tuple(float(v) for v in target_position_3d)
                if target_position_3d is not None else None
            ),
            "start_pose": (
                tuple(float(v) for v in start_pose)
                if start_pose is not None else None
            ),
            "source_driver": source_driver,
            "source_mission_source": source_mission_source,
            "paraphrases": tuple(paraphrases or ()),
            "leg_initial_distance_m": (
                float(leg_initial_distance_m)
                if leg_initial_distance_m is not None else None
            ),
            "episode_split": split,
            "injection_mode": injection_mode,
            "generator_metadata": dict(generator_metadata or {}),
            "realized_d555_mount_quat": (
                tuple(float(v) for v in realized_d555_mount_quat)
                if realized_d555_mount_quat is not None else None
            ),
            "episode_index": episode_index,
        }
        self._frame_count_this_episode = 0
        return episode_index

    def _validate_camera_arg(
        self,
        token: str,
        arr: np.ndarray | None,
        res: tuple[int, int],
        argname: str,
        *,
        require_uint8: bool,
    ) -> None:
        """Enforce the declared sensor stack: a declared modality must be
        present (right dtype/shape); an undeclared one must be absent."""
        if token in self._cameras_required:
            if arr is None:
                raise ValueError(
                    f"{argname} is required: '{token}' is in the declared "
                    f"sensor stack {self._cameras_required}",
                )
            if require_uint8 and arr.dtype != np.uint8:
                raise ValueError(f"{argname} must be uint8; got {arr.dtype}")
            if arr.shape[:2] != res:
                raise ValueError(
                    f"{argname} shape {arr.shape[:2]} != expected {res}",
                )
        elif arr is not None:
            raise ValueError(
                f"{argname} provided but '{token}' is not in the declared "
                f"sensor stack {self._cameras_required}",
            )

    def add_frame(
        self,
        *,
        sim_time: float,
        pose: Sequence[float],
        achieved_vel: Sequence[float],
        action: Sequence[float],
        rgb_perception: np.ndarray | None = None,
        rgb_policy: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
        depth_policy_m: np.ndarray | None = None,
        detections: Sequence[DetectedBbox] | None = None,
    ) -> None:
        """Append a tick to the current episode buffer.

        Each camera argument is validated against the declared
        ``cameras_required`` stack: a declared modality must be supplied, an
        undeclared one must be omitted. ``rgb_*`` arrays are uint8
        ``(H, W, 3)``. ``depth_m`` is float32 ``(H, W)`` in meters for the
        perception camera (640x360); ``depth_policy_m`` the same for the
        policy camera (80x60). Declared depth modalities ride as 16UC1 PNG
        sidecars next to the LeRobot dataset proper.

        ``detections`` follows the same declared-modality discipline keyed
        on ``detections_max``: when declared it must be supplied every frame
        (an empty sequence means "no detections this frame" and packs as
        all-padding rows); when undeclared it must be omitted. Boxes are
        padded / truncated to ``detections_max`` rows and labels accumulate
        into ``meta/detection_labels.json`` — see
        :mod:`strafer_lab.tools.lerobot_detections`.
        """
        if self._finalized:
            raise RuntimeError("add_frame called after finalize()")
        if self._open_episode is None:
            raise RuntimeError(
                "add_frame called outside of an open episode; "
                "call begin_episode first",
            )

        self._validate_camera_arg(
            "rgb_full", rgb_perception, _PERCEPTION_RES, "rgb_perception",
            require_uint8=True,
        )
        self._validate_camera_arg(
            "rgb_policy", rgb_policy, _POLICY_RES, "rgb_policy",
            require_uint8=True,
        )
        self._validate_camera_arg(
            "depth_full", depth_m, _PERCEPTION_RES, "depth_m",
            require_uint8=False,
        )
        self._validate_camera_arg(
            "depth_policy", depth_policy_m, _POLICY_RES, "depth_policy_m",
            require_uint8=False,
        )
        if self._detections_max is not None:
            if detections is None:
                raise ValueError(
                    "detections is required every frame: the writer was "
                    f"constructed with detections_max={self._detections_max}; "
                    "pass an empty sequence for a frame with no detections",
                )
        elif detections is not None:
            raise ValueError(
                "detections provided but the writer was constructed without "
                "detections_max; the schema declares no detections columns",
            )

        state = np.asarray(list(pose) + list(achieved_vel), dtype=np.float32)
        action_arr = np.asarray(list(action), dtype=np.float32)
        frame: dict[str, Any] = {
            "observation.state": state,
            "action": action_arr,
            "task": self._open_episode["mission_text"],
        }
        if "rgb_full" in self._cameras_required:
            frame["observation.images.perception"] = rgb_perception
        # LeRobot's add_frame uses its own internal monotonic time
        # reference; we record sim_time separately by injecting it as
        # a metadata-side field on save_episode (not as a per-frame
        # column — adding one would require declaring it in features,
        # which inflates the schema). For now, sim_time per-frame is
        # captured via the LeRobot-built ``timestamp`` column derived
        # from ``fps``; sub-tick precision is recovered offline if
        # needed.
        _ = sim_time

        if "rgb_policy" in self._cameras_required:
            frame["observation.images.policy"] = rgb_policy
        if self._detections_max is not None:
            frame.update(
                pack_detections(
                    detections, self._detections_max, self._detection_vocab,
                ),
            )

        self._dataset.add_frame(frame)

        # Depth sidecars — written outside the LeRobot dataset proper, only
        # for the declared depth modalities. When a worker pool is configured,
        # hand each array off and let a background thread do the PIL encode +
        # fsync; that keeps the ~5–15 ms 16-bit PNG write off the env-step
        # thread. The pool copies the array because Isaac Sim recycles its
        # render buffer on the next step.
        ep_idx = self._open_episode["episode_index"]
        frame_idx = self._frame_count_this_episode
        for token, arr, feature in (
            ("depth_full", depth_m, PERCEPTION_DEPTH),
            ("depth_policy", depth_policy_m, POLICY_DEPTH),
        ):
            if token not in self._cameras_required:
                continue
            png_path = depth_frame_path(self._root, ep_idx, frame_idx, feature)
            if self._depth_pool is not None:
                self._depth_pool.submit(png_path, arr)
            else:
                write_depth_png(png_path, arr)

        self._frame_count_this_episode += 1

    def end_episode(
        self,
        *,
        outcome: str = "succeeded",
        outcome_category: str = "on_course",
        hard_negative_category: str | None = None,
        injection_mode_actual: str | None = None,
        original_target_position_3d: Sequence[float] | None = None,
        discard: bool = False,
    ) -> None:
        """Close the current episode buffer.

        When ``discard=True``, the underlying LeRobot ``save_episode`` is
        skipped (the episode buffer is cleared) and the episode index
        does **not** advance — the next ``begin_episode`` reuses this
        index. Discards never reach disk.
        """
        if self._finalized:
            raise RuntimeError("end_episode called after finalize()")
        if self._open_episode is None:
            raise RuntimeError(
                "end_episode called without an open episode",
            )
        episode_payload = self._open_episode
        self._open_episode = None

        if discard or self._frame_count_this_episode == 0:
            # Drain queued depth writes before clearing — otherwise a
            # worker could land a PNG into a directory whose episode index
            # is about to be reused, polluting the next episode.
            if self._depth_pool is not None:
                self._depth_pool.wait_until_done()
            # Clear LeRobot's per-frame buffer without persisting.
            try:
                self._dataset.clear_episode_buffer()
            except AttributeError:
                # Older lerobot — buffer key may differ. Tolerate by
                # bouncing the dataset state through writer internals.
                self._dataset.writer.clear_episode_buffer()
            return

        # Drain depth before lerobot's save_episode so both the MP4
        # consolidation and the depth PNG sequence reach disk before
        # the episode-index counter advances.
        if self._depth_pool is not None:
            self._depth_pool.wait_until_done()
        self._dataset.save_episode()

        ext = StraferEpisodeExtensions(
            episode_index=episode_payload["episode_index"],
            scene_id=episode_payload["scene_id"],
            target_label=episode_payload["target_label"],
            target_object_id=episode_payload["target_object_id"],
            target_position_3d=episode_payload["target_position_3d"],
            start_pose=episode_payload["start_pose"],
            outcome=outcome,
            outcome_category=outcome_category,
            paraphrases=episode_payload["paraphrases"],
            source_driver=episode_payload["source_driver"],
            source_mission_source=episode_payload["source_mission_source"],
            hard_negative_category=hard_negative_category,
            injection_mode=episode_payload["injection_mode"],
            injection_mode_actual=injection_mode_actual,
            original_target_position_3d=(
                tuple(float(v) for v in original_target_position_3d)
                if original_target_position_3d is not None else None
            ),
            operator_handle=self._operator_handle,
            session_id=self._session_id,
            generator_metadata=episode_payload["generator_metadata"],
            leg_initial_distance_m=episode_payload["leg_initial_distance_m"],
            episode_split=episode_payload["episode_split"],
            capture_git_sha=self._capture_git_sha,
            scene_metadata_hash=self._scene_metadata_hash,
            realized_d555_mount_quat=episode_payload["realized_d555_mount_quat"],
        )
        self._episode_extensions.append(ext)
        self._next_episode_index += 1

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def finalize(self) -> None:
        """Consolidate the LeRobot dataset + write the strafer sidecar.

        Idempotent — calling twice is a no-op after the first.
        """
        if self._finalized:
            return
        if self._open_episode is not None:
            # An episode is open at finalize time → discard rather than
            # persist a partial.
            self.end_episode(discard=True)
        # Drain + stop the depth pool before lerobot's finalize so any
        # in-flight depth PNGs land before the dataset is consolidated.
        if self._depth_pool is not None:
            try:
                self._depth_pool.wait_until_done()
            finally:
                self._depth_pool.stop()
                self._depth_pool = None
        self._dataset.finalize()
        self._write_strafer_sidecar()
        self._write_scene_metadata_sidecar()
        self._write_splits_sidecar()
        if self._detection_vocab is not None:
            # Always written when detections are declared — an empty
            # labels[] is the valid vocab for a capture that saw none.
            self._detection_vocab.write(self._root)
        self._finalized = True

    # ------------------------------------------------------------------
    # Introspection (for tests + downstream tools)
    # ------------------------------------------------------------------

    @property
    def root(self) -> Path:
        return self._root

    @property
    def num_episodes(self) -> int:
        return self._next_episode_index

    @property
    def features(self) -> dict[str, dict[str, Any]]:
        return dict(self._features)

    @property
    def cameras_required(self) -> tuple[str, ...]:
        """The declared sensor stack this writer's schema was built for."""
        return self._cameras_required

    @property
    def detections_max(self) -> int | None:
        """Declared detections slot count; ``None`` when not captured."""
        return self._detections_max

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_strafer_sidecar(self) -> None:
        """Persist the strafer per-episode extension parquet."""
        if not self._episode_extensions:
            return
        # pyarrow is a lerobot dep.
        import pyarrow as pa
        import pyarrow.parquet as pq

        records = [ext.as_record() for ext in self._episode_extensions]
        table = pa.Table.from_pylist(records)
        out_path = self._root / "meta" / "strafer_episodes.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)

    def _write_scene_metadata_sidecar(self) -> None:
        """Embed the scene metadata dict so pose-derived labels work offline.

        Only the hash travels in the per-episode columns; without the dict
        every region / room / narration label has to re-read the scene USD.
        """
        if not self._scene_metadata or self._sidecar_scene_id is None:
            return
        out_path = (
            self._root / "meta" / "scenes" / self._sidecar_scene_id
            / "scene_metadata.json"
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(self._scene_metadata, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    def _write_splits_sidecar(self) -> None:
        """Record named (non train/val) splits in ``meta/splits.jsonl``.

        Each named split present in the per-episode metadata becomes one
        whole-scene row ``{name, episode_indices, scope, description}`` so the
        held-out decision travels with the data. The optional sidecar is
        skipped when every episode used a default split.
        """
        named: dict[str, list[int]] = {}
        for ext in self._episode_extensions:
            if ext.episode_split in ("train", "val"):
                continue
            named.setdefault(ext.episode_split, []).append(int(ext.episode_index))
        if not named:
            return
        out_path = self._root / "meta" / "splits.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "name": name,
                "episode_indices": indices,
                "scope": "scene",
                "description": f"Whole-scene {name} selection assigned at capture time.",
            }
            for name, indices in sorted(named.items())
        ]
        out_path.write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )


# ---------------------------------------------------------------------------
# Public helpers (for consumers + tests)
# ---------------------------------------------------------------------------


def hash_scene_metadata(metadata: dict[str, Any]) -> str:
    """sha256-hexdigest of the canonical embedded scene-metadata dict.

    The metadata now lives in the scene USD's ``customData`` rather than a
    sidecar file, so the capture stamp hashes the canonical serialization
    of the dict (``json.dumps(..., sort_keys=True)``) instead of file
    bytes. Detects scene mutations across captures. Returns ``""`` for a
    falsy/empty dict so a missing scene degrades to an empty stamp rather
    than raising.
    """
    if not metadata:
        return ""
    canonical = json.dumps(metadata, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def read_strafer_episodes(dataset_root: Path | str) -> list[dict[str, Any]]:
    """Read the strafer extension parquet back as a list of dicts.

    Returns ``[]`` when the sidecar does not exist (dataset captured
    without strafer extensions).
    """
    import pyarrow.parquet as pq

    path = Path(dataset_root) / "meta" / "strafer_episodes.parquet"
    if not path.is_file():
        return []
    table = pq.read_table(path)
    return table.to_pylist()
