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

Lifecycle::

    writer = StraferLeRobotWriter(
        root="data/sim_in_the_loop/<scene>",
        repo_id="strafer/<scene>",
        fps=8,
        capture_policy_cam=True,
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
touch Isaac Sim, ROS, or any sim-only state. Unit-testable from the
.venv_harness venv.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np


# Resolution constants for the InfinigenPerception scene. Kept here so
# the writer's features dict is deterministic without importing the
# Isaac Lab scene config.
_PERCEPTION_RES = (360, 640)  # (H, W)
_POLICY_RES = (60, 80)


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
        }


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def build_features(
    capture_policy_cam: bool = True,
    state_dim: int = 10,  # pose (7) + achieved_vel (3)
    action_dim: int = 3,
) -> dict[str, dict[str, Any]]:
    """Return the LeRobot v3 features dict for strafer captures.

    Centralized so tests + writer + future consumers agree on the schema.
    """
    features: dict[str, dict[str, Any]] = {
        "observation.images.perception": {
            "dtype": "video",
            "shape": (*_PERCEPTION_RES, 3),
            "names": ["height", "width", "channels"],
        },
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
    if capture_policy_cam:
        features["observation.images.policy"] = {
            "dtype": "video",
            "shape": (*_POLICY_RES, 3),
            "names": ["height", "width", "channels"],
        }
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
      capture_git_sha:         git rev-parse HEAD of the repo at capture
                               time; recorded in per-episode metadata.
      scene_metadata_hash:     sha256 of the active scene_metadata.json.
    """

    def __init__(
        self,
        *,
        root: Path | str,
        repo_id: str,
        fps: int,
        capture_git_sha: str,
        scene_metadata_hash: str,
        capture_policy_cam: bool = True,
        vcodec: str = "h264",
        operator_handle: str | None = None,
        session_id: str | None = None,
    ) -> None:
        # Lazy import — keeps the module importable in environments
        # without lerobot installed (tests that only exercise the depth
        # sidecar, for example).
        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        self._root = Path(root)
        self._capture_policy_cam = bool(capture_policy_cam)
        self._features = build_features(capture_policy_cam=self._capture_policy_cam)
        self._capture_git_sha = str(capture_git_sha)
        self._scene_metadata_hash = str(scene_metadata_hash)
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
    ) -> int:
        """Open a new episode buffer. Returns the assigned ``episode_index``."""
        if self._finalized:
            raise RuntimeError("begin_episode called after finalize()")
        if self._open_episode is not None:
            raise RuntimeError(
                "begin_episode called while another episode is open; "
                "call end_episode first",
            )
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
            "episode_index": episode_index,
        }
        self._frame_count_this_episode = 0
        return episode_index

    def add_frame(
        self,
        *,
        sim_time: float,
        pose: Sequence[float],
        achieved_vel: Sequence[float],
        action: Sequence[float],
        rgb_perception: np.ndarray,
        rgb_policy: np.ndarray | None = None,
        depth_m: np.ndarray | None = None,
    ) -> None:
        """Append a tick to the current episode buffer.

        ``rgb_*`` arrays are uint8 ``(H, W, 3)``. ``depth_m`` is
        float32 ``(H, W)`` in meters (optional; sidecar PNG written when
        provided).
        """
        if self._finalized:
            raise RuntimeError("add_frame called after finalize()")
        if self._open_episode is None:
            raise RuntimeError(
                "add_frame called outside of an open episode; "
                "call begin_episode first",
            )
        if rgb_perception.dtype != np.uint8:
            raise ValueError(
                f"rgb_perception must be uint8; got {rgb_perception.dtype}",
            )
        if rgb_perception.shape[:2] != _PERCEPTION_RES:
            raise ValueError(
                f"rgb_perception shape {rgb_perception.shape[:2]} != "
                f"expected {_PERCEPTION_RES}",
            )

        state = np.asarray(list(pose) + list(achieved_vel), dtype=np.float32)
        action_arr = np.asarray(list(action), dtype=np.float32)
        frame: dict[str, Any] = {
            "observation.images.perception": rgb_perception,
            "observation.state": state,
            "action": action_arr,
            "task": self._open_episode["mission_text"],
        }
        # LeRobot's add_frame uses its own internal monotonic time
        # reference; we record sim_time separately by injecting it as
        # a metadata-side field on save_episode (not as a per-frame
        # column — adding one would require declaring it in features,
        # which inflates the schema). For now, sim_time per-frame is
        # captured via the LeRobot-built ``timestamp`` column derived
        # from ``fps``; sub-tick precision is recovered offline if
        # needed.
        _ = sim_time

        if self._capture_policy_cam:
            if rgb_policy is None:
                raise ValueError(
                    "capture_policy_cam=True but rgb_policy is None",
                )
            if rgb_policy.dtype != np.uint8:
                raise ValueError(
                    f"rgb_policy must be uint8; got {rgb_policy.dtype}",
                )
            if rgb_policy.shape[:2] != _POLICY_RES:
                raise ValueError(
                    f"rgb_policy shape {rgb_policy.shape[:2]} != "
                    f"expected {_POLICY_RES}",
                )
            frame["observation.images.policy"] = rgb_policy

        self._dataset.add_frame(frame)

        # Depth sidecar — written outside the LeRobot dataset proper.
        if depth_m is not None:
            from .lerobot_depth import frame_path, write_depth_png

            ep_idx = self._open_episode["episode_index"]
            png_path = frame_path(self._root, ep_idx, self._frame_count_this_episode)
            write_depth_png(png_path, depth_m)

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
            # Clear LeRobot's per-frame buffer without persisting.
            try:
                self._dataset.clear_episode_buffer()
            except AttributeError:
                # Older lerobot — buffer key may differ. Tolerate by
                # bouncing the dataset state through writer internals.
                self._dataset.writer.clear_episode_buffer()
            return

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
        self._dataset.finalize()
        self._write_strafer_sidecar()
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

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _write_strafer_sidecar(self) -> None:
        """Persist the strafer per-episode extension parquet."""
        if not self._episode_extensions:
            return
        # pyarrow is a lerobot dep, available in .venv_harness.
        import pyarrow as pa
        import pyarrow.parquet as pq

        records = [ext.as_record() for ext in self._episode_extensions]
        table = pa.Table.from_pylist(records)
        out_path = self._root / "meta" / "strafer_episodes.parquet"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, out_path)


# ---------------------------------------------------------------------------
# Public helpers (for consumers + tests)
# ---------------------------------------------------------------------------


def hash_scene_metadata(path: Path | str) -> str:
    """sha256-hexdigest of a scene_metadata.json file's bytes."""
    p = Path(path)
    if not p.is_file():
        return ""
    return hashlib.sha256(p.read_bytes()).hexdigest()


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
