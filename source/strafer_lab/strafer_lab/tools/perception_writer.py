"""Per-frame perception data writer for Isaac Sim data collection.

Writes one episode directory per teleop run:

    <root>/
        episode_0000/
            frames.jsonl              # one JSON record per frame
            frame_0000.jpg             # RGB, JPEG quality 90 by default
            frame_0000.depth.npy       # optional, float32 meters
            frame_0001.jpg
            frame_0001.depth.npy
            ...

This is the layout the DGX-side batch pipeline already consumes —
:mod:`strafer_lab.scripts.generate_descriptions` iterates ``frames.jsonl``
files via :func:`iter_frame_records`, and
:mod:`strafer_lab.scripts.prepare_vlm_finetune_data` reads both the
JSONL records and the images referenced in them. Keeping the layout
identical here means Task 2 (perception data collection) and Tasks 9/12
(description pipeline, VLM SFT data prep) can be wired end-to-end with
no translation step between them.

The writer is deliberately pure-Python — it imports ``numpy`` and
``PIL.Image`` at module top, and nothing Isaac Sim / Omniverse related.
Callers pass it plain numpy arrays pulled from their own Isaac Sim
scene handles. That keeps it unit-testable from ``.venv_vlm`` via the
``strafer_lab`` namespace stub without needing AppLauncher.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image


_DEFAULT_JPEG_QUALITY = 90


@dataclass
class PerceptionWriterStats:
    """Aggregate counters for a writer session (single root dir)."""

    episodes_started: int = 0
    episodes_kept: int = 0
    episodes_discarded: int = 0
    frames_written: int = 0
    frames_dropped: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "episodes_started": self.episodes_started,
            "episodes_kept": self.episodes_kept,
            "episodes_discarded": self.episodes_discarded,
            "frames_written": self.frames_written,
            "frames_dropped": self.frames_dropped,
        }


@dataclass
class PerceptionFrameWriter:
    """Owns a single root directory and rotates per-episode subdirectories.

    Typical usage from a teleop loop::

        writer = PerceptionFrameWriter(output_root=Path("data/perception"))
        writer.begin_episode()
        while collecting:
            info = writer.save_frame(
                frame_id=step_idx,
                rgb=rgb_numpy,                    # (H, W, 3) uint8
                depth=depth_numpy,                # (H, W) float32 meters or None
                scene_name="scene_01",
                scene_type="infinigen",
                robot_pos=[x, y, z],
                robot_quat=[qx, qy, qz, qw],
                cam_pos=[x, y, z],
                cam_quat=[qx, qy, qz, qw],
                bboxes=[{...}, {...}],            # list of dicts from
                                                   # ReplicatorBboxExtractor
                image_width=640,
                image_height=360,
            )
            if done_button:
                writer.end_episode(keep=True)
                break
        writer.close()

    The class does NOT hold open file handles across frames — it appends
    one line to ``frames.jsonl`` per ``save_frame`` call and closes the
    file. This is resilient to crashes (partial episodes retain every
    frame written up to the crash) and matches how the DGX batch scripts
    consume the data (line-by-line streaming).
    """

    output_root: Path
    jpeg_quality: int = _DEFAULT_JPEG_QUALITY
    depth_enabled: bool = True
    episode_prefix: str = "episode_"
    episode_index_digits: int = 4

    stats: PerceptionWriterStats = field(default_factory=PerceptionWriterStats)

    # Internal — do not set directly from outside. Kept out of repr for
    # readable debug output.
    _episode_index: int = field(default=0, repr=False)
    _current_episode_dir: Path | None = field(default=None, repr=False)
    _current_frame_count: int = field(default=0, repr=False)
    _current_meta_path: Path | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.output_root = Path(self.output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        # Resume numbering from the highest existing episode index so
        # incremental collection across sessions does not clobber prior
        # episodes. Matches the collect_demos.py incremental-folder mode.
        self._episode_index = self._next_episode_index()

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def begin_episode(self) -> Path:
        """Allocate the next episode directory. Returns its path."""
        if self._current_episode_dir is not None:
            raise RuntimeError(
                "begin_episode called while an episode is already open; "
                "call end_episode first"
            )
        episode_dir = self._episode_dir_for(self._episode_index)
        episode_dir.mkdir(parents=True, exist_ok=True)
        self._current_episode_dir = episode_dir
        self._current_frame_count = 0
        self._current_meta_path = episode_dir / "frames.jsonl"
        # Truncate any stale frames.jsonl left behind by a previous crash
        # at the same index (should be rare given _next_episode_index,
        # but belt-and-suspenders for crash recovery).
        self._current_meta_path.write_text("")
        self.stats.episodes_started += 1
        return episode_dir

    def end_episode(self, *, keep: bool) -> None:
        """Finalise the current episode, advancing the index.

        Parameters
        ----------
        keep:
            If ``True``, stats are incremented and the episode directory
            is left on disk. If ``False``, the entire episode directory
            is wiped and the index does NOT advance so the next episode
            reuses the same slot.
        """
        if self._current_episode_dir is None:
            return

        if keep:
            self.stats.episodes_kept += 1
            self._episode_index += 1
        else:
            # Discard: delete the directory and its contents so the slot
            # is reusable on the next begin_episode call.
            self.stats.episodes_discarded += 1
            self._wipe_dir(self._current_episode_dir)

        self._current_episode_dir = None
        self._current_frame_count = 0
        self._current_meta_path = None

    # ------------------------------------------------------------------
    # Per-frame write
    # ------------------------------------------------------------------

    def save_frame(
        self,
        *,
        frame_id: int | str,
        rgb: np.ndarray,
        depth: np.ndarray | None,
        scene_name: str,
        scene_type: str,
        robot_pos: list[float] | tuple[float, ...] | np.ndarray,
        robot_quat: list[float] | tuple[float, ...] | np.ndarray,
        cam_pos: list[float] | tuple[float, ...] | np.ndarray | None = None,
        cam_quat: list[float] | tuple[float, ...] | np.ndarray | None = None,
        bboxes: list[Mapping[str, Any]] | None = None,
        image_width: int | None = None,
        image_height: int | None = None,
        extras: Mapping[str, Any] | None = None,
    ) -> str:
        """Write one frame to the current episode directory.

        Returns the relative image path (``frame_0000.jpg``) that will
        appear in the ``frames.jsonl`` record so callers can log it.

        Parameters
        ----------
        rgb:
            ``(H, W, 3)`` uint8 numpy array. Isaac Sim's RGB output is
            typically ``(H, W, 4)`` RGBA — callers should slice to
            ``[..., :3]`` before passing here.
        depth:
            ``(H, W)`` float32 distance-to-image-plane in meters, or
            ``None`` if depth capture is disabled. When non-None and
            ``depth_enabled`` is True, saved as ``frame_NNNN.depth.npy``.
        image_width / image_height:
            Pixel dimensions for the frames.jsonl record. If not given,
            inferred from ``rgb.shape``. Downstream consumers
            (``prepare_vlm_finetune_data``) need these to rescale pixel
            bboxes into Qwen's 0..1000 coordinate space.
        extras:
            Additional key/value pairs to merge into the JSONL record.
            Useful for recording teleop metadata (gamepad input, etc.).
        """
        if self._current_episode_dir is None or self._current_meta_path is None:
            raise RuntimeError(
                "save_frame called outside of an open episode; "
                "call begin_episode first"
            )

        if rgb.ndim != 3 or rgb.shape[2] not in (3, 4):
            raise ValueError(
                f"rgb must be (H, W, 3) or (H, W, 4), got shape {rgb.shape}"
            )
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            # Be forgiving: Isaac Sim sometimes returns float; clamp and cast.
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        frame_basename = f"frame_{self._current_frame_count:04d}"
        image_rel = f"{frame_basename}.jpg"
        image_path = self._current_episode_dir / image_rel
        Image.fromarray(rgb, mode="RGB").save(
            image_path, format="JPEG", quality=self.jpeg_quality,
        )

        depth_rel: str | None = None
        if depth is not None and self.depth_enabled:
            if depth.ndim == 3 and depth.shape[-1] == 1:
                depth = depth[..., 0]
            if depth.ndim != 2:
                raise ValueError(
                    f"depth must be (H, W) or (H, W, 1), got shape {depth.shape}"
                )
            depth_rel = f"{frame_basename}.depth.npy"
            np.save(
                self._current_episode_dir / depth_rel,
                depth.astype(np.float32),
            )

        if image_width is None:
            image_height_eff = int(rgb.shape[0])
            image_width_eff = int(rgb.shape[1])
        else:
            image_width_eff = int(image_width)
            image_height_eff = int(
                image_height if image_height is not None else rgb.shape[0]
            )

        record: dict[str, Any] = {
            "frame_id": frame_id,
            "image_path": image_rel,
            "image_width": image_width_eff,
            "image_height": image_height_eff,
            "scene_name": scene_name,
            "scene_type": scene_type,
            "robot_pos": _to_list(robot_pos),
            "robot_quat": _to_list(robot_quat),
            "bboxes": list(bboxes) if bboxes is not None else [],
        }
        if depth_rel is not None:
            record["depth_path"] = depth_rel
        if cam_pos is not None:
            record["cam_pos"] = _to_list(cam_pos)
        if cam_quat is not None:
            record["cam_quat"] = _to_list(cam_quat)
        if extras:
            for key, value in extras.items():
                if key not in record:
                    # Do not eagerly jsonify — json.dumps(default=_jsonify)
                    # below handles nested dicts / lists natively and falls
                    # back to the coercer only for unknown types (numpy
                    # scalars). Eagerly calling _jsonify here would
                    # stringify ordinary dicts via its repr() fallback.
                    record[key] = value

        with self._current_meta_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=_jsonify) + "\n")

        self._current_frame_count += 1
        self.stats.frames_written += 1
        return image_rel

    def drop_frame(self) -> None:
        """Record a frame as dropped without writing anything to disk.

        Useful when the teleop loop receives a frame but decides to skip
        it (e.g., first frame after reset has stale bbox data, or the
        gamepad sent no input and the user wants to throttle storage).
        """
        self.stats.frames_dropped += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Finalise any in-progress episode (kept) and stop accepting writes."""
        if self._current_episode_dir is not None:
            self.end_episode(keep=True)

    @property
    def next_episode_index(self) -> int:
        return self._episode_index

    @property
    def current_episode_dir(self) -> Path | None:
        return self._current_episode_dir

    @property
    def current_frame_count(self) -> int:
        return self._current_frame_count

    def write_stats(self, path: Path | None = None) -> Path:
        """Persist stats to ``<output_root>/writer_stats.json`` (or ``path``).

        Returns the file path written. Called by the teleop script on exit.
        """
        target = Path(path) if path is not None else self.output_root / "writer_stats.json"
        target.write_text(json.dumps(self.stats.to_dict(), indent=2))
        return target

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _episode_dir_for(self, index: int) -> Path:
        name = f"{self.episode_prefix}{index:0{self.episode_index_digits}d}"
        return self.output_root / name

    def _next_episode_index(self) -> int:
        """Find the next free episode index in :attr:`output_root`."""
        if not self.output_root.is_dir():
            return 0
        highest = -1
        for child in self.output_root.iterdir():
            if not child.is_dir() or not child.name.startswith(self.episode_prefix):
                continue
            tail = child.name[len(self.episode_prefix):]
            try:
                idx = int(tail)
            except ValueError:
                continue
            if idx > highest:
                highest = idx
        return highest + 1

    @staticmethod
    def _wipe_dir(path: Path) -> None:
        """Recursively delete ``path`` (used for discarded episodes)."""
        if not path.exists():
            return
        for child in sorted(path.iterdir(), reverse=True):
            if child.is_dir():
                PerceptionFrameWriter._wipe_dir(child)
            else:
                child.unlink()
        path.rmdir()


# ---------------------------------------------------------------------------
# JSON coercion helpers — handle numpy scalars / arrays the teleop loop may
# pass through without forcing callers to cast everything by hand.
# ---------------------------------------------------------------------------


def _to_list(value: Any) -> list:
    """Coerce numpy arrays / tuples / torch tensors to a plain Python list."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        return value.tolist()  # type: ignore[no-any-return]
    if isinstance(value, (list, tuple)):
        return [float(v) for v in value]
    return [float(value)]


def _jsonify(value: Any) -> Any:
    """Fallback json.dumps ``default=`` coercer for numpy scalars.

    Handles numpy float32, int32, bool_, and ndarrays without forcing
    callers to pre-convert. Anything else is converted via ``repr`` as a
    last resort so serialization does not explode mid-frame.
    """
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (bytes, bytearray)):
        return value.decode("utf-8", errors="replace")
    try:
        return repr(value)
    except Exception:  # pragma: no cover - extremely defensive
        return None
