"""16UC1 PNG depth sidecar for strafer LeRobot v3 datasets.

LeRobot v3's video pipeline is MP4-only; depth at 16-bit precision can't
ride that path (no 16-bit video codec). Strafer ships depth as a sidecar
PNG sequence per-episode, deterministically named so consumers find each
frame from the parquet's ``(episode_index, frame_index)``.

Layout per dataset root::

    <root>/videos/observation.depth.perception/
      episode-000000/
        000000.png        # 16-bit single-channel PNG, millimeters
        000001.png
        ...

Format: 16UC1 little-endian PNG, depth in millimeters. Matches the
real-robot perception stack's convention in
``strafer_perception/depth_downsampler.py`` so sim and real depth share
one format.

This module is pure-Python — no torch, no LeRobot. Unit-testable from
``.venv_harness`` against synthetic arrays. Callers (writer +
consumers) import from here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image


# Default feature-name for the perception-camera depth sidecar. Other
# cameras (e.g. the policy camera) ship under their own feature-name —
# pass it explicitly to ``depth_root`` / ``episode_dir`` / ``frame_path``.
PERCEPTION_DEPTH = "observation.depth.perception"
POLICY_DEPTH = "observation.depth.policy"

_DEPTH_SCALE_M_PER_UNIT = 0.001  # 16UC1 unit = 1 mm = 0.001 m


def depth_root(
    dataset_root: Path | str,
    feature: str = PERCEPTION_DEPTH,
) -> Path:
    """Return ``<root>/videos/<feature>/``."""
    return Path(dataset_root) / "videos" / feature


def episode_dir(
    dataset_root: Path | str,
    episode_index: int,
    feature: str = PERCEPTION_DEPTH,
) -> Path:
    """Return the per-episode depth directory.

    Naming convention: ``episode-{episode_index:06d}``. The 6-digit width
    matches LeRobot v3's per-shard file naming convention and stays
    sortable up to ~1M episodes.
    """
    return depth_root(dataset_root, feature) / f"episode-{int(episode_index):06d}"


def frame_path(
    dataset_root: Path | str,
    episode_index: int,
    frame_index: int,
    feature: str = PERCEPTION_DEPTH,
) -> Path:
    """Return the depth PNG path for one frame."""
    return episode_dir(dataset_root, episode_index, feature) / f"{int(frame_index):06d}.png"


def write_depth_png(
    path: Path | str,
    depth_m: np.ndarray,
) -> None:
    """Write a (H, W) float32 depth-in-meters array as a 16UC1 PNG.

    Quantization: ``depth_m * 1000`` (1mm precision). Values are clipped to
    the 16-bit range ``[0, 65535]`` mm = ``[0, 65.535]`` m, which spans
    well beyond the D555's effective range (~6m indoors). NaN / Inf become
    0 (invalid depth, matches the real driver's convention).
    """
    arr = np.asarray(depth_m)
    if arr.ndim != 2:
        raise ValueError(
            f"depth_m must be (H, W); got shape {arr.shape}",
        )
    finite = np.where(np.isfinite(arr), arr, 0.0)
    mm = np.clip(np.rint(finite * 1000.0), 0.0, 65535.0).astype(np.uint16)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mm, mode="I;16").save(path, format="PNG")


def read_depth_png(path: Path | str) -> np.ndarray:
    """Decode a 16UC1 PNG to a (H, W) float32 array in meters.

    Inverse of :func:`write_depth_png`. Zero-valued pixels (the
    "invalid depth" sentinel) come back as 0.0 m, not NaN — callers
    that need a mask should compare against 0.0 themselves to keep the
    semantics explicit.
    """
    img = Image.open(Path(path))
    arr = np.asarray(img, dtype=np.uint16)
    return (arr.astype(np.float32) * _DEPTH_SCALE_M_PER_UNIT).astype(np.float32)


def episode_frame_paths(
    dataset_root: Path | str,
    episode_index: int,
    feature: str = PERCEPTION_DEPTH,
) -> Iterable[Path]:
    """Yield depth PNG paths for one episode in frame order."""
    ep_dir = episode_dir(dataset_root, episode_index, feature)
    if not ep_dir.is_dir():
        return iter(())
    return iter(sorted(ep_dir.glob("*.png")))
