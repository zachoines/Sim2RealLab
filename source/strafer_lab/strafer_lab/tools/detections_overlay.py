"""Render a capture's recorded detections as an annotated video.

Detections are stored as **columns** in a strafer LeRobot dataset
(``observation.detections.{bbox,label_id,valid,occlusion}``), not painted
into the camera video — see the Detections section of
``harness-architecture``. This tool is the post-hoc visualization layer:
it decodes a camera video and draws the recorded boxes + class labels on
top, writing an annotated MP4. It is pure read-side — works on **any**
strafer capture regardless of how the detections were produced.

The drawing is a pure function (:func:`draw_detection_boxes`); the
dataset plumbing (:func:`overlay_detections_video`) is the I/O wrapper.

CLI::

    python -m strafer_lab.tools.detections_overlay \\
        --dataset data/sim_in_the_loop/<scene>_<run> \\
        --output  docs/artifacts/<name>.mp4
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Any, Sequence

from strafer_lab.tools.lerobot_detections import (
    DETECTIONS_BBOX,
    DETECTIONS_LABEL_ID,
    DETECTIONS_OCCLUSION,
    DETECTIONS_VALID,
    read_detection_labels,
)

DEFAULT_CAMERA_KEY = "observation.images.perception"

# Deterministic per-class colours (BGR, for OpenCV). Cycled by label id so
# the same class is the same colour across frames and runs.
_PALETTE_BGR: tuple[tuple[int, int, int], ...] = (
    (0, 255, 0), (0, 165, 255), (255, 128, 0), (0, 0, 255), (255, 0, 255),
    (255, 255, 0), (128, 0, 255), (0, 255, 255), (180, 105, 255), (0, 128, 0),
)


def _color_for(label_id: int) -> tuple[int, int, int]:
    return _PALETTE_BGR[int(label_id) % len(_PALETTE_BGR)]


def draw_detection_boxes(
    frame_bgr: Any,
    bboxes: Sequence[Sequence[float]],
    label_ids: Sequence[int],
    valid: Sequence[bool],
    labels: Sequence[str],
    *,
    occlusions: Sequence[float] | None = None,
    occlusion_max: float | None = None,
) -> Any:
    """Draw the valid detection boxes + class names on ``frame_bgr`` in place.

    ``bboxes`` are pixel ``(x_min, y_min, x_max, y_max)`` in the frame's
    resolution. Rows where ``valid`` is false are skipped (padding). When
    ``occlusion_max`` is set, rows whose ``occlusions`` exceed it are
    skipped too. ``label_ids`` index ``labels``. Returns the frame.
    """
    import cv2  # noqa: WPS433 — heavy optional dep, imported lazily

    h, w = frame_bgr.shape[:2]
    for i, (box, lid, is_valid) in enumerate(zip(bboxes, label_ids, valid)):
        if not bool(is_valid):
            continue
        if (
            occlusion_max is not None
            and occlusions is not None
            and float(occlusions[i]) > occlusion_max
        ):
            continue
        lid = int(lid)
        name = labels[lid] if 0 <= lid < len(labels) else str(lid)
        color = _color_for(lid)
        x1, y1, x2, y2 = (int(round(c)) for c in box)
        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
        ty = y1 - 4 if y1 - 4 > 8 else min(h - 2, y2 + 14)
        cv2.putText(
            frame_bgr, name, (max(0, x1), ty),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return frame_bgr


def _dataset_fps(dataset_root: Path, default: int = 8) -> int:
    info = dataset_root / "meta" / "info.json"
    if info.is_file():
        try:
            return int(json.loads(info.read_text(encoding="utf-8")).get("fps", default))
        except (ValueError, TypeError):
            pass
    return default


def overlay_detections_video(
    dataset_root: Path | str,
    output_path: Path | str,
    *,
    episode_index: int = 0,
    camera_key: str = DEFAULT_CAMERA_KEY,
    fps: int | None = None,
    occlusion_max: float | None = None,
) -> Path:
    """Write an annotated MP4 of ``episode_index``'s detections.

    Decodes the ``camera_key`` video and draws each frame's recorded
    detection boxes. Frames are matched to detection rows by their shared
    sequential order (LeRobot writes a row and a video frame per step).
    Returns the output path. Needs ``opencv`` + ``pyarrow``.
    """
    import cv2  # noqa: WPS433
    import numpy as np
    import pyarrow.parquet as pq

    root = Path(dataset_root)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = read_detection_labels(root)
    fps = fps if fps is not None else _dataset_fps(root)

    data_shards = sorted(glob.glob(str(root / "data" / "**" / "*.parquet"), recursive=True))
    if not data_shards:
        raise FileNotFoundError(f"no data parquet under {root}/data/")
    video_shards = sorted(
        glob.glob(str(root / "videos" / camera_key / "**" / "*.mp4"), recursive=True)
    )
    if not video_shards:
        raise FileNotFoundError(f"no {camera_key} video under {root}/videos/")

    # Detection rows, in global capture order (matches video-frame order).
    cols = [DETECTIONS_BBOX, DETECTIONS_LABEL_ID, DETECTIONS_VALID, DETECTIONS_OCCLUSION,
            "episode_index"]
    rows: dict[str, list] = {c: [] for c in cols}
    for shard in data_shards:
        tbl = pq.read_table(shard)
        present = [c for c in cols if c in tbl.column_names]
        d = tbl.select(present).to_pydict()
        n = len(d[present[0]])
        for c in cols:
            rows[c].extend(d.get(c, [None] * n))

    writer = None
    written = 0
    frame_cursor = 0  # global frame index across video shards
    try:
        for shard in video_shards:
            cap = cv2.VideoCapture(shard)
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                idx = frame_cursor
                frame_cursor += 1
                if idx >= len(rows[DETECTIONS_VALID]):
                    break
                if rows["episode_index"][idx] not in (None, episode_index):
                    continue
                bbox = np.asarray(rows[DETECTIONS_BBOX][idx]).reshape(-1, 4)
                lid = np.asarray(rows[DETECTIONS_LABEL_ID][idx]).reshape(-1)
                valid = np.asarray(rows[DETECTIONS_VALID][idx], dtype=bool).reshape(-1)
                occ = np.asarray(rows[DETECTIONS_OCCLUSION][idx]).reshape(-1)
                draw_detection_boxes(
                    frame, bbox, lid, valid, labels,
                    occlusions=occ, occlusion_max=occlusion_max,
                )
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(
                        str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h),
                    )
                writer.write(frame)
                written += 1
            cap.release()
    finally:
        if writer is not None:
            writer.release()

    if written == 0:
        raise RuntimeError(
            f"no frames written for episode {episode_index} from {root}"
        )
    return output_path


def _cli_main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Capture dataset root.")
    parser.add_argument("--output", required=True, help="Annotated MP4 path.")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--camera", default=DEFAULT_CAMERA_KEY)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument(
        "--occlusion-max", type=float, default=None,
        help="Drop boxes whose occlusion ratio exceeds this (0..1).",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)
    out = overlay_detections_video(
        args.dataset, args.output,
        episode_index=args.episode, camera_key=args.camera,
        fps=args.fps, occlusion_max=args.occlusion_max,
    )
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli_main())
