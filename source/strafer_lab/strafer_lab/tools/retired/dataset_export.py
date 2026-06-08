"""Export perception data + descriptions into training-ready formats.

Produces two quick-iteration formats from the outputs of the scene
description pipeline (:mod:`strafer_lab.scripts.generate_descriptions`)
and the scene metadata (:mod:`strafer_lab.scripts.extract_scene_metadata`):

1. ``clip_descriptions.csv`` — (image_path, description) pairs for
   OpenCLIP contrastive fine-tuning. Multiple descriptions per image
   are emitted as separate rows.

2. ``vlm_grounding.jsonl`` — basic Qwen2.5-VL grounding SFT format:
   ``{"image": ..., "conversations": [user, assistant]}`` with
   ``<ref>label</ref><box>(x1,y1),(x2,y2)</box>`` coordinates scaled to
   the 0..1000 range Qwen expects. This is a *subset* of the full SFT
   dataset — :mod:`strafer_lab.scripts.prepare_vlm_finetune_data` is
   the comprehensive producer that adds negatives, multi-object
   examples, and description-preservation examples.

Both outputs exclude frames from ``scene_type == "procroom"``: the
primitive-shape ProcRoom scenes (solid-color boxes / cylinders) do not
transfer to real rooms for VLM or CLIP training and would only add
noise to the dataset.
"""

from __future__ import annotations

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

logger = logging.getLogger("dataset_export")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ExportStats:
    frames_seen: int = 0
    procroom_skipped: int = 0
    frames_without_descriptions: int = 0
    clip_rows: int = 0
    vlm_positive_rows: int = 0
    vlm_negative_rows: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


@dataclass
class ExportOptions:
    perception_root: Path
    descriptions_root: Path
    output_root: Path
    scene_metadata_dir: Path | None = None
    include_clip: bool = True
    include_vlm: bool = True
    vlm_negative_ratio: int = 3
    max_image_side: int = 1024
    seed: int = 0


# ---------------------------------------------------------------------------
# Coordinate helpers
# ---------------------------------------------------------------------------


def pixel_bbox_to_qwen(
    bbox: tuple[int, int, int, int], *, image_width: int, image_height: int,
) -> tuple[int, int, int, int]:
    """Scale a pixel bbox into Qwen's normalized 0..1000 coordinate space.

    The returned coordinates are clamped to ``[0, 1000]`` and at least
    one pixel wide/tall so the downstream SFT tokenizer sees a valid
    box even when the source bbox was degenerate.
    """
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image_width and image_height must be positive")
    x1, y1, x2, y2 = bbox
    sx = 1000.0 / image_width
    sy = 1000.0 / image_height
    nx1 = max(0, min(1000, int(round(x1 * sx))))
    nx2 = max(0, min(1000, int(round(x2 * sx))))
    ny1 = max(0, min(1000, int(round(y1 * sy))))
    ny2 = max(0, min(1000, int(round(y2 * sy))))
    if nx2 <= nx1:
        nx2 = min(1000, nx1 + 1)
    if ny2 <= ny1:
        ny2 = min(1000, ny1 + 1)
    return (nx1, ny1, nx2, ny2)


def format_qwen_grounding_answer(
    label: str, bbox: tuple[int, int, int, int],
) -> str:
    """Format one ``<ref>label</ref><box>(x1,y1),(x2,y2)</box>`` reply."""
    x1, y1, x2, y2 = bbox
    return f"<ref>{label}</ref><box>({x1},{y1}),({x2},{y2})</box>"


# ---------------------------------------------------------------------------
# Iterators
# ---------------------------------------------------------------------------


def iter_description_records(descriptions_root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield (episode_dir, record) pairs from the description pipeline output."""
    descriptions_root = Path(descriptions_root)
    if not descriptions_root.exists():
        return
    for episode_dir in sorted(descriptions_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        path = episode_dir / "descriptions.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line in %s: %s", path, exc)
                    continue
                if isinstance(record, dict):
                    yield episode_dir, record


def iter_perception_frames(perception_root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield (episode_dir, record) pairs from the perception data tree."""
    perception_root = Path(perception_root)
    if not perception_root.exists():
        return
    for episode_dir in sorted(perception_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        path = episode_dir / "frames.jsonl"
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Skipping malformed line in %s: %s", path, exc)
                    continue
                if isinstance(record, dict):
                    yield episode_dir, record


# ---------------------------------------------------------------------------
# CLIP CSV export
# ---------------------------------------------------------------------------


def export_clip_csv(
    *,
    descriptions_root: Path,
    output_path: Path,
    stats: ExportStats,
) -> None:
    """Write ``image_path,description`` rows for every validated description.

    ``image_path`` is stored as ``{episode_name}/{relative_path}`` so the
    CLIP trainer can resolve it against the shared perception root.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "description"])
        for episode_dir, record in iter_description_records(descriptions_root):
            stats.frames_seen += 1
            if record.get("scene_type") == "procroom":
                stats.procroom_skipped += 1
                continue
            descriptions = record.get("descriptions") or []
            if not descriptions:
                stats.frames_without_descriptions += 1
                continue
            image_rel = record.get("image_path") or ""
            if not image_rel:
                continue
            full_image_path = f"{episode_dir.name}/{image_rel}"
            for desc in descriptions:
                text = str(desc.get("text", "")).strip()
                if not text:
                    continue
                writer.writerow([full_image_path, text])
                stats.clip_rows += 1


# ---------------------------------------------------------------------------
# VLM JSONL export
# ---------------------------------------------------------------------------


def _grounding_example(
    *,
    image_rel: str,
    label: str,
    bbox_qwen: tuple[int, int, int, int],
) -> dict[str, Any]:
    return {
        "image": image_rel,
        "conversations": [
            {"role": "user", "content": f"<image>Locate the {label} in this image."},
            {
                "role": "assistant",
                "content": format_qwen_grounding_answer(label, bbox_qwen),
            },
        ],
    }


def _negative_example(*, image_rel: str, label: str) -> dict[str, Any]:
    return {
        "image": image_rel,
        "conversations": [
            {"role": "user", "content": f"<image>Locate the {label} in this image."},
            {
                "role": "assistant",
                "content": "The object is not visible in this image.",
            },
        ],
    }


def export_vlm_grounding_jsonl(
    *,
    perception_root: Path,
    descriptions_root: Path | None,
    output_path: Path,
    scene_metadata_dir: Path | None,
    negative_ratio: int,
    stats: ExportStats,
    seed: int = 0,
) -> None:
    """Emit single-object grounding JSONL examples.

    Positive examples come from the frame's labelled bboxes. Negative
    examples are produced by sampling ``negative_ratio × positives``
    labels from the scene's label set that are NOT present in the
    frame, then emitting the "not visible" response template.
    """
    import random

    # We use scene metadata (when available) to generate negatives, but
    # perception frames remain the source of positives.
    label_sets: dict[str, set[str]] = {}

    def _scene_labels_for(scene_name: str) -> set[str]:
        if scene_name in label_sets:
            return label_sets[scene_name]
        if scene_metadata_dir is None:
            label_sets[scene_name] = set()
            return label_sets[scene_name]
        path = Path(scene_metadata_dir) / scene_name / "scene_metadata.json"
        if not path.exists():
            label_sets[scene_name] = set()
            return label_sets[scene_name]
        try:
            with path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except json.JSONDecodeError:
            label_sets[scene_name] = set()
            return label_sets[scene_name]
        labels = {
            str(obj.get("label"))
            for obj in metadata.get("objects", []) or []
            if obj.get("label")
        }
        label_sets[scene_name] = labels
        return labels

    rng = random.Random(seed)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for episode_dir, record in iter_perception_frames(perception_root):
            if record.get("scene_type") == "procroom":
                stats.procroom_skipped += 1
                continue

            image_rel_raw = record.get("image_path") or ""
            if not image_rel_raw:
                continue
            image_rel = f"{episode_dir.name}/{image_rel_raw}"
            image_width = int(record.get("image_width") or 0)
            image_height = int(record.get("image_height") or 0)
            if image_width <= 0 or image_height <= 0:
                # Without image dims we cannot rescale pixel bboxes to
                # Qwen's 0..1000 space. Skip the frame rather than
                # emitting incorrect coordinates.
                continue

            bboxes = record.get("bboxes") or []
            positive_labels: list[str] = []
            for entry in bboxes:
                if not isinstance(entry, dict):
                    continue
                label = entry.get("label")
                bbox_raw = entry.get("bbox_2d")
                if not isinstance(label, str) or not label:
                    continue
                if not bbox_raw or len(bbox_raw) != 4:
                    continue
                try:
                    pixel_bbox = (
                        int(bbox_raw[0]),
                        int(bbox_raw[1]),
                        int(bbox_raw[2]),
                        int(bbox_raw[3]),
                    )
                except (TypeError, ValueError):
                    continue
                try:
                    qwen_bbox = pixel_bbox_to_qwen(
                        pixel_bbox,
                        image_width=image_width,
                        image_height=image_height,
                    )
                except ValueError:
                    continue
                f.write(
                    json.dumps(
                        _grounding_example(
                            image_rel=image_rel,
                            label=label,
                            bbox_qwen=qwen_bbox,
                        )
                    )
                    + "\n"
                )
                stats.vlm_positive_rows += 1
                positive_labels.append(label)

            scene_name = record.get("scene_name")
            if not scene_name or not positive_labels:
                continue
            scene_labels = _scene_labels_for(str(scene_name))
            present = set(positive_labels)
            candidates = [lbl for lbl in scene_labels if lbl not in present]
            if not candidates:
                continue
            num_negatives = len(positive_labels) * negative_ratio
            rng.shuffle(candidates)
            for label in candidates[:num_negatives]:
                f.write(
                    json.dumps(_negative_example(image_rel=image_rel, label=label))
                    + "\n"
                )
                stats.vlm_negative_rows += 1


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_export(options: ExportOptions) -> ExportStats:
    options.output_root.mkdir(parents=True, exist_ok=True)
    stats = ExportStats()

    if options.include_clip:
        clip_path = options.output_root / "clip_descriptions.csv"
        export_clip_csv(
            descriptions_root=options.descriptions_root,
            output_path=clip_path,
            stats=stats,
        )
        logger.info("Wrote CLIP dataset: %s (%d rows)", clip_path, stats.clip_rows)

    if options.include_vlm:
        vlm_path = options.output_root / "vlm_grounding.jsonl"
        export_vlm_grounding_jsonl(
            perception_root=options.perception_root,
            descriptions_root=options.descriptions_root,
            output_path=vlm_path,
            scene_metadata_dir=options.scene_metadata_dir,
            negative_ratio=options.vlm_negative_ratio,
            stats=stats,
            seed=options.seed,
        )
        logger.info(
            "Wrote VLM dataset: %s (%d positives, %d negatives)",
            vlm_path, stats.vlm_positive_rows, stats.vlm_negative_rows,
        )

    stats_path = options.output_root / "dataset_export_stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2))
    return stats
