"""Prepare the comprehensive VLM LoRA fine-tuning JSONL dataset.

Unlike the minimal exporter in :mod:`strafer_lab.tools.dataset_export`
this script is the production-quality data preparation stage for
Task 12. It emits three categories of examples that the LoRA run in
``finetune_vlm.py`` (a future script) consumes:

1. **Single-object grounding** (primary). One ``<ref>label</ref><box>...``
   answer per frame+label. Coordinates rescaled to Qwen's 0..1000 range.

2. **Negative examples** (1:3 ratio). The same frame is reused with a
   label drawn from the scene catalog that is NOT visible in the frame;
   the assistant answer is ``"The object is not visible in this image."``.
   Without negatives the model learns to always emit a bbox.

3. **Multi-object detection** (~20% of examples). Frames with 2-10
   visible objects produce a single ``list all objects`` example whose
   answer concatenates all ``<ref>/<box>`` pairs. This primes the VLM
   for the ``POST /detect_objects`` endpoint.

4. **Description preservation** (~10% of examples). Reuses Stage 2
   descriptions from Task 9 as plain image-to-text pairs so LoRA
   fine-tuning does not degrade the ``POST /describe`` capability via
   catastrophic forgetting.

All categories **exclude frames from ProcRoom** (``scene_type ==
"procroom"``) per Section 5.5.3 of the design doc.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

from strafer_lab.tools.dataset_export import (
    format_qwen_grounding_answer,
    iter_description_records,
    iter_perception_frames,
    pixel_bbox_to_qwen,
)

logger = logging.getLogger("prepare_vlm_finetune_data")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class VLMDataPrepConfig:
    perception_root: Path
    descriptions_root: Path
    scene_metadata_dir: Path
    output_path: Path
    negative_ratio: int = 3
    multi_object_fraction: float = 0.20
    description_fraction: float = 0.10
    min_multi_object_visible: int = 2
    max_multi_object_visible: int = 10
    seed: int = 0


@dataclass
class VLMDataPrepStats:
    frames_seen: int = 0
    procroom_skipped: int = 0
    single_object_examples: int = 0
    negative_examples: int = 0
    multi_object_examples: int = 0
    description_examples: int = 0
    skipped_missing_image_dims: int = 0

    def to_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


# ---------------------------------------------------------------------------
# Example builders
# ---------------------------------------------------------------------------


def _single_object_example(
    image_rel: str, label: str, bbox_qwen: tuple[int, int, int, int],
) -> dict[str, Any]:
    return {
        "image": image_rel,
        "conversations": [
            {
                "role": "user",
                "content": f"<image>Locate the {label} in this image.",
            },
            {
                "role": "assistant",
                "content": format_qwen_grounding_answer(label, bbox_qwen),
            },
        ],
    }


def _negative_example(image_rel: str, label: str) -> dict[str, Any]:
    return {
        "image": image_rel,
        "conversations": [
            {
                "role": "user",
                "content": f"<image>Locate the {label} in this image.",
            },
            {
                "role": "assistant",
                "content": "The object is not visible in this image.",
            },
        ],
    }


def _multi_object_example(
    image_rel: str,
    detections: list[tuple[str, tuple[int, int, int, int]]],
) -> dict[str, Any]:
    answer = "".join(
        format_qwen_grounding_answer(label, bbox) for label, bbox in detections
    )
    return {
        "image": image_rel,
        "conversations": [
            {
                "role": "user",
                "content": "<image>List all visible objects with their bounding boxes.",
            },
            {"role": "assistant", "content": answer},
        ],
    }


def _description_example(image_rel: str, description: str) -> dict[str, Any]:
    return {
        "image": image_rel,
        "conversations": [
            {
                "role": "user",
                "content": "<image>Describe the scene in one or two sentences.",
            },
            {"role": "assistant", "content": description},
        ],
    }


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------


def _load_scene_labels(
    scene_metadata_dir: Path, scene_name: str, cache: dict[str, set[str]],
) -> set[str]:
    if scene_name in cache:
        return cache[scene_name]
    path = Path(scene_metadata_dir) / scene_name / "scene_metadata.json"
    if not path.exists():
        cache[scene_name] = set()
        return cache[scene_name]
    try:
        with path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        cache[scene_name] = set()
        return cache[scene_name]
    labels = {
        str(obj.get("label"))
        for obj in metadata.get("objects", []) or []
        if obj.get("label")
    }
    cache[scene_name] = labels
    return labels


def _load_description_lookup(
    descriptions_root: Path,
) -> dict[str, list[dict[str, Any]]]:
    """Map ``image_rel`` → list of validated descriptions."""
    lookup: dict[str, list[dict[str, Any]]] = {}
    for episode_dir, record in iter_description_records(descriptions_root):
        if record.get("scene_type") == "procroom":
            continue
        image_rel_raw = record.get("image_path") or ""
        if not image_rel_raw:
            continue
        key = f"{episode_dir.name}/{image_rel_raw}"
        descs = [
            desc for desc in record.get("descriptions", []) or []
            if isinstance(desc, dict) and desc.get("text")
        ]
        if descs:
            lookup[key] = descs
    return lookup


def _extract_valid_pixel_bboxes(
    frame_record: dict[str, Any],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    out: list[tuple[str, tuple[int, int, int, int]]] = []
    for entry in frame_record.get("bboxes") or []:
        if not isinstance(entry, dict):
            continue
        label = entry.get("label")
        bbox = entry.get("bbox_2d")
        if not isinstance(label, str) or not label or not bbox or len(bbox) != 4:
            continue
        try:
            out.append((label, (int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3]))))
        except (TypeError, ValueError):
            continue
    return out


def generate_examples(config: VLMDataPrepConfig) -> tuple[list[dict[str, Any]], VLMDataPrepStats]:
    """Build the complete example list without writing to disk.

    Split out from :func:`write_examples` so unit tests can assert on
    the example structure without dealing with file I/O.
    """
    stats = VLMDataPrepStats()
    rng = random.Random(config.seed)
    scene_labels_cache: dict[str, set[str]] = {}
    description_lookup = _load_description_lookup(config.descriptions_root)

    # Collect examples per category, then interleave so the output file
    # has a deterministic ordering and respects the target mix ratios.
    single_object: list[dict[str, Any]] = []
    negatives: list[dict[str, Any]] = []
    multi_object: list[dict[str, Any]] = []

    for episode_dir, record in iter_perception_frames(config.perception_root):
        stats.frames_seen += 1
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
            stats.skipped_missing_image_dims += 1
            continue

        pixel_detections = _extract_valid_pixel_bboxes(record)
        if not pixel_detections:
            continue

        qwen_detections: list[tuple[str, tuple[int, int, int, int]]] = []
        for label, pbbox in pixel_detections:
            try:
                qbbox = pixel_bbox_to_qwen(
                    pbbox, image_width=image_width, image_height=image_height,
                )
            except ValueError:
                continue
            qwen_detections.append((label, qbbox))
            single_object.append(_single_object_example(image_rel, label, qbbox))
            stats.single_object_examples += 1

        present_labels = {label for label, _ in qwen_detections}
        scene_name = str(record.get("scene_name", ""))
        if scene_name:
            scene_labels = _load_scene_labels(
                config.scene_metadata_dir, scene_name, scene_labels_cache,
            )
            candidates = [lbl for lbl in scene_labels if lbl not in present_labels]
            rng.shuffle(candidates)
            n_negatives = min(
                len(candidates),
                len(qwen_detections) * config.negative_ratio,
            )
            for label in candidates[:n_negatives]:
                negatives.append(_negative_example(image_rel, label))
                stats.negative_examples += 1

        if (
            config.min_multi_object_visible
            <= len(qwen_detections)
            <= config.max_multi_object_visible
        ):
            multi_object.append(_multi_object_example(image_rel, qwen_detections))
            stats.multi_object_examples += 1

    # Description preservation: reuse Stage 2 descriptions from Task 9.
    description_examples: list[dict[str, Any]] = []
    for image_rel, descs in description_lookup.items():
        for desc in descs:
            description_examples.append(_description_example(image_rel, str(desc["text"])))

    # Target mix: single_object is the baseline; multi-object should be
    # ~multi_object_fraction and description ~description_fraction of the
    # total output.
    examples: list[dict[str, Any]] = list(single_object) + list(negatives)
    total_base = len(examples)

    target_multi = int(total_base * config.multi_object_fraction / max(1 - config.multi_object_fraction, 1e-6))
    if multi_object:
        rng.shuffle(multi_object)
        examples.extend(multi_object[:target_multi])

    target_desc = int(total_base * config.description_fraction / max(1 - config.description_fraction, 1e-6))
    if description_examples:
        rng.shuffle(description_examples)
        chosen_desc = description_examples[:target_desc]
        examples.extend(chosen_desc)
        stats.description_examples = len(chosen_desc)

    rng.shuffle(examples)
    return examples, stats


def write_examples(config: VLMDataPrepConfig) -> VLMDataPrepStats:
    examples, stats = generate_examples(config)
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with config.output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example) + "\n")
    stats_path = config.output_path.with_suffix(".stats.json")
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2))
    logger.info(
        "Wrote %d examples to %s "
        "(single=%d, negative=%d, multi=%d, describe=%d)",
        len(examples),
        config.output_path,
        stats.single_object_examples,
        stats.negative_examples,
        stats.multi_object_examples,
        stats.description_examples,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Iterable[str] | None = None) -> VLMDataPrepConfig:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-data", type=Path, required=True, dest="perception_root")
    parser.add_argument("--descriptions", type=Path, required=True, dest="descriptions_root")
    parser.add_argument("--scene-metadata", type=Path, required=True, dest="scene_metadata_dir")
    parser.add_argument("--output", type=Path, required=True, dest="output_path")
    parser.add_argument("--negative-ratio", type=int, default=3)
    parser.add_argument("--multi-object-fraction", type=float, default=0.20)
    parser.add_argument("--description-fraction", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(list(argv) if argv is not None else None)
    return VLMDataPrepConfig(
        perception_root=args.perception_root,
        descriptions_root=args.descriptions_root,
        scene_metadata_dir=args.scene_metadata_dir,
        output_path=args.output_path,
        negative_ratio=args.negative_ratio,
        multi_object_fraction=args.multi_object_fraction,
        description_fraction=args.description_fraction,
        seed=args.seed,
    )


def main(argv: Iterable[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    config = parse_args(argv)
    write_examples(config)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
