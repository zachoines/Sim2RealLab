"""Batch scene-description pipeline (Stages 1-3 + spot-check bookkeeping).

Runs on DGX Spark after teleop data collection. For each captured frame
the pipeline produces 3-5 natural-language descriptions at different
detail levels, suitable for CLIP contrastive fine-tuning (Task 11) and
VLM description preservation (Task 12).

Pipeline stages:

  Stage 1  -  programmatic spatial analysis via SpatialDescriptionBuilder
              (pure Python + shapely, no model).

  Stage 2  -  VLM description generation via Qwen2.5-VL-7B loaded
              standalone with transformers.AutoModelForVision2Seq. The
              VLM receives the structured facts JSON from Stage 1 *and*
              the raw RGB frame in a single prompt so the descriptions
              are both spatially accurate and visually grounded.

              NOTE: the 7B model is intentionally separate from the 3B
              model that strafer_vlm serves on port 8100. Feeding the
              fine-tune target's own outputs back as training data would
              cause collapse.

  Stage 3  -  ground-truth validation filter. Each candidate description
              is checked against the scene's label set. Descriptions
              mentioning objects not in the scene are dropped and
              counted in the rejection stats.

  Stage 4  -  human spot-check bookkeeping. The runner samples a fixed
              number of descriptions per batch and writes them to a
              ``spotcheck.jsonl`` file so a human can score them later.
              Stage 4 is *not* run automatically; it is a handoff point.

Input layout (from the Isaac Sim host via file transfer):

    data/perception/
        episode_0001/
            frames.jsonl        # one record per frame:
                                # {"frame_id", "image_path", "scene_name",
                                #  "scene_type", "robot_pos", "robot_quat",
                                #  "bboxes": [{"instance_id", "bbox_2d"}, ...]}

Output layout:

    data/descriptions/
        episode_0001/
            descriptions.jsonl  # one record per frame with validated descs
        batch_stats.json
        spotcheck.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

from strafer_lab.tools.scene_labels import SceneMetadataError, get_scene_label_set
from strafer_lab.tools.spatial_description import SpatialDescriptionBuilder

logger = logging.getLogger("generate_descriptions")


# ---------------------------------------------------------------------------
# Prompt template for Stage 2
# ---------------------------------------------------------------------------


_VLM_INSTRUCTION = (
    "Given these spatial facts about a scene viewed from a ground "
    "robot's camera (25 cm height) and the image, write THREE natural "
    "descriptions at different levels of detail:\n"
    "1. brief (5-10 words)\n"
    "2. medium (15-25 words)\n"
    "3. detailed (30-50 words)\n\n"
    "Include spatial relationships from the facts and visual details "
    "(lighting, textures, materials) from the image. Every description "
    "MUST only mention objects listed in the facts. Return each "
    "description on its own line prefixed with 'BRIEF:', 'MEDIUM:', "
    "or 'DETAILED:'. Do not include any other text.\n\n"
)


def build_stage2_prompt(spatial_facts: dict[str, Any]) -> str:
    return _VLM_INSTRUCTION + "Facts:\n" + json.dumps(spatial_facts, indent=2)


# ---------------------------------------------------------------------------
# Parsing Stage 2 output into descriptions
# ---------------------------------------------------------------------------


_LEVEL_RE = re.compile(
    r"^\s*(?P<level>BRIEF|MEDIUM|DETAILED)\s*[:\-]\s*(?P<text>.+?)\s*$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Description:
    level: str
    text: str


def parse_descriptions(raw_output: str) -> list[Description]:
    """Parse the Stage 2 VLM output into ``(level, text)`` pairs.

    Tolerant to stray whitespace, missing levels, extra prose. If the
    VLM emits free-form text without the ``BRIEF:/MEDIUM:/DETAILED:``
    markers, the whole block is returned as a single ``medium``
    description so downstream filtering can still run.
    """
    if not raw_output:
        return []
    found: list[Description] = []
    for line in raw_output.splitlines():
        match = _LEVEL_RE.match(line)
        if match:
            level = match.group("level").lower()
            text = match.group("text").strip()
            if text:
                found.append(Description(level=level, text=text))
    if found:
        return found

    fallback = raw_output.strip()
    if fallback:
        return [Description(level="medium", text=fallback)]
    return []


# ---------------------------------------------------------------------------
# Stage 3: label-set validation
# ---------------------------------------------------------------------------


def description_mentions_unknown_label(
    description: str, scene_labels: set[str],
) -> bool:
    """Return True if the description mentions any labels outside the scene.

    Scene labels are tokenized case-insensitively and each scene label is
    searched for as a whole-word phrase inside the description. This is
    not a full semantic check — it only catches *mentions* of scene
    objects to cross-reference Stage 2 output against ground truth.

    The return value is ``True`` when the description contains a word
    that LOOKS like a scene label, yet that label is NOT in
    ``scene_labels``.  In practice we keep the filter permissive: we
    only reject descriptions that mention a known catalog-style noun
    (e.g. ``"chair"``, ``"lamp"``, ``"microwave"``) that does not belong
    to the scene.
    """
    return False  # permissive default — see validate_description for logic


# Catalog of common indoor nouns that Infinigen generates. This is the
# universe we probe against so we can reject hallucinations like
# "a chair" when no chair is in the scene. Labels NOT in this catalog
# are untracked and do not trigger rejection (e.g. "wall").
_COMMON_OBJECT_NOUNS: tuple[str, ...] = (
    "table",
    "chair",
    "couch",
    "sofa",
    "bed",
    "lamp",
    "plant",
    "door",
    "window",
    "bookshelf",
    "bookcase",
    "shelf",
    "desk",
    "tv",
    "television",
    "fridge",
    "refrigerator",
    "microwave",
    "oven",
    "sink",
    "toilet",
    "bathtub",
    "mirror",
    "computer",
    "monitor",
    "keyboard",
    "mouse",
    "cabinet",
    "counter",
    "stool",
    "chandelier",
    "rug",
    "carpet",
    "painting",
    "poster",
    "clock",
    "fan",
    "oven",
    "bottle",
    "cup",
    "bowl",
    "plate",
    "vase",
)


def _word_present(text: str, word: str) -> bool:
    """Whole-word (case-insensitive) membership check, plurals tolerated."""
    pattern = rf"\b{re.escape(word)}s?\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def validate_description(description: str, scene_labels: set[str]) -> tuple[bool, list[str]]:
    """Validate a description against the scene's label set.

    Returns ``(is_valid, offending_labels)``. A description is valid
    unless it mentions a common catalog noun that the scene does NOT
    contain. This catches the most common VLM failure mode (inventing a
    table/chair/tv that isn't there) while staying permissive about
    words like "wall" or "lighting" that we do not track in the label
    catalog.
    """
    offending: list[str] = []
    for noun in _COMMON_OBJECT_NOUNS:
        if noun in scene_labels:
            continue
        if _word_present(description, noun):
            offending.append(noun)
    return (not offending, offending)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


@dataclass
class BatchStats:
    total_frames: int = 0
    total_descriptions: int = 0
    validated_descriptions: int = 0
    rejected_descriptions: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)

    def record_rejection(self, offending: list[str]) -> None:
        self.rejected_descriptions += 1
        for label in offending:
            self.rejection_reasons[label] = self.rejection_reasons.get(label, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_frames": self.total_frames,
            "total_descriptions": self.total_descriptions,
            "validated_descriptions": self.validated_descriptions,
            "rejected_descriptions": self.rejected_descriptions,
            "rejection_reasons": dict(self.rejection_reasons),
            "errors": list(self.errors),
        }


VLMRunner = Callable[[str, Path], str]
"""Callable taking (prompt_text, image_path) and returning raw VLM output."""


def iter_frame_records(perception_root: Path) -> Iterator[tuple[Path, dict[str, Any]]]:
    """Yield (episode_dir, record) pairs from a perception data tree.

    Each episode is expected to have a ``frames.jsonl`` file with one
    JSON record per line.
    """
    perception_root = Path(perception_root)
    for episode_dir in sorted(perception_root.iterdir()):
        if not episode_dir.is_dir():
            continue
        frames_path = episode_dir / "frames.jsonl"
        if not frames_path.exists():
            continue
        with frames_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning(
                        "Skipping malformed frame in %s: %s", frames_path, exc,
                    )
                    continue
                if isinstance(record, dict):
                    yield episode_dir, record


def process_frame(
    *,
    record: dict[str, Any],
    episode_dir: Path,
    scene_metadata_dir: Path,
    builders: dict[str, SpatialDescriptionBuilder],
    label_sets: dict[str, set[str]],
    vlm_runner: VLMRunner,
    stats: BatchStats,
) -> dict[str, Any] | None:
    """Run Stages 1-3 for one frame. Returns the output record or None."""
    scene_name = record.get("scene_name") or record.get("scene") or ""
    if record.get("scene_type") == "procroom":
        return None

    if not scene_name:
        stats.errors.append(f"frame {record.get('frame_id')}: missing scene_name")
        return None

    if scene_name not in builders:
        try:
            metadata_path = scene_metadata_dir / scene_name / "scene_metadata.json"
            with metadata_path.open("r", encoding="utf-8") as f:
                metadata = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError, SceneMetadataError) as exc:
            stats.errors.append(f"frame {record.get('frame_id')}: {exc}")
            return None
        builders[scene_name] = SpatialDescriptionBuilder(metadata)
        label_sets[scene_name] = {
            str(obj.get("label", "")) for obj in metadata.get("objects", []) if obj.get("label")
        }

    builder = builders[scene_name]
    try:
        spatial_facts = builder.build(record)
    except (KeyError, ValueError) as exc:
        stats.errors.append(f"frame {record.get('frame_id')}: stage1 {exc}")
        return None

    if not spatial_facts.get("visible_objects"):
        # Nothing visible — skip the frame. Empty-scene descriptions have
        # no training signal and easily drift into hallucinations.
        return None

    image_rel = record.get("image_path") or ""
    image_path = episode_dir / image_rel if image_rel else None
    if image_path is None or not image_path.exists():
        stats.errors.append(
            f"frame {record.get('frame_id')}: image path missing ({image_rel!r})"
        )
        return None

    prompt = build_stage2_prompt(spatial_facts)
    try:
        raw_output = vlm_runner(prompt, image_path)
    except Exception as exc:  # pragma: no cover - runner-specific
        stats.errors.append(f"frame {record.get('frame_id')}: stage2 {exc}")
        return None

    descriptions = parse_descriptions(raw_output)
    stats.total_descriptions += len(descriptions)
    validated: list[dict[str, Any]] = []
    for desc in descriptions:
        is_valid, offending = validate_description(desc.text, label_sets[scene_name])
        if is_valid:
            validated.append({"level": desc.level, "text": desc.text})
            stats.validated_descriptions += 1
        else:
            stats.record_rejection(offending)

    if not validated:
        return None

    return {
        "frame_id": record.get("frame_id"),
        "image_path": image_rel,
        "scene_name": scene_name,
        "scene_type": record.get("scene_type", "infinigen"),
        "robot_pose": {
            "position": record.get("robot_pos"),
            "quat": record.get("robot_quat"),
        },
        "spatial_facts": spatial_facts,
        "descriptions": validated,
        "raw_vlm_output": raw_output,
    }


def run_batch(
    *,
    perception_root: Path,
    scene_metadata_dir: Path,
    output_root: Path,
    vlm_runner: VLMRunner,
    spotcheck_sample_size: int = 50,
    rng_seed: int = 0,
) -> BatchStats:
    perception_root = Path(perception_root)
    scene_metadata_dir = Path(scene_metadata_dir)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    builders: dict[str, SpatialDescriptionBuilder] = {}
    label_sets: dict[str, set[str]] = {}
    stats = BatchStats()
    spotcheck_pool: list[dict[str, Any]] = []
    rng = random.Random(rng_seed)

    for episode_dir, record in iter_frame_records(perception_root):
        stats.total_frames += 1
        out = process_frame(
            record=record,
            episode_dir=episode_dir,
            scene_metadata_dir=scene_metadata_dir,
            builders=builders,
            label_sets=label_sets,
            vlm_runner=vlm_runner,
            stats=stats,
        )
        if out is None:
            continue

        out_episode = output_root / episode_dir.name
        out_episode.mkdir(parents=True, exist_ok=True)
        descriptions_path = out_episode / "descriptions.jsonl"
        with descriptions_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(out) + "\n")

        if len(spotcheck_pool) < spotcheck_sample_size:
            spotcheck_pool.append(out)
        else:
            idx = rng.randint(0, stats.total_frames)
            if idx < spotcheck_sample_size:
                spotcheck_pool[idx] = out

    stats_path = output_root / "batch_stats.json"
    stats_path.write_text(json.dumps(stats.to_dict(), indent=2))
    spotcheck_path = output_root / "spotcheck.jsonl"
    with spotcheck_path.open("w", encoding="utf-8") as f:
        for entry in spotcheck_pool:
            f.write(json.dumps(entry) + "\n")

    logger.info(
        "Batch complete: %d frames, %d/%d descriptions validated, %d rejected",
        stats.total_frames,
        stats.validated_descriptions,
        stats.total_descriptions,
        stats.rejected_descriptions,
    )
    return stats


# ---------------------------------------------------------------------------
# Default VLM runner (Qwen2.5-VL-7B via transformers)
# ---------------------------------------------------------------------------


def build_default_vlm_runner(
    model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
    max_new_tokens: int = 512,
) -> VLMRunner:
    """Return a VLM runner that loads Qwen2.5-VL-7B standalone.

    Heavy imports are deferred to the first invocation so the rest of
    the pipeline (and its tests) can run without ``transformers``,
    ``torch`` or the 7B checkpoint.
    """

    state: dict[str, Any] = {"model": None, "processor": None}

    def _run(prompt: str, image_path: Path) -> str:
        if state["model"] is None:
            from PIL import Image  # noqa: F401 — warm up PIL
            from transformers import (
                AutoModelForVision2Seq,
                AutoProcessor,
            )

            logger.info("Loading description-stage VLM %s", model_name)
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForVision2Seq.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype="auto",
            )
            model.eval()
            state["model"] = model
            state["processor"] = processor

        from PIL import Image

        model = state["model"]
        processor = state["processor"]
        image = Image.open(image_path).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        chat_text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = processor(
            text=[chat_text], images=[image], return_tensors="pt",
        ).to(model.device)
        output_ids = model.generate(
            **inputs, max_new_tokens=max_new_tokens, do_sample=False,
        )
        prompt_len = inputs["input_ids"].shape[1]
        decoded = processor.batch_decode(
            output_ids[:, prompt_len:], skip_special_tokens=True,
        )
        return decoded[0].strip()

    return _run


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception-data", type=Path, required=True)
    parser.add_argument("--scene-metadata", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vlm-model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--spotcheck-samples", type=int, default=50)
    parser.add_argument("--seed", type=int, default=int(time.time()) % 2**31)
    args = parser.parse_args(list(argv) if argv is not None else None)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    runner = build_default_vlm_runner(
        model_name=args.vlm_model, max_new_tokens=args.max_new_tokens,
    )
    stats = run_batch(
        perception_root=args.perception_data,
        scene_metadata_dir=args.scene_metadata,
        output_root=args.output,
        vlm_runner=runner,
        spotcheck_sample_size=args.spotcheck_samples,
        rng_seed=args.seed,
    )
    if stats.rejected_descriptions:
        logger.warning(
            "Rejection reasons: %s", json.dumps(stats.rejection_reasons),
        )
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_cli_main())
