"""strafer_vlm — VLM grounding for Strafer robot navigation."""

__version__ = "0.1.0"

from strafer_vlm.inference.parsing import (
    SYSTEM_PROMPT_DEFAULT,
    GroundingExample,
    GroundingTarget,
    normalize_prompt,
    parse_grounding_prediction,
    parse_grounding_target,
    serialize_target,
)
from strafer_vlm.training.dataset_io import load_grounding_dataset

__all__ = [
    "SYSTEM_PROMPT_DEFAULT",
    "GroundingExample",
    "GroundingTarget",
    "load_grounding_dataset",
    "normalize_prompt",
    "parse_grounding_prediction",
    "parse_grounding_target",
    "serialize_target",
]
