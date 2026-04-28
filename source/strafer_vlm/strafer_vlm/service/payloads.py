"""Pydantic request / response models for the grounding service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class GroundRequest(BaseModel):
    """JSON body for ``POST /ground``."""

    request_id: str = Field(..., description="Caller-assigned unique request identifier.")
    prompt: str = Field(..., description="Natural-language object description, e.g. 'Locate: red chair'.")
    image_jpeg_b64: str = Field(..., description="JPEG image encoded as a base64 string.")
    image_stamp_sec: float = Field(0.0, description="Capture timestamp in seconds (for logging).")
    max_image_side: int = Field(1024, description="If > 0, resize so the longest side <= this value before inference.")
    return_debug_overlay: bool = Field(False, description="If true, save a bbox overlay image and return its path.")


class GroundResponse(BaseModel):
    """JSON response from ``POST /ground``."""

    request_id: str = Field(..., description="Echo of the caller's request_id.")
    found: bool = Field(..., description="Whether the target object was detected.")
    bbox_2d: list[int] | None = Field(None, description="[x1, y1, x2, y2] in normalized [0, 1000] coordinates.")
    label: str | None = Field(None, description="Model-assigned label for the detected object.")
    confidence: float | None = Field(None, description="Detection confidence in [0, 1].")
    raw_output: str | None = Field(None, description="Raw model text output (for debugging).")
    latency_s: float = Field(0.0, description="Inference wall-clock time in seconds.")
    debug_overlay_jpeg_b64: str | None = Field(None, description="JPEG-encoded base64 image with bbox overlay, returned when return_debug_overlay=true and a bbox was found.")


class HealthResponse(BaseModel):
    """JSON response from ``GET /health``."""

    status: str = Field(..., description="'ok' when the model is loaded and ready, else 'loading'.")
    model_loaded: bool = Field(..., description="Whether the model has been loaded into GPU memory.")
    model_name: str | None = Field(None, description="HuggingFace model name or local path.")


class DescribeRequest(BaseModel):
    """JSON body for ``POST /describe``."""

    request_id: str = Field(..., description="Caller-assigned unique request identifier.")
    image_jpeg_b64: str = Field(..., description="JPEG image encoded as a base64 string.")
    prompt: str = Field(
        "Describe the objects and layout visible in this image in one or two sentences.",
        description="Description prompt to send to the VLM.",
    )
    max_image_side: int = Field(1024, description="If > 0, resize so the longest side <= this value before inference.")


class DescribeResponse(BaseModel):
    """JSON response from ``POST /describe``."""

    request_id: str = Field(..., description="Echo of the caller's request_id.")
    description: str = Field(..., description="Free-text scene description from the VLM.")
    latency_s: float = Field(0.0, description="Inference wall-clock time in seconds.")


class DetectObjectsRequest(BaseModel):
    """JSON body for ``POST /detect_objects``."""

    request_id: str = Field(..., description="Caller-assigned unique request identifier.")
    image_jpeg_b64: str = Field(..., description="JPEG image encoded as a base64 string.")
    max_image_side: int = Field(1024, description="If > 0, resize so the longest side <= this value before inference.")
    max_objects: int = Field(20, description="Maximum number of objects to return.")
    min_confidence: float = Field(0.3, description="Minimum confidence (0..1) for a detection to be included.")


class DetectedObject(BaseModel):
    """One detected object in a ``DetectObjectsResponse``."""

    label: str = Field(..., description="Model-assigned label for the detected object.")
    bbox_2d: list[int] = Field(..., description="[x1, y1, x2, y2] in pixel coordinates of the (possibly resized) inference image.")
    confidence: float = Field(..., description="Detection confidence in [0, 1].")


class DetectObjectsResponse(BaseModel):
    """JSON response from ``POST /detect_objects``."""

    request_id: str = Field(..., description="Echo of the caller's request_id.")
    objects: list[DetectedObject] = Field(default_factory=list, description="Detected objects sorted by confidence descending.")
    raw_output: str | None = Field(None, description="Raw model text output (for debugging).")
    latency_s: float = Field(0.0, description="Inference wall-clock time in seconds.")
