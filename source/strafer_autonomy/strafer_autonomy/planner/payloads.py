"""Pydantic request / response models for the planner service."""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanRequest(BaseModel):
    """JSON body for ``POST /plan``."""

    request_id: str = Field(..., description="Caller-assigned unique request identifier.")
    raw_command: str = Field(..., description="Natural-language user command, e.g. 'go to the door'.")
    robot_state: dict | None = Field(None, description="Optional robot state snapshot.")
    active_mission_summary: dict | None = Field(None, description="Optional summary of the currently active mission.")
    available_skills: list[str] = Field(default_factory=list, description="Skill names the executor supports.")


class SkillCallPayload(BaseModel):
    """One step inside a mission plan response."""

    step_id: str = Field(..., description="Unique step identifier within the plan.")
    skill: str = Field(..., description="Skill name to execute.")
    args: dict = Field(default_factory=dict, description="Skill-specific arguments.")
    timeout_s: float | None = Field(None, description="Maximum execution time in seconds.")
    retry_limit: int = Field(0, description="Number of retries on failure.")


class PlanResponse(BaseModel):
    """JSON response from ``POST /plan``."""

    mission_id: str = Field(..., description="Unique mission identifier.")
    mission_type: str = Field(..., description="Classified intent type, e.g. 'go_to_target'.")
    raw_command: str = Field(..., description="Echo of the original user command.")
    steps: list[SkillCallPayload] = Field(..., description="Ordered skill steps to execute.")
    created_at: float = Field(..., description="Plan creation timestamp (epoch seconds).")


class PlannerHealthResponse(BaseModel):
    """JSON response from ``GET /health``."""

    status: str = Field(..., description="'ok' when the model is loaded and ready, else 'loading'.")
    model_loaded: bool = Field(..., description="Whether the LLM has been loaded into memory.")
    model_name: str | None = Field(None, description="HuggingFace model name or local path.")


class GroundResultPayload(BaseModel):
    """Subset of VLM ``GroundResponse`` fields forwarded to the planner caller."""

    found: bool = Field(..., description="Whether the target object was detected.")
    bbox_2d: list[int] | None = Field(None, description="[x1, y1, x2, y2] in normalized [0, 1000] coordinates.")
    label: str | None = Field(None, description="Model-assigned label for the detected object.")
    confidence: float | None = Field(None, description="Detection confidence in [0, 1].")
    latency_s: float = Field(0.0, description="VLM inference wall-clock time in seconds.")


class PlanWithGroundingRequest(PlanRequest):
    """JSON body for ``POST /plan_with_grounding``.

    Extends ``PlanRequest`` with an optional camera frame the planner can
    forward to the co-located VLM service for pre-grounding.
    """

    image_jpeg_b64: str | None = Field(
        None,
        description="Optional JPEG image (base64). If present and the parsed "
        "intent requires grounding, the planner forwards it to the VLM "
        "service on localhost.",
    )
    image_stamp_sec: float = Field(
        0.0,
        description="Capture timestamp in seconds (forwarded to VLM for logging).",
    )
    max_image_side: int = Field(
        1024,
        description="Forwarded to the VLM service — resize bound for inference.",
    )


class PlanWithGroundingResponse(PlanResponse):
    """JSON response from ``POST /plan_with_grounding``.

    Extends ``PlanResponse`` with an optional pre-grounding result produced
    by the co-located VLM service when a camera frame was provided.
    """

    pre_grounding: GroundResultPayload | None = Field(
        None,
        description="Pre-grounding result from the VLM service, or null if "
        "no image was provided, the intent did not require grounding, or "
        "the VLM call failed.",
    )
