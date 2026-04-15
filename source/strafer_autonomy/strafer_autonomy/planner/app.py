"""FastAPI planner service wrapping LLM-based mission planning.

Launch:
    uvicorn strafer_autonomy.planner.app:create_app --factory --host 0.0.0.0 --port 8200

Environment variables (read by the app at startup):
    PLANNER_MODEL          HF model name or local path (default: Qwen/Qwen3-4B)
    PLANNER_DEVICE_MAP     device_map for model loading (default: auto)
    PLANNER_TORCH_DTYPE    torch dtype (default: auto)
    PLANNER_LOAD_4BIT      set "1" to enable 4-bit quantisation
    PLANNER_MAX_TOKENS     max new tokens per inference (default: 256)

Host and port are controlled by uvicorn CLI arguments, not by the app.
"""

from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from strafer_autonomy.planner.intent_parser import IntentParseError, parse_intent
from strafer_autonomy.planner.llm_runtime import LLMRuntime
from strafer_autonomy.planner.payloads import (
    GroundResultPayload,
    PlannerHealthResponse,
    PlanRequest,
    PlanResponse,
    PlanWithGroundingRequest,
    PlanWithGroundingResponse,
    SkillCallPayload,
)
from strafer_autonomy.planner.plan_compiler import CompilationError, compile_plan
from strafer_autonomy.planner.prompt_builder import build_messages
from strafer_autonomy.schemas import MissionPlan, PlannerRequest, MissionIntent

logger = logging.getLogger("strafer_autonomy.planner")


# ---------------------------------------------------------------------------
# Runtime state — populated during lifespan
# ---------------------------------------------------------------------------

class _RuntimeState:
    """Mutable singleton holding the loaded LLM runtime."""

    def __init__(self) -> None:
        self.llm: LLMRuntime = LLMRuntime()
        self.inference_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)


_state = _RuntimeState()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _steps_payload(mission_plan: MissionPlan) -> list[SkillCallPayload]:
    return [
        SkillCallPayload(
            step_id=step.step_id,
            skill=step.skill,
            args=step.args,
            timeout_s=step.timeout_s,
            retry_limit=step.retry_limit,
        )
        for step in mission_plan.steps
    ]


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def _lifespan(app: FastAPI):
    model_name = _env("PLANNER_MODEL", "Qwen/Qwen3-4B")
    device_map = _env("PLANNER_DEVICE_MAP", "auto")
    torch_dtype = _env("PLANNER_TORCH_DTYPE", "auto")
    load_4bit = _env("PLANNER_LOAD_4BIT", "0") == "1"
    max_tokens = int(_env("PLANNER_MAX_TOKENS", "256"))

    logger.info(
        "Loading planner model %s (device_map=%s, dtype=%s, 4bit=%s)",
        model_name, device_map, torch_dtype, load_4bit,
    )
    try:
        _state.llm.load(
            model_name=model_name,
            device_map=device_map,
            torch_dtype=torch_dtype,
            load_4bit=load_4bit,
            max_tokens=max_tokens,
        )
    except Exception:
        logger.exception("Failed to load planner model — service will return 503 on /plan requests.")

    yield

    _state.llm.ready = False
    _state.llm.model = None
    _state.llm.tokenizer = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

async def _run_plan_pipeline(
    *,
    request_id: str,
    raw_command: str,
    robot_state: dict | None,
    active_mission_summary: dict | None,
    available_skills: tuple[str, ...],
) -> tuple[MissionIntent, MissionPlan]:
    """Execute the LLM → intent → compile pipeline and return intent + plan.

    Raises ``HTTPException`` with appropriate status codes on failure so
    both ``/plan`` and ``/plan_with_grounding`` can share this code path.
    """
    if not raw_command or not raw_command.strip():
        raise HTTPException(status_code=400, detail="raw_command must be a non-empty string.")

    planner_request = PlannerRequest(
        request_id=request_id,
        raw_command=raw_command,
        robot_state=robot_state,
        active_mission_summary=active_mission_summary,
        available_skills=available_skills,
    )
    messages = build_messages(planner_request)

    loop = asyncio.get_running_loop()
    try:
        raw_output = await loop.run_in_executor(
            _state.inference_pool,
            lambda: _state.llm.generate(messages),
        )
    except Exception as exc:
        logger.exception("[%s] LLM generation failed", request_id)
        raise HTTPException(status_code=500, detail=f"LLM generation error: {exc}") from exc

    logger.info("[%s] LLM output: %s", request_id, raw_output)

    try:
        intent = parse_intent(raw_output, raw_command)
    except IntentParseError as exc:
        logger.warning("[%s] Intent parsing failed: %s", request_id, exc)
        raise HTTPException(status_code=422, detail=f"Failed to parse planner output: {exc}") from exc

    try:
        mission_plan = compile_plan(intent)
    except CompilationError as exc:
        logger.warning("[%s] Plan compilation failed: %s", request_id, exc)
        raise HTTPException(status_code=422, detail=f"Plan compilation error: {exc}") from exc

    return intent, mission_plan


def create_app() -> FastAPI:
    app = FastAPI(
        title="Strafer LLM Planner Service",
        description="LLM-based mission planning exposed as a REST API.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.get("/health", response_model=PlannerHealthResponse, summary="Service readiness check")
    async def health() -> PlannerHealthResponse:
        return PlannerHealthResponse(
            status="ok" if _state.llm.ready else "loading",
            model_loaded=_state.llm.ready,
            model_name=_state.llm.model_name or None,
        )

    @app.post("/plan", response_model=PlanResponse, summary="Generate a mission plan from a natural-language command")
    async def plan(req: PlanRequest) -> PlanResponse:
        if not _state.llm.ready:
            raise HTTPException(status_code=503, detail="Planner model is not loaded yet.")

        logger.info("[%s] /plan command=%r", req.request_id, req.raw_command)

        _intent, mission_plan = await _run_plan_pipeline(
            request_id=req.request_id,
            raw_command=req.raw_command,
            robot_state=req.robot_state,
            active_mission_summary=req.active_mission_summary,
            available_skills=tuple(req.available_skills),
        )

        return PlanResponse(
            mission_id=mission_plan.mission_id,
            mission_type=mission_plan.mission_type,
            raw_command=mission_plan.raw_command,
            steps=_steps_payload(mission_plan),
            created_at=mission_plan.created_at,
        )

    vlm_ground_url = _env("VLM_GROUND_URL", "http://localhost:8100/ground")
    vlm_timeout_s = float(_env("VLM_GROUND_TIMEOUT_S", "10.0"))

    @app.post(
        "/plan_with_grounding",
        response_model=PlanWithGroundingResponse,
        summary="Generate a plan and optionally pre-ground a target via the co-located VLM",
    )
    async def plan_with_grounding(req: PlanWithGroundingRequest) -> PlanWithGroundingResponse:
        if not _state.llm.ready:
            raise HTTPException(status_code=503, detail="Planner model is not loaded yet.")

        logger.info(
            "[%s] /plan_with_grounding command=%r image=%s",
            req.request_id, req.raw_command, bool(req.image_jpeg_b64),
        )

        intent, mission_plan = await _run_plan_pipeline(
            request_id=req.request_id,
            raw_command=req.raw_command,
            robot_state=req.robot_state,
            active_mission_summary=req.active_mission_summary,
            available_skills=tuple(req.available_skills),
        )

        pre_grounding: GroundResultPayload | None = None
        if req.image_jpeg_b64 and intent.requires_grounding and intent.target_label:
            try:
                import httpx

                async with httpx.AsyncClient(timeout=vlm_timeout_s) as http_client:
                    vlm_resp = await http_client.post(
                        vlm_ground_url,
                        json={
                            "request_id": req.request_id,
                            "prompt": f"Locate: {intent.target_label}",
                            "image_jpeg_b64": req.image_jpeg_b64,
                            "image_stamp_sec": req.image_stamp_sec,
                            "max_image_side": req.max_image_side,
                        },
                    )
                    vlm_resp.raise_for_status()
                    body = vlm_resp.json()
                    pre_grounding = GroundResultPayload(
                        found=bool(body.get("found", False)),
                        bbox_2d=body.get("bbox_2d"),
                        label=body.get("label"),
                        confidence=body.get("confidence"),
                        latency_s=float(body.get("latency_s", 0.0)),
                    )
                    logger.info(
                        "[%s] pre-grounding found=%s latency=%.3fs",
                        req.request_id, pre_grounding.found, pre_grounding.latency_s,
                    )
            except Exception:
                logger.warning(
                    "[%s] VLM pre-grounding failed; returning plan without pre_grounding",
                    req.request_id, exc_info=True,
                )

        return PlanWithGroundingResponse(
            mission_id=mission_plan.mission_id,
            mission_type=mission_plan.mission_type,
            raw_command=mission_plan.raw_command,
            steps=_steps_payload(mission_plan),
            created_at=mission_plan.created_at,
            pre_grounding=pre_grounding,
        )

    return app
