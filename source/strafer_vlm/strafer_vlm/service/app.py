"""FastAPI grounding service wrapping Qwen2.5-VL inference.

Launch:
    uvicorn strafer_vlm.service.app:create_app --factory --host 0.0.0.0 --port 8100

Environment variables:
    GROUNDING_MODEL          HF model name or local path (default: Qwen/Qwen2.5-VL-3B-Instruct)
    GROUNDING_DEVICE_MAP     device_map for model loading (default: auto)
    GROUNDING_TORCH_DTYPE    torch dtype (default: auto)
    GROUNDING_LOAD_4BIT      set "1" to enable 4-bit quantisation
    GROUNDING_MAX_TOKENS     max new tokens per inference (default: 128)
    GROUNDING_MAX_IMAGE_MP   max decoded image megapixels (default: 20)
    GROUNDING_INFERENCE_TIMEOUT  max seconds for a single inference call (default: 30, 0=no limit)
    GROUNDING_HOST           bind host (default: 0.0.0.0)
    GROUNDING_PORT           bind port (default: 8100)
"""

from __future__ import annotations

import asyncio
import base64
import io
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException

from strafer_vlm.service.payloads import (
    DescribeRequest,
    DescribeResponse,
    DetectedObject,
    DetectObjectsRequest,
    DetectObjectsResponse,
    GroundRequest,
    GroundResponse,
    HealthResponse,
)

logger = logging.getLogger("strafer_vlm.service")


# ---------------------------------------------------------------------------
# Runtime state — populated during lifespan
# ---------------------------------------------------------------------------

class _RuntimeState:
    """Mutable singleton holding the loaded model and processor."""

    def __init__(self) -> None:
        self.model: Any = None
        self.processor: Any = None
        self.model_name: str = ""
        self.ready: bool = False
        self.inference_pool: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=1)


_state = _RuntimeState()


# ---------------------------------------------------------------------------
# Lifespan — load model once at startup
# ---------------------------------------------------------------------------

def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@asynccontextmanager
async def _lifespan(app: FastAPI):
    model_name = _env("GROUNDING_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    device_map = _env("GROUNDING_DEVICE_MAP", "auto")
    torch_dtype = _env("GROUNDING_TORCH_DTYPE", "auto")
    load_4bit = _env("GROUNDING_LOAD_4BIT", "0") == "1"

    logger.info("Loading model %s (device_map=%s, dtype=%s, 4bit=%s)", model_name, device_map, torch_dtype, load_4bit)
    try:
        from strafer_vlm.inference.qwen_runtime import load_qwen_model_and_processor

        model, processor = load_qwen_model_and_processor(
            model_name_or_path=model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=load_4bit,
        )
        model.eval()

        _state.model = model
        _state.processor = processor
        _state.model_name = model_name
        _state.ready = True
        logger.info("Model loaded and ready.")
    except Exception:
        logger.exception("Failed to load model — service will return 503 on all /ground requests.")
        _state.ready = False

    yield

    _state.ready = False
    _state.model = None
    _state.processor = None


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    max_image_mp = float(_env("GROUNDING_MAX_IMAGE_MP", "20"))
    inference_timeout = float(_env("GROUNDING_INFERENCE_TIMEOUT", "30"))

    app = FastAPI(
        title="Strafer VLM Grounding Service",
        description="Qwen2.5-VL grounding inference exposed as a REST API.",
        version="0.1.0",
        lifespan=_lifespan,
    )

    @app.get("/health", response_model=HealthResponse, summary="Service readiness check")
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if _state.ready else "loading",
            model_loaded=_state.ready,
            model_name=_state.model_name or None,
        )

    @app.post("/ground", response_model=GroundResponse, summary="Run grounding inference on one image")
    async def ground(req: GroundRequest) -> GroundResponse:
        if not _state.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")

        if not req.prompt or not req.prompt.strip():
            raise HTTPException(status_code=400, detail="prompt must be a non-empty string.")

        logger.info("[%s] /ground prompt=%r", req.request_id, req.prompt)

        from PIL import Image

        from strafer_vlm.inference.parsing import (
            SYSTEM_PROMPT_DEFAULT,
            normalize_prediction_bbox_to_1000,
            overlay_bbox,
            parse_grounding_prediction,
        )
        from strafer_vlm.inference.qwen_runtime import run_grounding_generation

        # Decode image
        try:
            image_bytes = base64.b64decode(req.image_jpeg_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")

        # Image size guard — reject images exceeding the configured megapixel limit
        megapixels = (image.width * image.height) / 1_000_000
        if megapixels > max_image_mp:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {megapixels:.1f} MP exceeds {max_image_mp:.0f} MP limit.",
            )

        # Resize if requested
        original_w, original_h = image.width, image.height
        if req.max_image_side > 0 and max(original_w, original_h) > req.max_image_side:
            image.thumbnail((req.max_image_side, req.max_image_side), Image.Resampling.LANCZOS)

        inference_w, inference_h = image.width, image.height

        # Run inference
        max_tokens = int(_env("GROUNDING_MAX_TOKENS", "128"))
        start = time.perf_counter()

        loop = asyncio.get_running_loop()
        try:
            raw_output = await asyncio.wait_for(
                loop.run_in_executor(
                    _state.inference_pool,
                    lambda: run_grounding_generation(
                        model=_state.model,
                        processor=_state.processor,
                        image=image,
                        prompt=req.prompt,
                        system_prompt=SYSTEM_PROMPT_DEFAULT,
                        max_new_tokens=max_tokens,
                        temperature=0.0,
                    ),
                ),
                timeout=inference_timeout if inference_timeout > 0 else None,
            )
        except asyncio.TimeoutError:
            latency_s = time.perf_counter() - start
            logger.error(
                "[%s] inference timed out after %.1fs", req.request_id, latency_s,
            )
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {inference_timeout:.0f}s.",
            )
        latency_s = time.perf_counter() - start

        # Parse and normalise
        prediction = parse_grounding_prediction(raw_output)
        if prediction is None:
            return GroundResponse(
                request_id=req.request_id,
                found=False,
                raw_output=raw_output,
                latency_s=round(latency_s, 4),
            )

        normalized = normalize_prediction_bbox_to_1000(
            prediction,
            image_width=inference_w,
            image_height=inference_h,
        )

        logger.info(
            "[%s] found=%s bbox=%s latency=%.3fs",
            req.request_id, normalized.found, normalized.bbox_2d, latency_s,
        )

        # Generate debug overlay if requested and a bbox was found
        debug_overlay_jpeg_b64: str | None = None
        if req.return_debug_overlay and normalized.found and normalized.bbox_2d:
            overlay_img = overlay_bbox(
                image,
                tuple(normalized.bbox_2d),
                label=normalized.label,
                coordinate_mode="normalized_1000",
            )
            buf = io.BytesIO()
            overlay_img.save(buf, format="JPEG", quality=85)
            debug_overlay_jpeg_b64 = base64.b64encode(buf.getvalue()).decode("ascii")

        return GroundResponse(
            request_id=req.request_id,
            found=normalized.found,
            bbox_2d=list(normalized.bbox_2d) if normalized.bbox_2d else None,
            label=normalized.label,
            confidence=round(normalized.confidence, 4) if normalized.confidence is not None else None,
            raw_output=raw_output,
            latency_s=round(latency_s, 4),
            debug_overlay_jpeg_b64=debug_overlay_jpeg_b64,
        )

    # ------------------------------------------------------------------
    # POST /describe — scene description using the same Qwen2.5-VL model
    # ------------------------------------------------------------------

    DESCRIBE_SYSTEM_PROMPT = (
        "You are a robot vision system. Describe the scene in the image concisely. "
        "List the main objects, surfaces, and spatial layout visible. "
        "Keep your response to 1-3 sentences. Do not speculate about objects not visible."
    )

    @app.post("/describe", response_model=DescribeResponse, summary="Generate a text description of a scene image")
    async def describe(req: DescribeRequest) -> DescribeResponse:
        if not _state.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")

        logger.info("[%s] /describe prompt=%r", req.request_id, req.prompt)

        from PIL import Image

        from strafer_vlm.inference.qwen_runtime import run_grounding_generation

        # Decode image
        try:
            image_bytes = base64.b64decode(req.image_jpeg_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")

        # Image size guard
        megapixels = (image.width * image.height) / 1_000_000
        if megapixels > max_image_mp:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {megapixels:.1f} MP exceeds {max_image_mp:.0f} MP limit.",
            )

        # Resize if requested
        if req.max_image_side > 0 and max(image.width, image.height) > req.max_image_side:
            image.thumbnail((req.max_image_side, req.max_image_side), Image.Resampling.LANCZOS)

        # Run inference with description prompt
        max_tokens = int(_env("GROUNDING_MAX_TOKENS", "128"))
        start = time.perf_counter()

        loop = asyncio.get_running_loop()
        try:
            raw_output = await asyncio.wait_for(
                loop.run_in_executor(
                    _state.inference_pool,
                    lambda: run_grounding_generation(
                        model=_state.model,
                        processor=_state.processor,
                        image=image,
                        prompt=req.prompt,
                        system_prompt=DESCRIBE_SYSTEM_PROMPT,
                        max_new_tokens=max_tokens,
                        temperature=0.0,
                    ),
                ),
                timeout=inference_timeout if inference_timeout > 0 else None,
            )
        except asyncio.TimeoutError:
            latency_s = time.perf_counter() - start
            logger.error("[%s] describe inference timed out after %.1fs", req.request_id, latency_s)
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {inference_timeout:.0f}s.",
            )
        latency_s = time.perf_counter() - start

        logger.info("[%s] describe completed in %.3fs", req.request_id, latency_s)

        return DescribeResponse(
            request_id=req.request_id,
            description=raw_output,
            latency_s=round(latency_s, 4),
        )

    # ------------------------------------------------------------------
    # POST /detect_objects — multi-object detection (Section 1.12)
    # ------------------------------------------------------------------

    DETECT_OBJECTS_SYSTEM_PROMPT = (
        "You are a robot vision system. List every salient object visible in "
        "the image using the Qwen grounding format: "
        "<ref>label</ref><box>(x1,y1),(x2,y2)</box>. "
        "Emit one <ref>/<box> pair per object. Coordinates are in the 0..1000 "
        "normalized range. Do not include prose or explanations."
    )
    DETECT_OBJECTS_USER_PROMPT = (
        "Locate: List all visible objects with their bounding boxes."
    )

    @app.post(
        "/detect_objects",
        response_model=DetectObjectsResponse,
        summary="Detect all visible objects with bounding boxes in one image",
    )
    async def detect_objects(req: DetectObjectsRequest) -> DetectObjectsResponse:
        if not _state.ready:
            raise HTTPException(status_code=503, detail="Model is not loaded yet.")

        if req.max_objects <= 0:
            raise HTTPException(status_code=400, detail="max_objects must be > 0.")
        if not 0.0 <= req.min_confidence <= 1.0:
            raise HTTPException(status_code=400, detail="min_confidence must be in [0, 1].")

        logger.info(
            "[%s] /detect_objects max_objects=%d min_conf=%.2f",
            req.request_id, req.max_objects, req.min_confidence,
        )

        from PIL import Image

        from strafer_vlm.inference.parsing import (
            bbox_to_pixel_coords,
            parse_ref_box_detections,
        )
        from strafer_vlm.inference.qwen_runtime import run_grounding_generation

        try:
            image_bytes = base64.b64decode(req.image_jpeg_b64)
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid image payload: {exc}")

        megapixels = (image.width * image.height) / 1_000_000
        if megapixels > max_image_mp:
            raise HTTPException(
                status_code=400,
                detail=f"Image too large: {megapixels:.1f} MP exceeds {max_image_mp:.0f} MP limit.",
            )

        if req.max_image_side > 0 and max(image.width, image.height) > req.max_image_side:
            image.thumbnail((req.max_image_side, req.max_image_side), Image.Resampling.LANCZOS)

        inference_w, inference_h = image.width, image.height

        # Allow more tokens than /ground — multi-object output is longer.
        max_tokens = int(_env("GROUNDING_DETECT_MAX_TOKENS", "512"))
        start = time.perf_counter()

        loop = asyncio.get_running_loop()
        try:
            raw_output = await asyncio.wait_for(
                loop.run_in_executor(
                    _state.inference_pool,
                    lambda: run_grounding_generation(
                        model=_state.model,
                        processor=_state.processor,
                        image=image,
                        prompt=DETECT_OBJECTS_USER_PROMPT,
                        system_prompt=DETECT_OBJECTS_SYSTEM_PROMPT,
                        max_new_tokens=max_tokens,
                        temperature=0.0,
                    ),
                ),
                timeout=inference_timeout if inference_timeout > 0 else None,
            )
        except asyncio.TimeoutError:
            latency_s = time.perf_counter() - start
            logger.error(
                "[%s] detect_objects inference timed out after %.1fs",
                req.request_id, latency_s,
            )
            raise HTTPException(
                status_code=504,
                detail=f"Inference timed out after {inference_timeout:.0f}s.",
            )
        latency_s = time.perf_counter() - start

        parsed = parse_ref_box_detections(raw_output)
        filtered: list[DetectedObject] = []
        for det in parsed:
            if det.confidence < req.min_confidence:
                continue
            pixel_bbox = bbox_to_pixel_coords(
                det.bbox_2d,
                image_width=inference_w,
                image_height=inference_h,
                coordinate_mode="normalized_1000",
            )
            filtered.append(
                DetectedObject(
                    label=det.label,
                    bbox_2d=list(pixel_bbox),
                    confidence=round(det.confidence, 4),
                )
            )
            if len(filtered) >= req.max_objects:
                break

        logger.info(
            "[%s] detect_objects found=%d latency=%.3fs",
            req.request_id, len(filtered), latency_s,
        )

        return DetectObjectsResponse(
            request_id=req.request_id,
            objects=filtered,
            raw_output=raw_output,
            latency_s=round(latency_s, 4),
        )

    return app
