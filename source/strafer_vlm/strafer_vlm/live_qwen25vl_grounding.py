#!/usr/bin/env python3
"""Live video grounding evaluation with runtime console prompt updates.

Example:
    python -m strafer_vlm.live_qwen25vl_grounding ^
      --source 0 ^
      --prompt "Locate: TV on dresser."

Console commands:
    Any non-empty line  Replace the active prompt
    status              Print current prompt and last result summary
    clear               Clear the active prompt
    quit                Stop the script
"""

from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image

from strafer_vlm.qwen_vl_common import (
    SYSTEM_PROMPT_DEFAULT,
    bbox_to_pixel_coords,
    get_model_device,
    load_qwen_model_and_processor,
    normalize_prediction_bbox_to_1000,
    parse_grounding_prediction,
    run_grounding_generation,
)

try:
    import cv2
except ImportError as exc:  # pragma: no cover - import guard
    raise RuntimeError(
        "Missing OpenCV dependency. Install with: pip install opencv-python"
    ) from exc


@dataclass
class InferenceResult:
    """One completed grounding inference."""

    frame_index: int
    prompt: str
    raw_output: str
    latency_seconds: float
    inference_width: int
    inference_height: int
    frame_bgr: Any | None = None
    prediction: Any | None = None
    error: str | None = None
    completed_at: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live Qwen2.5-VL grounding evaluation")
    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Camera index or video path (default: 0)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="",
        help="Initial prompt. Leave empty to wait for console input.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF model name or local model path",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=SYSTEM_PROMPT_DEFAULT,
        help="System prompt used for JSON grounding output",
    )
    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Torch dtype for model loading",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="HF device_map argument (e.g. auto, cuda:0, cpu)",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attention implementation, e.g. flash_attention_2",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load the model in 4-bit quantized mode (bitsandbytes required)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=128,
        help="Maximum generated tokens per query",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 = greedy)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling p when temperature > 0",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Downscale frame so max(width,height) <= this value before inference",
    )
    parser.add_argument(
        "--inference-interval",
        type=float,
        default=1.0,
        help="Minimum seconds between submitted inferences",
    )
    parser.add_argument(
        "--window-name",
        type=str,
        default="Qwen2.5-VL Live Eval",
        help="OpenCV display window name",
    )
    parser.add_argument(
        "--output-jsonl",
        type=str,
        default=None,
        help="Optional path to append per-inference JSONL records",
    )
    parser.add_argument(
        "--max-inferences",
        type=int,
        default=0,
        help="Optional limit for completed inferences (0 = unlimited)",
    )
    parser.add_argument(
        "--print-raw-output",
        action="store_true",
        help="Print raw model JSON for each completed inference",
    )
    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Run headless without opening an OpenCV window",
    )
    return parser.parse_args()


def parse_source(source: str) -> int | str:
    stripped = source.strip()
    if stripped.isdigit():
        return int(stripped)
    return stripped


def open_capture(source: int | str) -> Any:
    if isinstance(source, int) and sys.platform.startswith("win"):
        return cv2.VideoCapture(source, cv2.CAP_DSHOW)
    return cv2.VideoCapture(source)


def read_console_lines(command_queue: queue.Queue[str], stop_event: threading.Event) -> None:
    while not stop_event.is_set():
        try:
            line = input()
        except EOFError:
            break
        command_queue.put(line.strip())


def prepare_frame_for_inference(frame_bgr: Any, max_image_side: int) -> Image.Image:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    if max_image_side > 0 and max(image.width, image.height) > max_image_side:
        image.thumbnail((max_image_side, max_image_side), Image.Resampling.LANCZOS)
    return image


def run_inference_job(
    *,
    model: Any,
    processor: Any,
    frame_bgr: Any,
    frame_index: int,
    prompt: str,
    system_prompt: str,
    max_image_side: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    result_queue: queue.Queue[InferenceResult],
) -> None:
    image = prepare_frame_for_inference(frame_bgr, max_image_side)
    start = time.perf_counter()
    try:
        raw_output = run_grounding_generation(
            model=model,
            processor=processor,
            image=image,
            prompt=prompt,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
        latency_s = time.perf_counter() - start
        prediction = parse_grounding_prediction(raw_output)
        if prediction is not None:
            prediction = normalize_prediction_bbox_to_1000(
                prediction,
                image_width=image.width,
                image_height=image.height,
            )
        result_queue.put(
            InferenceResult(
                frame_index=frame_index,
                prompt=prompt,
                raw_output=raw_output,
                latency_seconds=latency_s,
                inference_width=image.width,
                inference_height=image.height,
                frame_bgr=frame_bgr.copy(),
                prediction=prediction,
                completed_at=time.time(),
            )
        )
    except Exception as exc:  # noqa: BLE001
        result_queue.put(
            InferenceResult(
                frame_index=frame_index,
                prompt=prompt,
                raw_output="",
                latency_seconds=time.perf_counter() - start,
                inference_width=image.width,
                inference_height=image.height,
                frame_bgr=frame_bgr.copy(),
                error=str(exc),
                completed_at=time.time(),
            )
        )


def format_result_summary(result: InferenceResult | None) -> str:
    if result is None:
        return "no result yet"
    if result.error:
        return f"error: {result.error}"
    if result.prediction is None:
        return f"frame {result.frame_index}: parse failed ({result.latency_seconds:.2f}s)"
    if not result.prediction.found:
        return f"frame {result.frame_index}: not found ({result.latency_seconds:.2f}s)"
    return (
        f"frame {result.frame_index}: {result.prediction.label or 'target'} "
        f"({result.latency_seconds:.2f}s)"
    )


def draw_status_panel(frame_bgr: Any, lines: list[str]) -> None:
    if not lines:
        return
    line_height = 22
    padding = 10
    panel_height = padding * 2 + line_height * len(lines)
    panel_width = min(frame_bgr.shape[1] - 20, 960)
    overlay = frame_bgr.copy()
    cv2.rectangle(overlay, (10, 10), (10 + panel_width, 10 + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.55, frame_bgr, 0.45, 0.0, frame_bgr)
    y = 10 + padding + 15
    for line in lines:
        cv2.putText(
            frame_bgr,
            line[:120],
            (20, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        y += line_height


def draw_prediction_overlay(frame_bgr: Any, result: InferenceResult | None) -> None:
    if result is None or result.error or result.prediction is None:
        return
    prediction = result.prediction
    if not prediction.found or prediction.bbox_2d is None:
        return
    x1, y1, x2, y2 = bbox_to_pixel_coords(
        prediction.bbox_2d,
        image_width=frame_bgr.shape[1],
        image_height=frame_bgr.shape[0],
        coordinate_mode=prediction.bbox_coordinate_mode,
    )
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    label = prediction.label or "target"
    if prediction.confidence is not None:
        label = f"{label} ({prediction.confidence:.2f})"
    cv2.putText(
        frame_bgr,
        label[:80],
        (x1, max(24, y1 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def write_jsonl_record(output_path: Path, result: InferenceResult) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp": result.completed_at,
        "frame_index": result.frame_index,
        "prompt": result.prompt,
        "latency_seconds": result.latency_seconds,
        "inference_width": result.inference_width,
        "inference_height": result.inference_height,
        "raw_output": result.raw_output,
        "error": result.error,
        "prediction": (
            result.prediction.__dict__ if result.prediction is not None else None
        ),
    }
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")


def main() -> None:
    args = parse_args()
    source = parse_source(args.source)
    output_jsonl = Path(args.output_jsonl).resolve() if args.output_jsonl else None

    cap = open_capture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {args.source}")

    print("\n=== Qwen2.5-VL Live Evaluation ===")
    print(f"Model: {args.model}")
    print(f"Source: {args.source}")
    print("Console: type a new query and press Enter to update the prompt.")
    print("Console commands: status | clear | quit")

    model, processor = load_qwen_model_and_processor(
        model_name_or_path=args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()
    print(f"Model device: {get_model_device(model)}")

    current_prompt = args.prompt.strip()
    if current_prompt:
        print(f"Initial prompt: {current_prompt}")
    else:
        print("Initial prompt: <empty, waiting for console input>")

    stop_event = threading.Event()
    command_queue: queue.Queue[str] = queue.Queue()
    result_queue: queue.Queue[InferenceResult] = queue.Queue()
    input_thread = threading.Thread(
        target=read_console_lines,
        args=(command_queue, stop_event),
        daemon=True,
    )
    input_thread.start()

    worker_thread: threading.Thread | None = None
    last_result: InferenceResult | None = None
    frame_index = 0
    completed_inferences = 0
    next_inference_at = 0.0
    source_is_camera = isinstance(source, int)

    try:
        while not stop_event.is_set():
            ok, frame_bgr = cap.read()
            if not ok:
                if source_is_camera:
                    time.sleep(0.05)
                    continue
                print("End of video stream.")
                break

            frame_index += 1

            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break
                if not command:
                    continue
                command_lower = command.lower()
                if command_lower in {"quit", "exit", ":q"}:
                    stop_event.set()
                    break
                if command_lower == "status":
                    print(f"Prompt: {current_prompt or '<empty>'}")
                    print(f"Last result: {format_result_summary(last_result)}")
                    continue
                if command_lower == "clear":
                    current_prompt = ""
                    print("Cleared active prompt.")
                    continue
                current_prompt = command
                print(f"Updated prompt: {current_prompt}")

            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break
                last_result = result
                worker_thread = None
                completed_inferences += 1
                if output_jsonl is not None:
                    write_jsonl_record(output_jsonl, result)
                if result.error:
                    print(f"[frame {result.frame_index}] inference error: {result.error}")
                else:
                    print(f"[frame {result.frame_index}] {format_result_summary(result)}")
                    if args.print_raw_output:
                        print(result.raw_output)
                if args.max_inferences > 0 and completed_inferences >= args.max_inferences:
                    stop_event.set()
                    break

            now = time.perf_counter()
            if (
                current_prompt
                and worker_thread is None
                and now >= next_inference_at
                and not stop_event.is_set()
            ):
                frame_copy = frame_bgr.copy()
                prompt_copy = current_prompt
                worker_thread = threading.Thread(
                    target=run_inference_job,
                    kwargs={
                        "model": model,
                        "processor": processor,
                        "frame_bgr": frame_copy,
                        "frame_index": frame_index,
                        "prompt": prompt_copy,
                        "system_prompt": args.system_prompt,
                        "max_image_side": args.max_image_side,
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                        "result_queue": result_queue,
                    },
                    daemon=True,
                )
                worker_thread.start()
                next_inference_at = now + max(0.0, args.inference_interval)

            if not args.no_display:
                if last_result is not None and last_result.frame_bgr is not None:
                    display_frame = last_result.frame_bgr.copy()
                else:
                    display_frame = frame_bgr.copy()
                draw_prediction_overlay(display_frame, last_result)
                status_lines = [
                    f"Prompt: {current_prompt or '<empty>'}",
                    f"Last result: {format_result_summary(last_result)}",
                    f"Inference: {'running' if worker_thread is not None else 'idle'}",
                    "Console: type a new query, or use status | clear | quit",
                    "Keyboard: q or Esc to exit",
                ]
                draw_status_panel(display_frame, status_lines)
                cv2.imshow(args.window_name, display_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in {27, ord('q')}:
                    stop_event.set()
            else:
                time.sleep(0.01)

        if worker_thread is not None:
            worker_thread.join()
            while True:
                try:
                    result = result_queue.get_nowait()
                except queue.Empty:
                    break
                last_result = result
                if output_jsonl is not None:
                    write_jsonl_record(output_jsonl, result)
                if result.error:
                    print(f"[frame {result.frame_index}] inference error: {result.error}")
                else:
                    print(f"[frame {result.frame_index}] {format_result_summary(result)}")
                    if args.print_raw_output:
                        print(result.raw_output)
    finally:
        stop_event.set()
        cap.release()
        if not args.no_display:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
