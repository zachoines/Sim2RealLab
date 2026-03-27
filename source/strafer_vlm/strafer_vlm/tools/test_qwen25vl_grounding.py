#!/usr/bin/env python3
"""Single-image smoke test for Qwen2.5-VL grounding on Windows.

Example:
    python -m strafer_vlm.test_qwen25vl_grounding ^
      --image data/vlm/images/frame_0001.jpg ^
      --prompt "the chair next to the couch" ^
      --output-image outputs/qwen_smoke_bbox.jpg
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from strafer_vlm.inference.parsing import (
    SYSTEM_PROMPT_DEFAULT,
    normalize_prediction_bbox_to_1000,
    overlay_bbox,
    parse_grounding_prediction,
)
from strafer_vlm.inference.qwen_runtime import (
    get_model_device,
    load_qwen_model_and_processor,
    run_grounding_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Qwen2.5-VL-3B grounding smoke test")
    parser.add_argument("--image", type=str, required=True, help="Path to local image")
    parser.add_argument(
        "--prompt",
        type=str,
        required=True,
        help="Referring expression, e.g. 'the red chair near the wall'",
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
        help="Optional attn implementation, e.g. flash_attention_2",
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
        help="Maximum generated tokens",
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
        "--output-image",
        type=str,
        default=None,
        help="Optional path to save bbox overlay image",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save raw output + parsed JSON",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Downscale image so max(width,height) <= this value before inference",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    image_path = Path(args.image).resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    print("\n=== Qwen2.5-VL Grounding Smoke Test ===")
    print(f"Model: {args.model}")
    print(f"Image: {image_path}")
    print(f"Prompt: {args.prompt}")

    model, processor = load_qwen_model_and_processor(
        model_name_or_path=args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
    )
    model.eval()
    print(f"Model device: {get_model_device(model)}")

    with Image.open(image_path) as pil_image:
        image_rgb = pil_image.convert("RGB")
        if args.max_image_side > 0 and max(image_rgb.width, image_rgb.height) > args.max_image_side:
            image_rgb.thumbnail((args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS)
        print(f"Inference image size: {image_rgb.width}x{image_rgb.height}")
        start = time.perf_counter()
        raw_output = run_grounding_generation(
            model=model,
            processor=processor,
            image=image_rgb,
            prompt=args.prompt,
            system_prompt=args.system_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        latency_s = time.perf_counter() - start

        prediction = parse_grounding_prediction(raw_output)
        if prediction is not None:
            prediction = normalize_prediction_bbox_to_1000(
                prediction,
                image_width=image_rgb.width,
                image_height=image_rgb.height,
            )

        print("\n--- Raw Model Output ---")
        print(raw_output)
        print(f"\nLatency: {latency_s:.2f}s")

        if prediction is None:
            print("\nParsed prediction: <failed to parse JSON>")
        else:
            print("\n--- Parsed Prediction ---")
            print(json.dumps(prediction.__dict__, indent=2))

        if args.output_image and prediction and prediction.found and prediction.bbox_2d:
            output_image_path = Path(args.output_image).resolve()
            output_image_path.parent.mkdir(parents=True, exist_ok=True)
            overlay = overlay_bbox(
                image_rgb,
                prediction.bbox_2d,
                label=prediction.label,
                coordinate_mode=prediction.bbox_coordinate_mode,
            )
            overlay.save(output_image_path)
            print(f"\nSaved bbox visualization: {output_image_path}")

        if args.output_json:
            output_json_path = Path(args.output_json).resolve()
            output_json_path.parent.mkdir(parents=True, exist_ok=True)
            payload = {
                "image": str(image_path),
                "prompt": args.prompt,
                "latency_seconds": latency_s,
                "raw_output": raw_output,
                "parsed": prediction.__dict__ if prediction is not None else None,
            }
            with output_json_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2)
            print(f"Saved JSON output: {output_json_path}")


if __name__ == "__main__":
    main()
