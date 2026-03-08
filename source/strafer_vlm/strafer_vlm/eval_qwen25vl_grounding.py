#!/usr/bin/env python3
"""Offline evaluation routine for Qwen2.5-VL grounding datasets.

Example:
    python -m strafer_vlm.eval_qwen25vl_grounding ^
      --dataset data/vlm/test.jsonl ^
      --model Qwen/Qwen2.5-VL-3B-Instruct ^
      --adapter outputs/qwen25vl_lora_run1 ^
      --output-dir outputs/qwen25vl_eval_run1
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from statistics import mean, median
from typing import Any

from PIL import Image

from strafer_vlm.qwen_vl_common import (
    SYSTEM_PROMPT_DEFAULT,
    bbox_iou,
    load_grounding_dataset,
    load_qwen_model_and_processor,
    normalize_prediction_bbox_to_1000,
    parse_grounding_prediction,
    run_grounding_generation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen2.5-VL grounding outputs")
    parser.add_argument("--dataset", type=str, required=True, help="Path to eval JSON/JSONL dataset")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF model or local merged model path",
    )
    parser.add_argument(
        "--adapter",
        type=str,
        default=None,
        help="Optional LoRA adapter directory to load on top of --model",
    )
    parser.add_argument(
        "--merge-adapter",
        action="store_true",
        help="Merge adapter into base model before evaluation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for predictions and summary",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap for quick validation runs",
    )
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Downscale images so max(width,height) <= this value before inference",
    )

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model load dtype",
    )
    parser.add_argument(
        "--device-map",
        type=str,
        default="auto",
        help="HF device_map argument",
    )
    parser.add_argument(
        "--attn-implementation",
        type=str,
        default=None,
        help="Optional attention implementation",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Load model in 4-bit quantized mode",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for 4-bit model",
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
        help="Generation temperature (0.0 = greedy decoding)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling p when temperature > 0",
    )
    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for Acc@IoU metric",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=SYSTEM_PROMPT_DEFAULT,
        help="System prompt used for generation",
    )
    return parser.parse_args()


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "evaluation_summary.json"

    examples = load_grounding_dataset(args.dataset)
    if args.max_samples is not None:
        examples = examples[: args.max_samples]
    if not examples:
        raise ValueError("Evaluation dataset is empty.")

    print("\n=== Qwen2.5-VL Grounding Evaluation ===")
    print(f"Dataset samples: {len(examples)}")
    print(f"Model: {args.model}")
    if args.adapter:
        print(f"Adapter: {args.adapter}")
    print(f"Output dir: {output_dir}")

    model, processor = load_qwen_model_and_processor(
        model_name_or_path=args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge_adapter:
            model = model.merge_and_unload()
    model.eval()

    total = len(examples)
    parse_success = 0
    found_correct = 0

    gt_positive = 0
    gt_negative = 0
    tp_found = 0
    fp_found = 0
    fn_found = 0
    tn_found = 0

    iou_values: list[float] = []
    latency_values: list[float] = []
    acc_at_thresh_hits = 0
    total_latency_s = 0.0

    with predictions_path.open("w", encoding="utf-8") as prediction_handle:
        for index, example in enumerate(examples, start=1):
            with Image.open(example.image_path) as image_handle:
                image = image_handle.convert("RGB")
                if args.max_image_side > 0 and max(image.width, image.height) > args.max_image_side:
                    image.thumbnail((args.max_image_side, args.max_image_side), Image.Resampling.LANCZOS)

                start = time.perf_counter()
                raw_output = run_grounding_generation(
                    model=model,
                    processor=processor,
                    image=image,
                    prompt=example.prompt,
                    system_prompt=args.system_prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )
                latency_s = time.perf_counter() - start
                total_latency_s += latency_s
                latency_values.append(latency_s)

            prediction = parse_grounding_prediction(raw_output)
            if prediction is not None:
                prediction = normalize_prediction_bbox_to_1000(
                    prediction,
                    image_width=image.width,
                    image_height=image.height,
                )
                parse_success += 1

            gt_found = bool(example.target.found)
            pred_found = bool(prediction.found) if prediction is not None else False

            if gt_found:
                gt_positive += 1
                if pred_found:
                    tp_found += 1
                else:
                    fn_found += 1
            else:
                gt_negative += 1
                if pred_found:
                    fp_found += 1
                else:
                    tn_found += 1

            if gt_found == pred_found:
                found_correct += 1

            iou = bbox_iou(example.target.bbox_2d, prediction.bbox_2d if prediction else None)
            if iou is not None and not math.isnan(iou):
                iou_values.append(iou)
                if gt_found and pred_found and iou >= args.iou_threshold:
                    acc_at_thresh_hits += 1

            row = {
                "index": index,
                "image_path": str(example.image_path),
                "prompt": example.prompt,
                "ground_truth": {
                    "found": example.target.found,
                    "bbox_2d": list(example.target.bbox_2d) if example.target.bbox_2d else None,
                    "label": example.target.label,
                },
                "prediction": (
                    {
                        "found": prediction.found,
                        "bbox_2d": list(prediction.bbox_2d) if prediction.bbox_2d else None,
                        "label": prediction.label,
                        "confidence": prediction.confidence,
                    }
                    if prediction
                    else None
                ),
                "raw_output": raw_output,
                "iou": iou,
                "latency_seconds": latency_s,
            }
            prediction_handle.write(json.dumps(row, ensure_ascii=True) + "\n")

            print(f"[{index:04d}/{total:04d}] parsed={prediction is not None} gt_found={gt_found} pred_found={pred_found} iou={iou}")

    precision = _safe_ratio(tp_found, tp_found + fp_found)
    recall = _safe_ratio(tp_found, tp_found + fn_found)
    f1 = None
    if precision is not None and recall is not None and (precision + recall) > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    summary: dict[str, Any] = {
        "num_examples": total,
        "num_gt_positive": gt_positive,
        "num_gt_negative": gt_negative,
        "parse_success_rate": _safe_ratio(parse_success, total),
        "found_accuracy": _safe_ratio(found_correct, total),
        "precision_found": precision,
        "recall_found": recall,
        "f1_found": f1,
        "mean_iou": mean(iou_values) if iou_values else None,
        "median_latency_seconds": median(latency_values) if latency_values else None,
        "mean_latency_seconds": total_latency_s / total,
        "acc_at_iou_threshold": _safe_ratio(acc_at_thresh_hits, gt_positive),
        "iou_threshold": args.iou_threshold,
        "tp_found": tp_found,
        "fp_found": fp_found,
        "fn_found": fn_found,
        "tn_found": tn_found,
        "predictions_file": str(predictions_path),
    }

    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nEvaluation complete.")
    print(json.dumps(summary, indent=2))
    print(f"Summary file: {summary_path}")
    print(f"Predictions file: {predictions_path}")


if __name__ == "__main__":
    main()
