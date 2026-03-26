#!/usr/bin/env python3
"""LoRA fine-tuning routine for Qwen2.5-VL grounding.

Dataset formats supported:
1) Flat JSON/JSONL
   {"image":"images/f001.jpg","prompt":"Locate: the chair","target":{"found":true,"bbox_2d":[...],"label":"chair"}}
2) Chat JSON/JSONL
   {"messages":[{"role":"user","content":[{"type":"image","image":"images/f001.jpg"},{"type":"text","text":"Locate: the chair"}]},
                {"role":"assistant","content":"{\"found\":true,\"bbox_2d\":[...],\"label\":\"chair\"}"}]}

Example:
    python -m strafer_vlm.train_qwen25vl_lora ^
      --train-dataset data/vlm/train.jsonl ^
      --eval-dataset data/vlm/val.jsonl ^
      --output-dir outputs/qwen25vl_lora_run1
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from PIL import Image

from strafer_vlm.qwen_vl_common import (
    GroundingExample,
    SYSTEM_PROMPT_DEFAULT,
    load_grounding_dataset,
    load_qwen_model_and_processor,
    normalize_prompt,
    serialize_target,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune Qwen2.5-VL-3B with LoRA")
    parser.add_argument("--train-dataset", type=str, required=True, help="Path to train JSON/JSONL dataset")
    parser.add_argument("--eval-dataset", type=str, default=None, help="Optional eval JSON/JSONL dataset")
    parser.add_argument(
        "--eval-split",
        type=float,
        default=0.1,
        help="If eval dataset is omitted, split this fraction from training data",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
        help="Optional cap for fast debugging",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
        help="Optional cap for fast debugging",
    )

    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-VL-3B-Instruct",
        help="HF model name or local path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for checkpoints and adapter weights",
    )
    parser.add_argument(
        "--merged-model-dir",
        type=str,
        default=None,
        help="Optional output directory for merged base+LoRA model",
    )

    parser.add_argument(
        "--torch-dtype",
        type=str,
        default="bfloat16",
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
        help="Optional attention implementation (e.g. flash_attention_2)",
    )
    parser.add_argument(
        "--load-in-4bit",
        action="store_true",
        help="Enable QLoRA-style 4-bit loading (bitsandbytes required)",
    )
    parser.add_argument(
        "--bnb-4bit-compute-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Compute dtype for 4-bit quantized model",
    )
    parser.add_argument(
        "--freeze-vision",
        action="store_true",
        default=True,
        help="Freeze vision tower and multimodal projector params",
    )
    parser.add_argument(
        "--no-freeze-vision",
        action="store_false",
        dest="freeze_vision",
        help="Allow training on vision-side params (not recommended for this use case)",
    )

    parser.add_argument("--lora-r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05, help="LoRA dropout")
    parser.add_argument(
        "--lora-target-modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj",
        help="Comma-separated module names for LoRA injection",
    )

    parser.add_argument("--num-train-epochs", type=float, default=5.0, help="Training epochs")
    parser.add_argument("--learning-rate", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight decay")
    parser.add_argument(
        "--lr-scheduler-type",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
        help="Learning-rate schedule",
    )
    parser.add_argument("--warmup-ratio", type=float, default=0.1, help="Warmup ratio")
    parser.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm")

    parser.add_argument("--per-device-train-batch-size", type=int, default=2, help="Train batch size")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=2, help="Eval batch size")
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=True,
        help="Enable gradient checkpointing",
    )
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_false",
        dest="gradient_checkpointing",
        help="Disable gradient checkpointing",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=2048,
        help="Tokenizer max sequence length",
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=0, help="Dataloader workers")
    parser.add_argument(
        "--max-image-side",
        type=int,
        default=1024,
        help="Downscale images so max(width,height) <= this value before training",
    )

    parser.add_argument("--logging-steps", type=int, default=10, help="Train logging interval")
    parser.add_argument("--save-steps", type=int, default=200, help="Checkpoint interval")
    parser.add_argument("--eval-steps", type=int, default=200, help="Eval interval when eval set exists")
    parser.add_argument("--save-total-limit", type=int, default=3, help="Max checkpoints to retain")
    parser.add_argument(
        "--report-to",
        type=str,
        default="none",
        help="Trainer report target, e.g. none, tensorboard, wandb",
    )
    parser.add_argument(
        "--merge-adapter-after-training",
        action="store_true",
        help="Merge LoRA adapter into a standalone model at the end",
    )
    parser.add_argument(
        "--system-prompt",
        type=str,
        default=SYSTEM_PROMPT_DEFAULT,
        help="System prompt used during SFT",
    )
    return parser.parse_args()


def _split_dataset(
    examples: list[GroundingExample],
    *,
    eval_split: float,
    seed: int,
) -> tuple[list[GroundingExample], list[GroundingExample]]:
    if not 0.0 <= eval_split < 1.0:
        raise ValueError("--eval-split must be in [0.0, 1.0).")
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    eval_count = int(round(len(shuffled) * eval_split))
    eval_count = max(1, eval_count) if len(shuffled) > 1 and eval_count > 0 else eval_count
    if eval_count == 0:
        return shuffled, []
    return shuffled[eval_count:], shuffled[:eval_count]


def _freeze_vision_modules(model: Any) -> int:
    patterns = ("visual", "vision_tower", "multi_modal_projector", "image_tower")
    frozen = 0
    for name, param in model.named_parameters():
        if any(pattern in name for pattern in patterns):
            param.requires_grad = False
            frozen += 1
    return frozen


class GroundingSFTDataset:
    """Simple dataset wrapper returning paths and serialized targets."""

    def __init__(self, examples: list[GroundingExample]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        example = self.examples[index]
        return {
            "image_path": str(example.image_path),
            "prompt": normalize_prompt(example.prompt),
            "target_json": serialize_target(example.target),
        }


class GroundingDataCollator:
    """Build multimodal SFT batches and mask non-assistant tokens from loss."""

    def __init__(
        self,
        *,
        processor: Any,
        system_prompt: str,
        max_seq_length: int,
        max_image_side: int,
    ):
        self.processor = processor
        self.system_prompt = system_prompt
        self.max_seq_length = max_seq_length
        self.max_image_side = max_image_side

    def __call__(self, features: list[dict[str, str]]) -> dict[str, Any]:
        images: list[Image.Image] = []
        full_texts: list[str] = []
        prompt_texts: list[str] = []

        for feature in features:
            image_path = Path(feature["image_path"])
            with Image.open(image_path) as image_handle:
                image = image_handle.convert("RGB")
                if self.max_image_side > 0 and max(image.width, image.height) > self.max_image_side:
                    image.thumbnail((self.max_image_side, self.max_image_side), Image.Resampling.LANCZOS)
                images.append(image)

            prompt = normalize_prompt(feature["prompt"])
            target_json = feature["target_json"]

            prompt_messages = [
                {"role": "system", "content": self.system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path.as_posix()},
                        {"type": "text", "text": prompt},
                    ],
                },
            ]
            full_messages = prompt_messages + [
                {"role": "assistant", "content": target_json},
            ]

            prompt_texts.append(
                self.processor.apply_chat_template(
                    prompt_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )
            full_texts.append(
                self.processor.apply_chat_template(
                    full_messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            )

        batch_inputs = self.processor(
            text=full_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )
        prompt_inputs = self.processor(
            text=prompt_texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=self.max_seq_length,
            return_tensors="pt",
        )

        labels = batch_inputs["input_ids"].clone()
        pad_token_id = self.processor.tokenizer.pad_token_id
        if pad_token_id is not None:
            labels[labels == pad_token_id] = -100

        for row_idx in range(labels.shape[0]):
            prompt_len = int(prompt_inputs["attention_mask"][row_idx].sum().item())
            prompt_len = min(prompt_len, labels.shape[1])
            labels[row_idx, :prompt_len] = -100

        batch_inputs["labels"] = labels
        return batch_inputs


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    train_examples = load_grounding_dataset(args.train_dataset)
    if args.max_train_samples is not None:
        train_examples = train_examples[: args.max_train_samples]

    if args.eval_dataset:
        eval_examples = load_grounding_dataset(args.eval_dataset)
    else:
        train_examples, eval_examples = _split_dataset(
            train_examples,
            eval_split=args.eval_split,
            seed=args.seed,
        )

    if args.max_eval_samples is not None:
        eval_examples = eval_examples[: args.max_eval_samples]

    if not train_examples:
        raise ValueError("Training set is empty after filtering.")
    if eval_examples and args.eval_steps <= 0:
        raise ValueError("--eval-steps must be > 0 when an eval dataset is used.")
    if eval_examples and args.save_steps % args.eval_steps != 0:
        raise ValueError("--save-steps must be a multiple of --eval-steps when eval is enabled.")

    print("\n=== Qwen2.5-VL LoRA Training ===")
    print(f"Model: {args.model}")
    print(f"Train examples: {len(train_examples)}")
    print(f"Eval examples: {len(eval_examples)}")
    print(f"Output dir: {output_dir}")

    model, processor = load_qwen_model_and_processor(
        model_name_or_path=args.model,
        torch_dtype=args.torch_dtype,
        device_map=args.device_map,
        attn_implementation=args.attn_implementation,
        load_in_4bit=args.load_in_4bit,
        bnb_4bit_compute_dtype=args.bnb_4bit_compute_dtype,
    )

    if args.load_in_4bit:
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=args.gradient_checkpointing,
        )

    if args.freeze_vision:
        frozen_params = _freeze_vision_modules(model)
        print(f"Froze {frozen_params} vision-related parameters.")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    from peft import LoraConfig, get_peft_model
    from transformers import Trainer, TrainingArguments
    import inspect

    lora_target_modules = [token.strip() for token in args.lora_target_modules.split(",") if token.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=lora_target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    if hasattr(model, "print_trainable_parameters"):
        model.print_trainable_parameters()

    train_dataset = GroundingSFTDataset(train_examples)
    eval_dataset = GroundingSFTDataset(eval_examples) if eval_examples else None
    collator = GroundingDataCollator(
        processor=processor,
        system_prompt=args.system_prompt,
        max_seq_length=args.max_seq_length,
        max_image_side=args.max_image_side,
    )

    report_to = [] if args.report_to.lower() == "none" else [args.report_to]
    training_arg_names = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = "evaluation_strategy" if "evaluation_strategy" in training_arg_names else "eval_strategy"
    training_kwargs: dict[str, Any] = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "lr_scheduler_type": args.lr_scheduler_type,
        "warmup_ratio": args.warmup_ratio,
        "max_grad_norm": args.max_grad_norm,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "save_strategy": "steps",
        "save_total_limit": args.save_total_limit,
        "remove_unused_columns": False,
        "dataloader_num_workers": args.dataloader_num_workers,
        "gradient_checkpointing": args.gradient_checkpointing,
        "report_to": report_to,
        "seed": args.seed,
        "bf16": args.torch_dtype == "bfloat16",
        "fp16": args.torch_dtype == "float16",
    }
    if eval_dataset is not None:
        training_kwargs.update(
            {
                eval_key: "steps",
                "eval_steps": args.eval_steps,
                "load_best_model_at_end": True,
                "metric_for_best_model": "eval_loss",
                "greater_is_better": False,
            }
        )
    else:
        training_kwargs[eval_key] = "no"

    training_args = TrainingArguments(**training_kwargs)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    train_result = trainer.train()
    trainer.save_state()
    trainer.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))

    summary: dict[str, Any] = {
        "train_metrics": dict(train_result.metrics),
        "train_dataset_size": len(train_examples),
        "eval_dataset_size": len(eval_examples),
        "config": {
            "model": args.model,
            "lora": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout,
                "target_modules": lora_target_modules,
            },
            "training": {
                "epochs": args.num_train_epochs,
                "learning_rate": args.learning_rate,
                "batch_size_per_device": args.per_device_train_batch_size,
                "gradient_accumulation_steps": args.gradient_accumulation_steps,
                "torch_dtype": args.torch_dtype,
                "load_in_4bit": args.load_in_4bit,
            },
        },
    }

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        summary["eval_metrics"] = eval_metrics

    if args.merge_adapter_after_training:
        merge_dir = (
            Path(args.merged_model_dir).resolve()
            if args.merged_model_dir
            else output_dir / "merged_model"
        )
        merge_dir.mkdir(parents=True, exist_ok=True)
        try:
            merged_model = trainer.model.merge_and_unload()
            merged_model.save_pretrained(str(merge_dir), safe_serialization=True)
            processor.save_pretrained(str(merge_dir))
            summary["merged_model_dir"] = str(merge_dir)
            print(f"Merged model saved to: {merge_dir}")
        except Exception as exc:  # noqa: BLE001
            summary["merge_error"] = str(exc)
            print(f"[WARN] Failed to merge adapter: {exc}")

    summary_path = output_dir / "training_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print("\nTraining complete.")
    print(f"Adapter output: {output_dir}")
    print(f"Summary file: {summary_path}")


if __name__ == "__main__":
    main()
