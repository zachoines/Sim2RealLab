"""Qwen2.5-VL model loading and grounding generation.

Functions in this module lazily import ``torch`` and ``transformers``
so they can be called only when GPU inference is needed.
"""

from __future__ import annotations

from typing import Any

from PIL import Image

from strafer_vlm.inference.parsing import SYSTEM_PROMPT_DEFAULT, normalize_prompt


def _resolve_torch_dtype(dtype_name: str) -> Any:
    import torch

    normalized = dtype_name.strip().lower()
    if normalized == "auto":
        return "auto"
    if normalized in {"float16", "fp16"}:
        return torch.float16
    if normalized in {"bfloat16", "bf16"}:
        return torch.bfloat16
    if normalized in {"float32", "fp32"}:
        return torch.float32
    raise ValueError(f"Unsupported torch dtype: {dtype_name}")


def get_model_device(model: Any) -> Any:
    """Best-effort retrieval of the model device."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        import torch

        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_qwen_model_and_processor(
    *,
    model_name_or_path: str,
    torch_dtype: str = "auto",
    device_map: str = "auto",
    attn_implementation: str | None = None,
    load_in_4bit: bool = False,
    bnb_4bit_compute_dtype: str = "bfloat16",
) -> tuple[Any, Any]:
    """Load Qwen2.5-VL model + processor with optional 4-bit quantization."""
    import torch

    try:
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:  # pragma: no cover - import guard
        raise RuntimeError(
            "Missing transformers dependency. Install with: "
            "pip install transformers accelerate"
        ) from exc

    resolved_dtype = _resolve_torch_dtype(torch_dtype)
    if resolved_dtype == "auto" and torch.cuda.is_available():
        resolved_dtype = torch.float16

    model_kwargs: dict[str, Any] = {
        "torch_dtype": resolved_dtype,
        "device_map": device_map,
    }
    if attn_implementation:
        model_kwargs["attn_implementation"] = attn_implementation
    if load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - import guard
            raise RuntimeError(
                "4-bit loading requested but bitsandbytes support is missing."
            ) from exc
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_resolve_torch_dtype(bnb_4bit_compute_dtype),
        )

    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        **model_kwargs,
    )
    processor = AutoProcessor.from_pretrained(model_name_or_path)
    return model, processor


def run_grounding_generation(
    *,
    model: Any,
    processor: Any,
    image: Image.Image,
    prompt: str,
    system_prompt: str = SYSTEM_PROMPT_DEFAULT,
    max_new_tokens: int = 128,
    temperature: float = 0.0,
    top_p: float = 0.9,
) -> str:
    """Run one grounding inference and return decoded assistant text."""
    user_prompt = normalize_prompt(prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "local_image"},
                {"type": "text", "text": user_prompt},
            ],
        },
    ]
    chat_text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    model_inputs = processor(
        text=[chat_text],
        images=[image.convert("RGB")],
        return_tensors="pt",
    )
    device = get_model_device(model)
    model_inputs = {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in model_inputs.items()
    }

    generation_kwargs: dict[str, Any] = {"max_new_tokens": max_new_tokens}
    if temperature > 0:
        generation_kwargs.update(
            {
                "do_sample": True,
                "temperature": temperature,
                "top_p": top_p,
            }
        )
    else:
        generation_kwargs["do_sample"] = False

    generated = model.generate(**model_inputs, **generation_kwargs)
    prompt_length = model_inputs["input_ids"].shape[1]
    completion_ids = generated[:, prompt_length:]
    decoded = processor.batch_decode(
        completion_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return decoded[0].strip()
