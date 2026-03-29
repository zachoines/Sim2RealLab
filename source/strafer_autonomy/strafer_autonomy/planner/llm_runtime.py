"""Model loading and text generation for the planner LLM.

All ``torch`` and ``transformers`` imports are lazy so this module can be
imported without GPU dependencies (e.g. during testing or on the Jetson).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


class LLMRuntime:
    """Wraps a HuggingFace causal-LM for planner inference."""

    def __init__(self) -> None:
        self.model: Any = None
        self.tokenizer: Any = None
        self.model_name: str = ""
        self.ready: bool = False

    def load(
        self,
        *,
        model_name: str,
        device_map: str = "auto",
        torch_dtype: str = "auto",
        load_4bit: bool = False,
        max_tokens: int = 256,
    ) -> None:
        """Load model and tokenizer into GPU memory."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(
            "Loading planner LLM %s (device_map=%s, dtype=%s, 4bit=%s)",
            model_name, device_map, torch_dtype, load_4bit,
        )

        dtype_map = {
            "auto": "auto",
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        resolved_dtype = dtype_map.get(torch_dtype, "auto")

        kwargs: dict[str, Any] = {
            "device_map": device_map,
            "torch_dtype": resolved_dtype,
        }

        if load_4bit:
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        self.model.eval()
        self.model_name = model_name
        self._max_tokens = max_tokens
        self.ready = True
        logger.info("Planner LLM loaded and ready.")

    def generate(self, messages: list[dict[str, str]]) -> str:
        """Run chat-completion and return the assistant response text.

        Uses non-thinking mode for Qwen3 (``enable_thinking=False``).
        """
        if not self.ready:
            raise RuntimeError("LLM is not loaded. Call load() first.")

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        import torch

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=self._max_tokens,
                do_sample=False,
            )

        # Decode only the generated tokens (skip the prompt)
        generated_ids = output_ids[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(generated_ids, skip_special_tokens=True)
