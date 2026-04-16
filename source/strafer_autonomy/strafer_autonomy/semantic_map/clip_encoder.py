"""OpenCLIP ViT-B/32 ONNX wrapper for image and text embedding."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

_logger = logging.getLogger(__name__)

_DEFAULT_MODEL_DIR = "~/.strafer/models"
_VISUAL_FILENAME = "clip_visual.onnx"
_TEXT_FILENAME = "clip_text.onnx"
_FALLBACK_FILENAME = "clip_vit_b32.onnx"
_EMBEDDING_DIM = 512
_IMAGE_SIZE = 224

# OpenCLIP ViT-B/32 normalization constants (ImageNet)
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _preprocess_image(image_rgb: np.ndarray) -> np.ndarray:
    """Resize, center-crop, normalize an RGB uint8 image to CLIP input tensor."""
    h, w = image_rgb.shape[:2]
    short_side = min(h, w)
    scale = _IMAGE_SIZE / short_side
    new_h, new_w = int(h * scale), int(w * scale)

    try:
        import cv2
        resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    except ImportError:
        from PIL import Image
        pil = Image.fromarray(image_rgb).resize((new_w, new_h), Image.BILINEAR)
        resized = np.array(pil)

    top = (new_h - _IMAGE_SIZE) // 2
    left = (new_w - _IMAGE_SIZE) // 2
    cropped = resized[top : top + _IMAGE_SIZE, left : left + _IMAGE_SIZE]

    img = cropped.astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    # HWC -> 1CHW
    return np.expand_dims(img.transpose(2, 0, 1), axis=0)


class CLIPEncoder:
    """ONNX-based CLIP encoder with separate image and text towers.

    Gracefully degrades to disabled state if ONNX models are not found.
    """

    def __init__(self, model_dir: str = _DEFAULT_MODEL_DIR) -> None:
        self._model_dir = Path(model_dir).expanduser()
        self._visual_session: Any = None
        self._text_session: Any = None
        self._enabled = False
        self._load_models()

    @property
    def enabled(self) -> bool:
        return self._enabled

    def _load_models(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            _logger.warning("onnxruntime not installed; CLIP encoder disabled")
            return

        providers = ort.get_available_providers()

        visual_path = self._model_dir / _VISUAL_FILENAME
        text_path = self._model_dir / _TEXT_FILENAME
        fallback_path = self._model_dir / _FALLBACK_FILENAME

        if visual_path.exists() and text_path.exists():
            self._visual_session = ort.InferenceSession(
                str(visual_path), providers=providers,
            )
            self._text_session = ort.InferenceSession(
                str(text_path), providers=providers,
            )
            self._enabled = True
            _logger.info("CLIP encoder loaded (split towers) from %s", self._model_dir)
        elif fallback_path.exists():
            session = ort.InferenceSession(
                str(fallback_path), providers=providers,
            )
            self._visual_session = session
            self._text_session = session
            self._enabled = True
            _logger.info("CLIP encoder loaded (single model) from %s", fallback_path)
        else:
            _logger.warning(
                "CLIP ONNX models not found in %s; encoder disabled", self._model_dir,
            )

    def encode_image(self, image_rgb: np.ndarray) -> np.ndarray:
        """Encode an RGB uint8 HxWx3 image to a 512-dim L2-normalized vector."""
        if not self._enabled or self._visual_session is None:
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        input_tensor = _preprocess_image(image_rgb)
        input_name = self._visual_session.get_inputs()[0].name
        output_name = self._visual_session.get_outputs()[0].name
        result = self._visual_session.run([output_name], {input_name: input_tensor})
        embedding = result[0].flatten().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding

    def encode_text(self, text: str) -> np.ndarray:
        """Encode text to a 512-dim L2-normalized vector."""
        if not self._enabled or self._text_session is None:
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        try:
            from open_clip import get_tokenizer
            tokenizer = get_tokenizer("ViT-B-32")
            tokens = tokenizer([text]).numpy().astype(np.int64)
        except ImportError:
            _logger.warning("open_clip not available for tokenization; returning zeros")
            return np.zeros(_EMBEDDING_DIM, dtype=np.float32)

        input_name = self._text_session.get_inputs()[0].name
        output_name = self._text_session.get_outputs()[0].name
        result = self._text_session.run([output_name], {input_name: tokens})
        embedding = result[0].flatten().astype(np.float32)

        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding /= norm
        return embedding
