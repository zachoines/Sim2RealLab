"""Shared pytest fixtures for strafer_vlm tests."""

from __future__ import annotations

import base64
import io
from contextlib import asynccontextmanager
from unittest.mock import MagicMock

import pytest
from PIL import Image


@pytest.fixture()
def tiny_jpeg_b64() -> str:
    """Return a base64-encoded 64x64 red JPEG."""
    img = Image.new("RGB", (64, 64), color="red")
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return base64.b64encode(buf.getvalue()).decode()


@pytest.fixture()
def tiny_jpeg_path(tmp_path):
    """Write a 64x64 red JPEG to disk and return its Path."""
    img = Image.new("RGB", (64, 64), color="red")
    path = tmp_path / "test_image.jpg"
    img.save(str(path), format="JPEG")
    return path


@pytest.fixture()
def mock_vlm_client():
    """Return a TestClient against the FastAPI app with a mocked model."""
    from strafer_vlm.service.app import _state, create_app

    _state.model = MagicMock()
    _state.processor = MagicMock()
    _state.model_name = "mock-model"
    _state.ready = True

    @asynccontextmanager
    async def _noop_lifespan(app):
        yield

    app = create_app()
    app.router.lifespan_context = _noop_lifespan

    from starlette.testclient import TestClient

    with TestClient(app) as c:
        yield c

    _state.ready = False
    _state.model = None
    _state.processor = None
