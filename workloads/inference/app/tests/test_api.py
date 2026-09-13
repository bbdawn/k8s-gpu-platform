"""API tests.

Two groups:
  * the default tests never load the OCR model, so they run on any CPU machine;
  * the real inference test runs only with RUN_OCR_TESTS=1 and a working
    PaddleOCR install (GPU or CPU).

Run from the `app` directory:  pytest tests
"""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from unittest import mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

# Tests live next to the app modules, which are imported flat (main, ocr, gpu).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import main  # noqa: E402
import ocr  # noqa: E402

# Captured before any fixture patches it, so the real-engine fixture below
# is not affected by the mock installed by the fast `client` fixture.
REAL_CREATE_OCR_ENGINE = ocr.create_ocr_engine


def make_png(text: str = "TEST 12345") -> bytes:
    """Small in-memory PNG with some text on it."""
    image = Image.new("RGB", (480, 160), "white")
    ImageDraw.Draw(image).text((20, 60), text, fill="black")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@pytest.fixture(scope="module")
def client():
    """Client whose OCR engine intentionally fails to load, so no model is pulled."""
    with mock.patch.object(
        ocr, "create_ocr_engine", side_effect=RuntimeError("model not loaded in tests")
    ):
        with TestClient(main.app) as test_client:
            yield test_client


@pytest.fixture(scope="module")
def ocr_client():
    """Client with the real engine. Skipped unless explicitly enabled."""
    if os.getenv("RUN_OCR_TESTS") != "1":
        pytest.skip("set RUN_OCR_TESTS=1 to run tests that load the OCR model")
    with mock.patch.object(ocr, "create_ocr_engine", REAL_CREATE_OCR_ENGINE):
        with TestClient(main.app) as test_client:
            if main.app.state.ocr_engine is None:
                pytest.skip(f"OCR engine unavailable: {main.app.state.ocr_error}")
            yield test_client


# --- /health ----------------------------------------------------------------


def test_health_returns_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- /gpu -------------------------------------------------------------------


def test_gpu_reports_state_without_crashing(client):
    """Must answer on a CPU-only machine too, just with cuda_available=false."""
    response = client.get("/gpu")
    assert response.status_code == 200

    body = response.json()
    assert isinstance(body["cuda_available"], bool)
    assert isinstance(body["gpu_count"], int)
    assert body["ocr_device"] in {"cpu", "gpu:0"} or body["ocr_device"].startswith("gpu:")
    # The engine is deliberately broken in this fixture.
    assert body["ocr_ready"] is False


# --- /ocr: bad requests -----------------------------------------------------


def test_ocr_without_file_is_rejected(client):
    response = client.post("/ocr")
    assert response.status_code == 422


def test_ocr_with_empty_file_is_rejected(client):
    response = client.post("/ocr", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_ocr_with_unsupported_extension_is_rejected(client):
    response = client.post("/ocr", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert response.status_code == 400


def test_ocr_with_undecodable_image_is_rejected(client):
    response = client.post("/ocr", files={"file": ("broken.png", b"not-an-image", "image/png")})
    assert response.status_code == 400
    assert "decode" in response.json()["detail"].lower()


def test_ocr_returns_503_when_engine_failed_to_load(client):
    """A valid image still cannot be processed without an engine."""
    response = client.post("/ocr", files={"file": ("ok.png", make_png(), "image/png")})
    assert response.status_code == 503
    assert "not available" in response.json()["detail"]


# --- /ocr: real inference ---------------------------------------------------


def test_ocr_returns_expected_shape(ocr_client):
    response = ocr_client.post("/ocr", files={"file": ("receipt.png", make_png(), "image/png")})
    assert response.status_code == 200

    body = response.json()
    assert body["count"] == len(body["text"])
    assert body["elapsed_ms"] > 0
    for item in body["text"]:
        assert isinstance(item["text"], str)
        assert 0.0 <= item["confidence"] <= 1.0
