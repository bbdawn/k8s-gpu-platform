"""PaddleOCR engine creation and inference.

The engine is created once (at application startup) and reused for every
request, so that model loading time never leaks into benchmark numbers.
"""

from __future__ import annotations

import io
import logging
import os
import threading

import numpy as np
from PIL import Image, UnidentifiedImageError

from gpu import resolve_device

logger = logging.getLogger(__name__)

# --- configuration (all overridable via env / ConfigMap) ---------------------

# PaddleOCR language model, e.g. korean / en / ch / japan.
OCR_LANG = os.getenv("OCR_LANG", "korean")

# Text-line orientation classification costs extra inference time. Off by
# default so the benchmark measures detection + recognition only.
USE_TEXTLINE_ORIENTATION = os.getenv("OCR_USE_TEXTLINE_ORIENTATION", "false").lower() == "true"

# A PaddleOCR predictor is not thread-safe. This caps how many predict() calls
# may run against the shared engine at once:
#   1 -> safe default, one GPU stream per pod (scale with replicas)
#   N -> allow N in-flight calls, for in-pod concurrency experiments
MAX_CONCURRENCY = max(1, int(os.getenv("OCR_MAX_CONCURRENCY", "1")))

# Reject oversized uploads before decoding them.
MAX_IMAGE_BYTES = int(os.getenv("OCR_MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png"}
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
ALLOWED_IMAGE_FORMATS = {"JPEG", "PNG"}

_predict_slots = threading.BoundedSemaphore(MAX_CONCURRENCY)


class ImageError(ValueError):
    """Raised when the uploaded bytes are not a usable jpg/png image."""


def create_ocr_engine():
    """Build the PaddleOCR engine. Called once during startup."""
    from paddleocr import PaddleOCR

    device = resolve_device()
    logger.info(
        "initializing PaddleOCR (device=%s, lang=%s, textline_orientation=%s, max_concurrency=%d)",
        device,
        OCR_LANG,
        USE_TEXTLINE_ORIENTATION,
        MAX_CONCURRENCY,
    )

    # Document orientation / unwarping are extra pipeline stages we do not need
    # for a benchmark workload; disabling them keeps runs comparable.
    engine = PaddleOCR(
        device=device,
        lang=OCR_LANG,
        use_doc_orientation_classify=False,
        use_doc_unwarping=False,
        use_textline_orientation=USE_TEXTLINE_ORIENTATION,
    )
    logger.info("PaddleOCR ready on %s", device)
    return engine


def warm_up(engine) -> None:
    """Run one throwaway inference so the first real request is not the slowest."""
    blank = np.full((320, 320, 3), 255, dtype=np.uint8)
    try:
        run_ocr(engine, blank)
        logger.info("warm-up inference finished")
    except Exception as exc:  # noqa: BLE001 - warm-up must never block startup
        logger.warning("warm-up inference failed: %s", exc)


def validate_upload(filename: str | None, content_type: str | None) -> None:
    """Cheap checks on the upload metadata, before reading the body."""
    extension = os.path.splitext(filename or "")[1].lower()
    if extension and extension not in ALLOWED_EXTENSIONS:
        raise ImageError(f"unsupported file extension '{extension}', allowed: jpg, jpeg, png")

    if content_type and content_type.split(";")[0].strip().lower() not in ALLOWED_CONTENT_TYPES:
        raise ImageError(f"unsupported content type '{content_type}', allowed: image/jpeg, image/png")


def decode_image(raw: bytes) -> np.ndarray:
    """Decode uploaded bytes into a BGR numpy array, entirely in memory."""
    if not raw:
        raise ImageError("uploaded file is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ImageError(f"image is larger than the {MAX_IMAGE_BYTES} byte limit")

    try:
        with Image.open(io.BytesIO(raw)) as image:
            image_format = image.format
            if image_format not in ALLOWED_IMAGE_FORMATS:
                raise ImageError(f"unsupported image format '{image_format}', allowed: JPEG, PNG")
            rgb = image.convert("RGB")
            array = np.asarray(rgb)
    except ImageError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise ImageError(f"failed to decode image: {exc}") from exc

    # PaddleOCR expects BGR channel order, the OpenCV convention.
    return array[:, :, ::-1].copy()


def _extract_texts(page) -> tuple[list, list]:
    """Pull (rec_texts, rec_scores) out of one PaddleOCR result page."""
    data = page
    try:
        # Some pipeline versions nest the payload under a "res" key.
        if "res" in data and isinstance(data["res"], dict):
            data = data["res"]
        return list(data.get("rec_texts", [])), list(data.get("rec_scores", []))
    except (TypeError, KeyError, AttributeError):
        logger.warning("unexpected OCR result shape: %s", type(page))
        return [], []


def run_ocr(engine, image: np.ndarray) -> list[dict]:
    """Run OCR and flatten the result into [{"text": ..., "confidence": ...}]."""
    with _predict_slots:
        pages = engine.predict(image)

    items: list[dict] = []
    for page in pages or []:
        texts, scores = _extract_texts(page)
        for index, text in enumerate(texts):
            confidence = float(scores[index]) if index < len(scores) else 0.0
            items.append({"text": str(text), "confidence": round(confidence, 4)})
    return items
