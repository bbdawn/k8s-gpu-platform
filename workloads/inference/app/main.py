"""OCR inference API used as a Kubernetes GPU benchmark workload.

The point of this service is not OCR features, but a small, predictable
GPU workload that can be run unchanged on Full GPU / Time-slicing / MPS / MIG
and compared. Keep it simple.
"""

from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import ocr
from gpu import get_gpu_info, resolve_device

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger("ocr-api")

WARM_UP_ON_STARTUP = os.getenv("OCR_WARMUP", "true").lower() == "true"
STATIC_DIR = Path(__file__).resolve().parent / "static"


# --- response models --------------------------------------------------------


class HealthResponse(BaseModel):
    status: str


class GpuResponse(BaseModel):
    cuda_available: bool
    gpu_count: int
    gpu_name: str | None = None
    gpu_memory_total_mb: int | None = None
    compiled_with_cuda: bool
    paddle_version: str | None = None
    paddle_device: str | None = None
    device_setting: str
    ocr_device: str
    ocr_ready: bool
    error: str | None = None


class TextItem(BaseModel):
    text: str
    confidence: float


class OcrResponse(BaseModel):
    text: list[TextItem]
    count: int
    elapsed_ms: float


# --- application ------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the OCR model once, before the first request is served."""
    app.state.ocr_engine = None
    app.state.ocr_error = None
    try:
        app.state.ocr_engine = ocr.create_ocr_engine()
        if WARM_UP_ON_STARTUP:
            await run_in_threadpool(ocr.warm_up, app.state.ocr_engine)
    except Exception as exc:  # noqa: BLE001
        # Starting without an engine is deliberate: /health and /gpu stay
        # usable so the failure is visible instead of a crash loop.
        app.state.ocr_error = f"{type(exc).__name__}: {exc}"
        logger.error("OCR engine initialization failed: %s", app.state.ocr_error)

    yield

    app.state.ocr_engine = None


app = FastAPI(
    title="OCR Inference Workload",
    description="PaddleOCR inference API for Kubernetes GPU benchmarking.",
    version="0.1.0",
    lifespan=lifespan,
)


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    """Minimal upload page. The API itself is also usable via /docs."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check. Intentionally does not touch the GPU."""
    return HealthResponse(status="ok")


@app.get("/gpu", response_model=GpuResponse)
def gpu() -> GpuResponse:
    """Report what this process can see, whether or not CUDA is available."""
    info = get_gpu_info()
    return GpuResponse(
        **info,
        ocr_device=resolve_device(),
        ocr_ready=app.state.ocr_engine is not None,
    )


@app.post("/ocr", response_model=OcrResponse)
async def run_ocr(file: UploadFile = File(...)) -> OcrResponse:
    """Run OCR on an uploaded jpg/png image."""
    # Validate the request first, so a bad request is a clear 400 even when the
    # engine is missing (which is the normal case on a CPU dev machine).
    try:
        ocr.validate_upload(file.filename, file.content_type)
        raw = await file.read()
        image = ocr.decode_image(raw)
    except ocr.ImageError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        await file.close()

    engine = app.state.ocr_engine
    if engine is None:
        raise HTTPException(
            status_code=503,
            detail=f"OCR engine is not available: {app.state.ocr_error}",
        )

    started = time.perf_counter()
    try:
        # predict() is blocking GPU work, so keep it off the event loop.
        items = await run_in_threadpool(ocr.run_ocr, engine, image)
    except Exception as exc:  # noqa: BLE001
        logger.exception("OCR inference failed")
        raise HTTPException(
            status_code=500, detail=f"OCR inference failed: {type(exc).__name__}: {exc}"
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    logger.info("ocr done: %d texts in %.1f ms", len(items), elapsed_ms)
    return OcrResponse(
        text=[TextItem(**item) for item in items],
        count=len(items),
        elapsed_ms=round(elapsed_ms, 2),
    )
