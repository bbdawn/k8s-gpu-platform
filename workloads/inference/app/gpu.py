"""GPU / device helpers.

Every paddle call is guarded, because this workload must also start on a
CPU-only machine (developer laptop, CI) and simply report that CUDA is not
available instead of crashing.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# "auto" -> gpu:0 when CUDA is usable, otherwise cpu.
# Override with OCR_DEVICE=cpu / gpu:0 / gpu:1 (easy to move into a ConfigMap).
DEVICE_SETTING = os.getenv("OCR_DEVICE", "auto").strip().lower()


def _import_paddle():
    """Import paddle lazily so a missing/broken install does not kill the app."""
    try:
        import paddle

        return paddle
    except Exception as exc:  # noqa: BLE001 - any import failure is non-fatal here
        logger.warning("paddle import failed: %s", exc)
        return None


def cuda_is_usable() -> bool:
    """True only when paddle is built with CUDA and a device is actually visible."""
    paddle = _import_paddle()
    if paddle is None:
        return False
    try:
        if not paddle.is_compiled_with_cuda():
            return False
        return paddle.device.cuda.device_count() > 0
    except Exception as exc:  # noqa: BLE001
        logger.warning("CUDA probe failed: %s", exc)
        return False


def resolve_device() -> str:
    """Device string handed to PaddleOCR, e.g. "gpu:0" or "cpu"."""
    if DEVICE_SETTING != "auto":
        return DEVICE_SETTING
    return "gpu:0" if cuda_is_usable() else "cpu"


def get_gpu_info() -> dict:
    """Snapshot of the GPU state for the /gpu endpoint. Never raises."""
    info = {
        "cuda_available": False,
        "gpu_count": 0,
        "gpu_name": None,
        "gpu_memory_total_mb": None,
        "compiled_with_cuda": False,
        "paddle_version": None,
        "paddle_device": None,
        "device_setting": DEVICE_SETTING,
        "error": None,
    }

    paddle = _import_paddle()
    if paddle is None:
        info["error"] = "paddle is not installed or failed to import"
        return info

    try:
        info["paddle_version"] = paddle.__version__
        info["compiled_with_cuda"] = bool(paddle.is_compiled_with_cuda())
        # Device paddle currently runs on, e.g. "gpu:0" / "cpu".
        info["paddle_device"] = paddle.device.get_device()

        if info["compiled_with_cuda"]:
            count = paddle.device.cuda.device_count()
            info["gpu_count"] = count
            info["cuda_available"] = count > 0
            if count > 0:
                props = paddle.device.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["gpu_memory_total_mb"] = round(props.total_memory / 1024 / 1024)
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        logger.warning("failed to collect GPU info: %s", exc)

    return info
