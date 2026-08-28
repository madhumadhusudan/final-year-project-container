"""Central configuration for image validation and YOLO inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _confidence_setting() -> float:
    raw_value = os.getenv("YOLO_CONFIDENCE_THRESHOLD", "0.35")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("YOLO_CONFIDENCE_THRESHOLD must be a number") from exc
    if not 0 <= value <= 1:
        raise ValueError("YOLO_CONFIDENCE_THRESHOLD must be between 0 and 1")
    return value


def _face_confidence_setting() -> float:
    raw_value = os.getenv("FACE_CONFIDENCE_THRESHOLD", "0.75")
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError("FACE_CONFIDENCE_THRESHOLD must be a number") from exc
    if not 0 <= value <= 1:
        raise ValueError("FACE_CONFIDENCE_THRESHOLD must be between 0 and 1")
    return value


BACKEND_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
    yolo_confidence_threshold: float = _confidence_setting()
    yolo_model_name: str = os.getenv("YOLO_MODEL_NAME", "yolov8n")
    yolo_weights_path: Path = Path(
        os.getenv("YOLO_WEIGHTS_PATH", str(BACKEND_DIR / "models" / "yolov8n.pt"))
    )
    face_confidence_threshold: float = _face_confidence_setting()
    face_model_path: Path = Path(
        os.getenv("FACE_MODEL_PATH", str(BACKEND_DIR / "models" / "face_detection_yunet_2023mar.onnx"))
    )


settings = Settings()

ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
