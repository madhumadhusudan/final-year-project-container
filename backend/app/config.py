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
    license_plate_model_path: Path = Path(
        os.getenv("LICENSE_PLATE_MODEL_PATH", str(BACKEND_DIR / "models" / "license_plate_detector.pt"))
    )
    card_model_path: Path = Path(
        os.getenv("CARD_MODEL_PATH", str(BACKEND_DIR / "models" / "card_detector.pt"))
    )
    privacy_object_confidence_threshold: float = float(os.getenv("PRIVACY_OBJECT_CONFIDENCE_THRESHOLD", "0.35"))
    license_plate_model_source: str = os.getenv("LICENSE_PLATE_MODEL_SOURCE", "")
    card_model_source: str = os.getenv("CARD_MODEL_SOURCE", "")
    subject_size_weight: float = 0.45
    subject_center_weight: float = 0.25
    subject_confidence_weight: float = 0.15
    subject_person_weight: float = 0.15
    subject_single_threshold: float = 0.45
    subject_score_threshold: float = 0.60
    subject_ambiguity_margin: float = 0.12
    ocr_languages: tuple[str, ...] = tuple(
        language.strip() for language in os.getenv("OCR_LANGUAGES", "en").split(",") if language.strip()
    )
    ocr_model_directory: Path = Path(
        os.getenv("OCR_MODEL_DIRECTORY", str(BACKEND_DIR / "models" / "easyocr"))
    )


settings = Settings()

ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})
