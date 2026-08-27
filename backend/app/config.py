"""Central configuration for image validation and model inference."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _float_setting(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None:
        return default
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


@dataclass(frozen=True)
class Settings:
    max_image_bytes: int = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
    default_confidence_threshold: float = _float_setting("DEFAULT_CONFIDENCE_THRESHOLD", 0.55)
    face_confidence_threshold: float = _float_setting("FACE_CONFIDENCE_THRESHOLD", 0.62)
    person_confidence_threshold: float = _float_setting("PERSON_CONFIDENCE_THRESHOLD", 0.55)
    main_subject_area_threshold: float = _float_setting("MAIN_SUBJECT_AREA_THRESHOLD", 0.08)
    main_subject_score_threshold: float = _float_setting("MAIN_SUBJECT_SCORE_THRESHOLD", 0.55)
    detector_backend: str = os.getenv("DETECTOR_BACKEND", "opencv").strip().lower()
    yolo_weights_path: Path = Path(os.getenv("YOLO_WEIGHTS_PATH", "models/yolov8n.pt"))
    inference_max_dimension: int = int(os.getenv("INFERENCE_MAX_DIMENSION", "1600"))


settings = Settings()

ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

# Categories intentionally include future privacy classes without claiming support today.
SUPPORTED_CATEGORIES = frozenset({"face", "person", "background_person"})
FUTURE_CATEGORIES = frozenset(
    {
        "license_plate",
        "aadhaar_card",
        "pan_card",
        "passport",
        "driving_licence",
        "voter_id",
        "payment_card",
        "personal_document",
        "qr_code",
        "barcode",
        "mobile_screen",
        "laptop_screen",
        "sensitive_text",
    }
)
