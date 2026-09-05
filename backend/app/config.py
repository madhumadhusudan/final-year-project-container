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


def _bounded_float_setting(name: str, default: str) -> float:
    raw_value = os.getenv(name, default)
    try:
        value = float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not 0 <= value <= 1:
        raise ValueError(f"{name} must be between 0 and 1")
    return value


def _image_size_setting(name: str, default: str) -> int:
    raw_value = os.getenv(name, default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if not 320 <= value <= 2048:
        raise ValueError(f"{name} must be between 320 and 2048")
    return value


def _positive_int_setting(name: str, default: str, minimum: int = 1) -> int:
    raw_value = os.getenv(name, default)
    try:
        value = int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer") from exc
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
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
    document_model_path: Path = Path(
        os.getenv("DOCUMENT_MODEL_PATH", str(BACKEND_DIR / "models" / "document_detector.pt"))
    )
    privacy_object_confidence_threshold: float = float(os.getenv("PRIVACY_OBJECT_CONFIDENCE_THRESHOLD", "0.35"))
    card_confidence_threshold: float = _bounded_float_setting("CARD_CONFIDENCE_THRESHOLD", "0.25")
    card_inference_image_size: int = _image_size_setting("CARD_INFERENCE_IMAGE_SIZE", "960")
    card_tile_size: int = _image_size_setting("CARD_TILE_SIZE", "640")
    card_tile_inference_image_size: int = _image_size_setting("CARD_TILE_INFERENCE_IMAGE_SIZE", "640")
    card_tile_overlap: float = _bounded_float_setting("CARD_TILE_OVERLAP", "0.40")
    document_confidence_threshold: float = _bounded_float_setting("DOCUMENT_CONFIDENCE_THRESHOLD", "0.35")
    document_inference_image_size: int = _image_size_setting("DOCUMENT_INFERENCE_IMAGE_SIZE", "960")
    license_plate_model_source: str = os.getenv("LICENSE_PLATE_MODEL_SOURCE", "")
    card_model_source: str = os.getenv(
        "CARD_MODEL_SOURCE", "https://huggingface.co/lambdaWalker/creditCardDetection"
    )
    document_model_source: str = os.getenv("DOCUMENT_MODEL_SOURCE", "")
    document_model_license: str = os.getenv("DOCUMENT_MODEL_LICENSE", "")
    app_environment: str = os.getenv("APP_ENV", "development").strip().lower()
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
    max_video_bytes: int = _positive_int_setting("MAX_VIDEO_BYTES", str(150 * 1024 * 1024))
    max_video_duration_seconds: int = _positive_int_setting("MAX_VIDEO_DURATION_SECONDS", "180")
    max_concurrent_video_jobs: int = _positive_int_setting("MAX_CONCURRENT_VIDEO_JOBS", "1")
    video_output_ttl_seconds: int = _positive_int_setting("VIDEO_OUTPUT_TTL_SECONDS", "3600", 60)
    video_track_expiry_frames: int = _positive_int_setting("VIDEO_TRACK_EXPIRY_FRAMES", "18")
    video_default_fps: int = _positive_int_setting("VIDEO_DEFAULT_FPS", "25")


settings = Settings()

ALLOWED_MIME_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
ALLOWED_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png", ".webp"})

VIDEO_MIME_TYPES = {
    ".mp4": frozenset({"video/mp4", "application/mp4"}),
    ".mov": frozenset({"video/quicktime"}),
    ".avi": frozenset({"video/x-msvideo", "video/avi"}),
    ".webm": frozenset({"video/webm"}),
}
ALLOWED_VIDEO_EXTENSIONS = frozenset(VIDEO_MIME_TYPES)
