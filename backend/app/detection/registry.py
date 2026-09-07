"""One-time detector registry with lazy, cached optional models."""

from __future__ import annotations

import threading
from collections.abc import Callable

import cv2

from app.config import Settings
from app.ocr.ocr_service import OCRService

from .card_detector import CardDetector
from .code_detector import CodeDetector
from .document_detector import DocumentDetector
from .face_detector import FaceDetector, UnavailableFaceDetector
from .license_plate_detector import LicensePlateDetector
from .yolo_detector import YoloDetector


def detected_device() -> str:
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class LazyDetector:
    def __init__(self, factory: Callable, name: str) -> None:
        self._factory = factory
        self._instance = None
        self._lock = threading.Lock()
        self.name = name

    def _get(self):
        if self._instance is None:
            with self._lock:
                if self._instance is None:
                    self._instance = self._factory()
        return self._instance

    def detect(self, image):
        return self._get().detect(image)

    def detect_with_diagnostics(self, image):
        return self._get().detect_with_diagnostics(image)

    def __getattr__(self, name):
        return getattr(self._get(), name)


class LazyOCRService:
    name = "easyocr_lazy"

    def __init__(self, factory: Callable) -> None:
        self._resource = LazyDetector(factory, self.name)

    def extract_text(self, image):
        return self._resource._get().extract_text(image)


class LazyCodeDetector:
    qr_name = "opencv_qrcode_detector"
    barcode_name = "opencv_barcode_detector"
    supported_barcode_formats = CodeDetector.supported_barcode_formats
    supports_exhaustive_qr = True
    qr_available = hasattr(cv2, "QRCodeDetector")
    barcode_available = hasattr(cv2, "barcode_BarcodeDetector")

    def __init__(self) -> None:
        self._resource = LazyDetector(CodeDetector, "opencv_code_detector")
        self._inference_lock = threading.Lock()

    def detect_qr(self, image, exhaustive=False):
        with self._inference_lock:
            return self._resource._get().detect_qr(image, exhaustive=exhaustive)

    def detect_barcodes(self, image):
        with self._inference_lock:
            return self._resource._get().detect_barcodes(image)


class DetectorRegistry:
    """Build core models once and defer optional heavy models until requested."""

    def __init__(self, app_settings: Settings) -> None:
        self.settings = app_settings
        self.device = detected_device()
        self.yolo = YoloDetector(
            app_settings.yolo_weights_path, app_settings.yolo_confidence_threshold,
            device=self.device, max_detections=app_settings.yolo_max_detections,
        )
        try:
            self.face = FaceDetector(app_settings.face_model_path, app_settings.face_confidence_threshold)
        except RuntimeError:
            self.face = UnavailableFaceDetector()
        self.plate = LazyDetector(
            lambda: LicensePlateDetector(
                app_settings.license_plate_model_path, app_settings.privacy_object_confidence_threshold,
            ), app_settings.license_plate_model_path.name,
        ) if app_settings.license_plate_model_path.is_file() else None
        self.card = LazyDetector(
            lambda: CardDetector(
                app_settings.card_model_path, app_settings.card_confidence_threshold,
                app_settings.card_inference_image_size, app_settings.card_tile_size,
                app_settings.card_tile_overlap, app_settings.card_tile_inference_image_size,
            ), app_settings.card_model_path.name,
        ) if app_settings.card_model_path.is_file() else None
        self.document = LazyDetector(
            lambda: DocumentDetector(
                app_settings.document_model_path, app_settings.document_confidence_threshold,
                app_settings.document_inference_image_size,
            ), app_settings.document_model_path.name,
        ) if app_settings.document_model_path.is_file() else None
        self.ocr = LazyOCRService(
            lambda: OCRService(list(app_settings.ocr_languages), app_settings.ocr_model_directory),
        )
        self.code = LazyCodeDetector()
