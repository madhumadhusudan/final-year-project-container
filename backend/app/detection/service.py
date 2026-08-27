"""One-time model lifecycle, inference, and response assembly."""

from __future__ import annotations

import time
from functools import lru_cache

from app.config import Settings, settings
from app.schemas import AnalysisDetails, AnalysisResponse, BoundingBox, DetectionResult, ImageDetails, PerformanceDetails
from app.utils.image_validation import DecodedImage

from .yolo_detector import YoloDetector


class DetectionService:
    def __init__(self, detector: YoloDetector, app_settings: Settings = settings) -> None:
        self._detector = detector
        self._settings = app_settings

    def analyze(self, decoded: DecodedImage, filename: str) -> AnalysisResponse:
        started = time.perf_counter()
        predictions = self._detector.detect(decoded.pixels_bgr)
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        valid_predictions = []
        for prediction in predictions:
            clipped = prediction.clipped(decoded.width, decoded.height)
            if clipped.width > 0 and clipped.height > 0:
                valid_predictions.append(clipped)

        detections = [
            DetectionResult(
                id=index,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=round(detection.confidence, 6),
                bounding_box=BoundingBox(x1=detection.x1, y1=detection.y1, x2=detection.x2, y2=detection.y2),
            )
            for index, detection in enumerate(valid_predictions, start=1)
        ]
        return AnalysisResponse(
            image=ImageDetails(filename=filename, width=decoded.width, height=decoded.height, format=decoded.format),
            analysis=AnalysisDetails(
                model=self._settings.yolo_model_name,
                detection_count=len(detections),
                detections=detections,
            ),
            performance=PerformanceDetails(inference_time_ms=elapsed_ms),
        )


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    detector = YoloDetector(settings.yolo_weights_path, settings.yolo_confidence_threshold)
    return DetectionService(detector)
