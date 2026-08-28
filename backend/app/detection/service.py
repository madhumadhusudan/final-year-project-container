"""One-time model lifecycle, inference, and response assembly."""

from __future__ import annotations

import time
from functools import lru_cache

from app.config import Settings, settings
from app.schemas import (
    AnalysisDetails, AnalysisResponse, BoundingBox, DetectionResult, FaceDetectionDetails,
    FaceResult, ImageDetails, ObjectDetectionDetails, PerformanceDetails, Point,
)
from app.utils.image_validation import DecodedImage

from .yolo_detector import YoloDetector
from .face_detector import FaceDetector, UnavailableFaceDetector


class DetectionService:
    def __init__(self, detector: YoloDetector, face_detector: FaceDetector, app_settings: Settings = settings) -> None:
        self._detector = detector
        self._face_detector = face_detector
        self._settings = app_settings

    def analyze(self, decoded: DecodedImage, filename: str) -> AnalysisResponse:
        total_started = time.perf_counter()
        started = time.perf_counter()
        predictions = self._detector.detect(decoded.pixels_bgr)
        object_ms = max(0, round((time.perf_counter() - started) * 1000))
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
        face_started = time.perf_counter()
        face_error = None
        try:
            raw_faces = self._face_detector.detect(decoded.pixels_bgr)
        except RuntimeError:
            raw_faces = []
            face_error = "Face detection was unavailable for this image."
        face_ms = max(0, round((time.perf_counter() - face_started) * 1000))
        image_area = decoded.width * decoded.height
        image_diagonal = (decoded.width ** 2 + decoded.height ** 2) ** 0.5
        faces = []
        for face in raw_faces:
            x1 = max(0, min(face.x1, decoded.width))
            y1 = max(0, min(face.y1, decoded.height))
            x2 = max(x1, min(face.x2, decoded.width))
            y2 = max(y1, min(face.y2, decoded.height))
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            distance = (((center_x - decoded.width / 2) ** 2 + (center_y - decoded.height / 2) ** 2) ** 0.5) / image_diagonal
            faces.append(FaceResult(
                face_id=len(faces) + 1,
                confidence=round(face.confidence, 6),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                width=width, height=height, area=width * height,
                area_ratio=round((width * height) / image_area, 8),
                center=Point(x=center_x, y=center_y),
                normalized_center=Point(x=center_x / decoded.width, y=center_y / decoded.height),
                distance_from_image_center=round(distance, 8),
            ))
        total_ms = max(0, round((time.perf_counter() - total_started) * 1000))
        objects = ObjectDetectionDetails(model=self._settings.yolo_model_name, detection_count=len(detections), detections=detections)
        return AnalysisResponse(
            image=ImageDetails(filename=filename, width=decoded.width, height=decoded.height, format=decoded.format),
            analysis=AnalysisDetails(
                model=self._settings.yolo_model_name,
                detection_count=len(detections),
                detections=detections,
                object_detection=objects,
                face_detection=FaceDetectionDetails(
                    status="error" if face_error else "completed", detector=self._face_detector.name,
                    face_count=len(faces), faces=faces, error=face_error,
                ),
            ),
            performance=PerformanceDetails(
                inference_time_ms=object_ms, object_detection_ms=object_ms,
                face_detection_ms=face_ms, total_analysis_ms=total_ms,
            ),
        )


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    detector = YoloDetector(settings.yolo_weights_path, settings.yolo_confidence_threshold)
    try:
        face_detector = FaceDetector(settings.face_model_path, settings.face_confidence_threshold)
    except RuntimeError:
        face_detector = UnavailableFaceDetector()
    return DetectionService(detector, face_detector)
