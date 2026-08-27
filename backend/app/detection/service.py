"""Image preprocessing, inference, filtering and response assembly."""

from __future__ import annotations

import time
from functools import lru_cache

import cv2
import numpy as np

from app.config import Settings, settings
from app.context import assign_subject_context
from app.schemas import (
    AnalysisResponse,
    BoundingBox,
    CenterPoint,
    DetectionResult,
    DetectionSummary,
    ImageDetails,
    NormalizedBoundingBox,
    RawDetection,
)
from app.utils.image_validation import DecodedImage

from .base import CompositeDetector, Detector
from .opencv_detector import OpenCVFaceDetector, OpenCVPersonDetector
from .yolo_detector import YoloDetector


class DetectionService:
    def __init__(self, detector: Detector, app_settings: Settings = settings) -> None:
        self._detector = detector
        self._settings = app_settings

    def _prepare_image(self, image: np.ndarray) -> tuple[np.ndarray, float]:
        height, width = image.shape[:2]
        max_dimension = max(height, width)
        if max_dimension <= self._settings.inference_max_dimension:
            return image, 1.0
        scale = self._settings.inference_max_dimension / max_dimension
        resized = cv2.resize(
            image,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        return resized, scale

    def _passes_threshold(self, detection: RawDetection) -> bool:
        threshold = self._settings.default_confidence_threshold
        if detection.category == "face":
            threshold = self._settings.face_confidence_threshold
        elif detection.category == "person":
            threshold = self._settings.person_confidence_threshold
        return detection.confidence >= threshold

    @staticmethod
    def _restore_coordinates(detection: RawDetection, scale: float) -> RawDetection:
        if scale == 1.0:
            return detection
        return RawDetection(
            category=detection.category,
            confidence=detection.confidence,
            x=round(detection.x / scale),
            y=round(detection.y / scale),
            width=round(detection.width / scale),
            height=round(detection.height / scale),
            source=detection.source,
        )

    @staticmethod
    def _to_result(
        detection: RawDetection,
        index: int,
        image_width: int,
        image_height: int,
    ) -> DetectionResult:
        clipped = detection.clipped(image_width, image_height)
        relative_area = (clipped.width * clipped.height) / (image_width * image_height)
        normalized = NormalizedBoundingBox(
            x=clipped.x / image_width,
            y=clipped.y / image_height,
            width=clipped.width / image_width,
            height=clipped.height / image_height,
        )
        if clipped.category == "face":
            category = "face"
            label = "Main subject" if clipped.is_main_subject else "Background face"
        else:
            category = "person" if clipped.is_main_subject else "background_person"
            label = "Main subject person" if clipped.is_main_subject else "Background person"
        return DetectionResult(
            id=f"det_{clipped.category}_{index}",
            category=category,
            label=label,
            confidence=round(clipped.confidence, 6),
            boundingBox=BoundingBox(
                x=clipped.x,
                y=clipped.y,
                width=clipped.width,
                height=clipped.height,
                x2=clipped.x2,
                y2=clipped.y2,
            ),
            normalizedBoundingBox=normalized,
            center=CenterPoint(
                x=(clipped.x + clipped.width / 2) / image_width,
                y=(clipped.y + clipped.height / 2) / image_height,
            ),
            relativeArea=round(relative_area, 6),
            isMainSubject=clipped.is_main_subject,
            subjectScore=round(clipped.subject_score, 6) if clipped.subject_score is not None else None,
            recommendedAnonymization=not clipped.is_main_subject,
            explanation=clipped.explanation,
            source=clipped.source,
        )

    def analyze(self, decoded: DecodedImage) -> AnalysisResponse:
        started = time.perf_counter()
        inference_image, scale = self._prepare_image(decoded.pixels_bgr)
        predictions = self._detector.detect(inference_image)
        filtered = [
            self._restore_coordinates(detection, scale).clipped(decoded.width, decoded.height)
            for detection in predictions
            if self._passes_threshold(detection)
        ]
        filtered = [detection for detection in filtered if detection.width > 0 and detection.height > 0]
        contextualized = assign_subject_context(
            filtered,
            decoded.width,
            decoded.height,
            self._settings,
        )
        results = [
            self._to_result(detection, index, decoded.width, decoded.height)
            for index, detection in enumerate(contextualized, start=1)
        ]
        face_results = [result for result in results if result.category == "face"]
        person_results = [result for result in results if result.category in {"person", "background_person"}]
        elapsed_ms = max(0, round((time.perf_counter() - started) * 1000))
        return AnalysisResponse(
            image=ImageDetails(width=decoded.width, height=decoded.height, format=decoded.format),
            detections=results,
            summary=DetectionSummary(
                totalObjects=len(results),
                faces=len(face_results),
                people=len(person_results),
                backgroundFaces=sum(not result.isMainSubject for result in face_results),
                mainSubjectDetected=any(result.isMainSubject for result in results),
            ),
            processingTimeMs=elapsed_ms,
        )


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    face_detector = OpenCVFaceDetector()
    if settings.detector_backend == "opencv":
        person_detector: Detector = OpenCVPersonDetector()
    elif settings.detector_backend == "yolo":
        person_detector = YoloDetector(settings.yolo_weights_path)
    else:
        raise RuntimeError("DETECTOR_BACKEND must be 'opencv' or 'yolo'")
    return DetectionService(CompositeDetector([face_detector, person_detector]))
