"""One-time model lifecycle, inference, and response assembly."""

from __future__ import annotations

import time
from functools import lru_cache

from app.config import Settings, settings
from app.schemas import (
    AnalysisDetails, AnalysisResponse, BoundingBox, CardDetectionDetails, DetectionResult,
    FaceDetectionDetails, FaceResult, ImageDetails, LicensePlateDetectionDetails,
    ObjectDetectionDetails, PerformanceDetails, Point, PrivacyObjectResult,
)
from app.context.main_subject_analyzer import MainSubjectAnalyzer
from app.utils.image_validation import DecodedImage

from .yolo_detector import YoloDetector
from .face_detector import FaceDetector, UnavailableFaceDetector
from .card_detector import CardDetector
from .license_plate_detector import LicensePlateDetector


class DetectionService:
    def __init__(self, detector: YoloDetector, face_detector: FaceDetector, app_settings: Settings = settings,
                 plate_detector: LicensePlateDetector | None = None, card_detector: CardDetector | None = None) -> None:
        self._detector = detector
        self._face_detector = face_detector
        self._settings = app_settings
        self._plate_detector = plate_detector
        self._card_detector = card_detector
        self._subject_analyzer = MainSubjectAnalyzer(app_settings)

    @staticmethod
    def _privacy_results(raw_results, image_width: int, image_height: int) -> list[PrivacyObjectResult]:
        results = []
        for raw in raw_results:
            x1, y1 = max(0, min(raw.x1, image_width)), max(0, min(raw.y1, image_height))
            x2, y2 = max(x1, min(raw.x2, image_width)), max(y1, min(raw.y2, image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            results.append(PrivacyObjectResult(
                id=len(results) + 1, class_name=raw.class_name, confidence=round(raw.confidence, 6),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            ))
        return results

    @staticmethod
    def _match_plates_to_vehicles(plates: list[PrivacyObjectResult], detections: list[DetectionResult]) -> None:
        vehicles = [item for item in detections if item.class_name in {"car", "truck", "bus", "motorcycle"}]
        for plate in plates:
            center_x = (plate.bounding_box.x1 + plate.bounding_box.x2) / 2
            center_y = (plate.bounding_box.y1 + plate.bounding_box.y2) / 2
            containing = [vehicle for vehicle in vehicles if
                          vehicle.bounding_box.x1 <= center_x <= vehicle.bounding_box.x2 and
                          vehicle.bounding_box.y1 <= center_y <= vehicle.bounding_box.y2]
            if containing:
                plate.matched_vehicle_id = min(
                    containing,
                    key=lambda vehicle: (vehicle.bounding_box.x2 - vehicle.bounding_box.x1)
                    * (vehicle.bounding_box.y2 - vehicle.bounding_box.y1),
                ).id

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
        plate_started = time.perf_counter()
        plate_error = None
        if self._plate_detector is None:
            raw_plates = []
            plate_status = "unavailable"
        else:
            try:
                raw_plates = self._plate_detector.detect(decoded.pixels_bgr)
                plate_status = "completed"
            except RuntimeError:
                raw_plates, plate_status, plate_error = [], "error", "License plate detection failed locally."
        plate_ms = max(0, round((time.perf_counter() - plate_started) * 1000))
        plates = self._privacy_results(raw_plates, decoded.width, decoded.height)
        self._match_plates_to_vehicles(plates, detections)

        card_started = time.perf_counter()
        card_error = None
        if self._card_detector is None:
            raw_cards = []
            card_status = "unavailable"
        else:
            try:
                raw_cards = self._card_detector.detect(decoded.pixels_bgr)
                card_status = "completed"
            except RuntimeError:
                raw_cards, card_status, card_error = [], "error", "Payment card detection failed locally."
        card_ms = max(0, round((time.perf_counter() - card_started) * 1000))
        cards = self._privacy_results(raw_cards, decoded.width, decoded.height)

        context_started = time.perf_counter()
        main_subject = self._subject_analyzer.analyze(faces, detections)
        context_ms = max(0, round((time.perf_counter() - context_started) * 1000))
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
                main_subject=main_subject,
                license_plate_detection=LicensePlateDetectionDetails(
                    status=plate_status, detector=getattr(self._plate_detector, "name", "model_required"),
                    plate_count=len(plates), plates=plates,
                    message=plate_error or ("Dedicated license plate model required." if plate_status == "unavailable" else None),
                    model_source=self._settings.license_plate_model_source or None,
                    supported_classes=sorted(LicensePlateDetector.supported_classes),
                ),
                card_detection=CardDetectionDetails(
                    status=card_status, detector=getattr(self._card_detector, "name", "model_required"),
                    card_count=len(cards), cards=cards,
                    message=card_error or ("Dedicated payment card model required." if card_status == "unavailable" else None),
                    model_source=self._settings.card_model_source or None,
                    supported_classes=sorted(CardDetector.supported_classes),
                ),
            ),
            performance=PerformanceDetails(
                inference_time_ms=object_ms, object_detection_ms=object_ms,
                face_detection_ms=face_ms, total_analysis_ms=total_ms,
                license_plate_detection_ms=plate_ms, card_detection_ms=card_ms,
                context_analysis_ms=context_ms,
            ),
        )


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    detector = YoloDetector(settings.yolo_weights_path, settings.yolo_confidence_threshold)
    try:
        face_detector = FaceDetector(settings.face_model_path, settings.face_confidence_threshold)
    except RuntimeError:
        face_detector = UnavailableFaceDetector()
    try:
        plate_detector = LicensePlateDetector(settings.license_plate_model_path, settings.privacy_object_confidence_threshold)
    except RuntimeError:
        plate_detector = None
    try:
        card_detector = CardDetector(settings.card_model_path, settings.privacy_object_confidence_threshold)
    except RuntimeError:
        card_detector = None
    return DetectionService(detector, face_detector, settings, plate_detector, card_detector)
