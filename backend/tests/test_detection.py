from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.detection.service import DetectionService, get_detection_service
from app.detection.face_detector import RawFace
from app.detection.face_detector import FaceDetector
from app.detection.privacy_object_detector import RawPrivacyObject
from app.ocr.ocr_service import RawOCRText
from app.schemas import RawDetection
from app.utils.image_validation import DecodedImage
from main import app


class StaticDetector:
    def __init__(self, detections: list[RawDetection]) -> None:
        self.detections = detections

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        return self.detections


class StaticFaceDetector:
    name = "test_face_detector"

    def __init__(self, faces: list[RawFace] | None = None, fails: bool = False) -> None:
        self.faces = faces or []
        self.fails = fails

    def detect(self, image_bgr: np.ndarray) -> list[RawFace]:
        if self.fails:
            raise RuntimeError("private face detector detail")
        return self.faces


class StaticPrivacyDetector:
    name = "test_dedicated_model"

    def __init__(self, results: list[RawPrivacyObject] | None = None, fails: bool = False) -> None:
        self.results = results or []
        self.fails = fails

    def detect(self, image_bgr: np.ndarray) -> list[RawPrivacyObject]:
        if self.fails:
            raise RuntimeError("private detector error")
        return self.results


class StaticOCRService:
    name = "test_ocr"

    def __init__(self, results: list[RawOCRText] | None = None, fails: bool = False) -> None:
        self.results = results or []
        self.fails = fails

    def extract_text(self, image_bgr: np.ndarray) -> list[RawOCRText]:
        if self.fails:
            raise RuntimeError("private OCR detail")
        return self.results


class FailingService:
    def analyze(self, decoded: DecodedImage, filename: str) -> None:
        raise RuntimeError("private model detail")


def test_settings() -> Settings:
    return Settings(yolo_weights_path=Path("unused.pt"))


def encoded_png(width: int = 120, height: int = 80) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("test image encoding failed")
    return encoded.tobytes()


class DetectionServiceTests(unittest.TestCase):
    def test_real_face_detector_model_initializes(self) -> None:
        detector = FaceDetector(test_settings().face_model_path, test_settings().face_confidence_threshold)
        self.assertEqual(detector.name, "opencv_yunet")

    def test_response_contains_real_detector_fields_and_clipped_box(self) -> None:
        detector = StaticDetector([RawDetection(0, "person", 0.9234567, -4, 10, 130, 90)])
        response = DetectionService(detector, StaticFaceDetector(), test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"),
            "portrait.png",
        )

        self.assertEqual(response.image.filename, "portrait.png")
        self.assertEqual(response.analysis.detection_count, 1)
        self.assertEqual(response.analysis.detections[0].class_name, "person")
        self.assertEqual(response.analysis.detections[0].bounding_box.model_dump(), {"x1": 0, "y1": 10, "x2": 120, "y2": 80})
        self.assertGreaterEqual(response.performance.inference_time_ms, 0)

    def test_zero_and_invalid_dimension_detections_are_safe(self) -> None:
        detector = StaticDetector([RawDetection(2, "car", 0.9, 30, 20, 10, 40)])
        response = DetectionService(detector, StaticFaceDetector(), test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "empty.png"
        )
        self.assertEqual(response.analysis.detections, [])
        self.assertEqual(response.analysis.detection_count, 0)

    def test_multiple_faces_include_day_six_context_measurements(self) -> None:
        faces = [RawFace(0.97, 20, 10, 60, 50), RawFace(0.88, 80, 20, 110, 50)]
        response = DetectionService(StaticDetector([]), StaticFaceDetector(faces), test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "group.png"
        )
        self.assertEqual(response.analysis.face_detection.face_count, 2)
        first = response.analysis.face_detection.faces[0]
        self.assertEqual(first.face_id, 1)
        self.assertEqual(first.area, 1600)
        self.assertAlmostEqual(first.area_ratio, 1 / 6)
        self.assertEqual(first.center.model_dump(), {"x": 40.0, "y": 30.0})
        self.assertEqual(first.normalized_center.model_dump(), {"x": 1 / 3, "y": 0.375})

    def test_face_failure_does_not_discard_object_results(self) -> None:
        detector = StaticDetector([RawDetection(2, "car", 0.9, 10, 10, 50, 50)])
        response = DetectionService(detector, StaticFaceDetector(fails=True), test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "car.png"
        )
        self.assertEqual(response.analysis.detection_count, 1)
        self.assertEqual(response.analysis.face_detection.status, "error")
        self.assertEqual(response.analysis.face_detection.face_count, 0)
        self.assertNotIn("private", response.analysis.face_detection.error)

    def test_clear_subject_matches_person_and_marks_background_face(self) -> None:
        objects = [RawDetection(0, "person", 0.95, 0, 0, 75, 80)]
        faces = [RawFace(0.98, 20, 10, 60, 55), RawFace(0.85, 92, 15, 112, 35)]
        response = DetectionService(StaticDetector(objects), StaticFaceDetector(faces), test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "foreground.png"
        )
        self.assertEqual(response.analysis.main_subject.status, "identified")
        self.assertEqual(response.analysis.main_subject.face_id, 1)
        self.assertEqual(response.analysis.main_subject.matched_person_id, 1)
        self.assertEqual(response.analysis.face_detection.faces[0].role, "main_subject")
        self.assertEqual(response.analysis.face_detection.faces[1].role, "background_face")

    def test_similarly_prominent_group_is_uncertain(self) -> None:
        faces = [RawFace(0.95, 10, 15, 40, 45), RawFace(0.95, 80, 15, 110, 45)]
        response = DetectionService(StaticDetector([]), StaticFaceDetector(faces), test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "group.png"
        )
        self.assertEqual(response.analysis.main_subject.status, "uncertain")
        self.assertTrue(all(face.role == "unclassified" for face in response.analysis.face_detection.faces))

    def test_missing_dedicated_models_are_not_reported_as_zero_success(self) -> None:
        response = DetectionService(StaticDetector([]), StaticFaceDetector(), test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "nature.png"
        )
        self.assertEqual(response.analysis.main_subject.status, "not_found")
        self.assertEqual(response.analysis.license_plate_detection.status, "unavailable")
        self.assertEqual(response.analysis.card_detection.status, "unavailable")
        self.assertIn("model required", response.analysis.card_detection.message.lower())

    def test_privacy_boxes_are_clipped_and_plate_matches_vehicle(self) -> None:
        objects = [RawDetection(2, "car", 0.9, 10, 10, 110, 70)]
        plate = StaticPrivacyDetector([RawPrivacyObject("license_plate", 0.92, -5, 45, 125, 65)])
        card = StaticPrivacyDetector([RawPrivacyObject("card", 0.88, 80, 20, 130, 60)])
        response = DetectionService(
            StaticDetector(objects), StaticFaceDetector(), test_settings(), plate, card
        ).analyze(DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "privacy.png")
        self.assertEqual(response.analysis.license_plate_detection.status, "completed")
        detected_plate = response.analysis.license_plate_detection.plates[0]
        self.assertEqual(detected_plate.bounding_box.model_dump(), {"x1": 0, "y1": 45, "x2": 120, "y2": 65})
        self.assertEqual(detected_plate.matched_vehicle_id, 1)
        self.assertEqual(response.analysis.card_detection.cards[0].bounding_box.x2, 120)

    def test_privacy_detector_failure_is_isolated(self) -> None:
        response = DetectionService(
            StaticDetector([]), StaticFaceDetector(), test_settings(), StaticPrivacyDetector(fails=True), None
        ).analyze(DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "image.png")
        self.assertEqual(response.analysis.license_plate_detection.status, "error")
        self.assertEqual(response.analysis.card_detection.status, "unavailable")
        self.assertNotIn("private", response.analysis.license_plate_detection.message)

    def test_ocr_and_sensitive_text_are_separate_and_boxes_are_clipped(self) -> None:
        ocr = StaticOCRService([RawOCRText("Call 9876543210", "Call 9876543210", 0.93, -5, 10, 130, 40)])
        response = DetectionService(
            StaticDetector([]), StaticFaceDetector(), test_settings(), None, None, ocr
        ).analyze(DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "text.png")
        self.assertEqual(response.analysis.ocr.status, "completed")
        self.assertEqual(response.analysis.ocr.text_count, 1)
        self.assertEqual(response.analysis.ocr.texts[0].raw_text, "Call 9876543210")
        sensitive = response.analysis.sensitive_text.items[0]
        self.assertEqual(sensitive.type, "phone_number")
        self.assertEqual(sensitive.masked_value, "******3210")
        self.assertEqual(sensitive.bounding_box.model_dump(), {"x1": 0, "y1": 10, "x2": 120, "y2": 40})
        self.assertGreater(response.analysis.privacy_risk.score, 0)
        self.assertEqual(response.analysis.privacy_risk.breakdown.sensitive_text, response.analysis.privacy_risk.score)

    def test_ocr_failure_is_not_reported_as_zero_success(self) -> None:
        response = DetectionService(
            StaticDetector([]), StaticFaceDetector(), test_settings(), None, None, StaticOCRService(fails=True)
        ).analyze(DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "text.png")
        self.assertEqual(response.analysis.ocr.status, "error")
        self.assertEqual(response.analysis.sensitive_text.status, "error")
        self.assertNotIn("private", response.analysis.ocr.message)


class AnalysisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        detector = StaticDetector([RawDetection(0, "person", 0.98, 30, 10, 90, 70)])
        app.dependency_overrides[get_detection_service] = lambda: DetectionService(detector, StaticFaceDetector(), test_settings())
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_valid_image_returns_day_four_contract(self) -> None:
        response = self.client.post("/analyze", files={"image": ("portrait.png", encoded_png(), "image/png")})
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "success")
        self.assertEqual(payload["image"]["width"], 120)
        self.assertEqual(payload["analysis"]["detection_count"], 1)
        detection = payload["analysis"]["detections"][0]
        self.assertEqual(detection["class_id"], 0)
        self.assertTrue(0 <= detection["confidence"] <= 1)
        self.assertEqual(set(detection["bounding_box"]), {"x1", "y1", "x2", "y2"})
        self.assertEqual(payload["analysis"]["face_detection"]["face_count"], 0)
        self.assertIn("total_analysis_ms", payload["performance"])
        self.assertEqual(payload["analysis"]["license_plate_detection"]["status"], "unavailable")
        self.assertEqual(payload["analysis"]["card_detection"]["status"], "unavailable")
        risk = payload["analysis"]["privacy_risk"]
        self.assertEqual(risk["score"], 0)
        self.assertEqual(risk["assessment"]["status"], "partial")
        self.assertIn("card_detection", risk["assessment"]["unavailable_modules"])

    def test_compatibility_endpoint_still_works(self) -> None:
        response = self.client.post("/api/v1/analyze/image", files={"image": ("image.png", encoded_png(), "image/png")})
        self.assertEqual(response.status_code, 200)

    def test_invalid_image_and_model_error_are_safe(self) -> None:
        invalid = self.client.post("/analyze", files={"image": ("fake.jpg", b"not-image", "image/jpeg")})
        self.assertEqual(invalid.status_code, 400)
        app.dependency_overrides[get_detection_service] = lambda: FailingService()
        failed = self.client.post("/analyze", files={"image": ("image.png", encoded_png(), "image/png")})
        self.assertEqual(failed.status_code, 503)
        self.assertNotIn("private model detail", failed.text)


if __name__ == "__main__":
    unittest.main()
