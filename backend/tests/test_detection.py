from __future__ import annotations

import inspect
import unittest
from dataclasses import replace

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import settings
from app.context import assign_subject_context
from app.detection.base import Detector
from app.detection.service import DetectionService, get_detection_service
from app.schemas import RawDetection
from app.utils.image_validation import DecodedImage
from main import app


class StaticDetector(Detector):
    name = "test_static"

    def __init__(self, detections: list[RawDetection]) -> None:
        self.detections = detections

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        return self.detections


class FailingService:
    def analyze(self, decoded: DecodedImage) -> None:
        raise RuntimeError("internal model detail")


def encoded_png(width: int = 120, height: int = 80) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("test image encoding failed")
    return encoded.tobytes()


class ContextTests(unittest.TestCase):
    def test_one_large_central_face_is_main_subject(self) -> None:
        face = RawDetection("face", 0.96, 300, 250, 400, 400, "test")
        contextualized = assign_subject_context([face], 1000, 1000, settings)

        self.assertTrue(contextualized[0].is_main_subject)
        self.assertGreaterEqual(contextualized[0].subject_score or 0, settings.main_subject_score_threshold)
        self.assertIn("16.0%", contextualized[0].explanation)

    def test_multiple_faces_are_ranked_by_size_position_and_confidence(self) -> None:
        central = RawDetection("face", 0.92, 350, 300, 300, 300, "test")
        edge = RawDetection("face", 0.99, 10, 10, 290, 290, "test")
        results = assign_subject_context([edge, central], 1000, 1000, settings)

        self.assertFalse(results[0].is_main_subject)
        self.assertTrue(results[1].is_main_subject)

    def test_small_face_is_background_with_calculated_explanation(self) -> None:
        face = RawDetection("face", 0.99, 480, 480, 40, 40, "test")
        result = assign_subject_context([face], 1000, 1000, settings)[0]

        self.assertFalse(result.is_main_subject)
        self.assertIn("below", result.explanation)


class DetectionServiceTests(unittest.TestCase):
    def test_confidence_filtering_and_coordinate_conversion(self) -> None:
        app_settings = replace(
            settings,
            inference_max_dimension=500,
            face_confidence_threshold=0.70,
            main_subject_area_threshold=0.02,
        )
        predictions = [
            RawDetection("face", 0.95, 100, 75, 200, 200, "test"),
            RawDetection("person", 0.20, 0, 0, 10, 10, "test"),
        ]
        service = DetectionService(StaticDetector(predictions), app_settings)
        decoded = DecodedImage(np.zeros((800, 1000, 3), dtype=np.uint8), 1000, 800, "PNG")
        response = service.analyze(decoded)

        self.assertEqual(response.summary.totalObjects, 1)
        self.assertEqual(response.detections[0].boundingBox.x, 200)
        self.assertEqual(response.detections[0].boundingBox.width, 400)
        self.assertAlmostEqual(response.detections[0].normalizedBoundingBox.x, 0.2)
        self.assertAlmostEqual(response.detections[0].normalizedBoundingBox.width, 0.4)

    def test_no_detections_is_safe(self) -> None:
        service = DetectionService(StaticDetector([]), settings)
        decoded = DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG")
        response = service.analyze(decoded)

        self.assertEqual(response.detections, [])
        self.assertEqual(response.summary.totalObjects, 0)
        self.assertFalse(response.summary.mainSubjectDetected)


class AnalysisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        detector = StaticDetector([RawDetection("face", 0.98, 30, 10, 60, 60, "test")])
        app.dependency_overrides[get_detection_service] = lambda: DetectionService(
            detector,
            replace(settings, main_subject_area_threshold=0.05),
        )
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def test_valid_image_upload_returns_structured_detection(self) -> None:
        response = self.client.post(
            "/api/v1/analyze/image",
            files={"image": ("portrait.png", encoded_png(), "image/png")},
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["image"]["width"], 120)
        self.assertEqual(payload["summary"]["faces"], 1)
        self.assertIn("normalizedBoundingBox", payload["detections"][0])
        self.assertIn("recommendedAnonymization", payload["detections"][0])

    def test_invalid_and_mismatched_images_are_rejected(self) -> None:
        invalid = self.client.post(
            "/api/v1/analyze/image",
            files={"image": ("fake.jpg", b"not-an-image", "image/jpeg")},
        )
        mismatch = self.client.post(
            "/api/v1/analyze/image",
            files={"image": ("fake.jpg", encoded_png(), "image/jpeg")},
        )

        self.assertEqual(invalid.status_code, 400)
        self.assertEqual(mismatch.status_code, 400)

    def test_oversized_image_is_rejected(self) -> None:
        content = b"\xff\xd8\xff" + bytes(settings.max_image_bytes)
        response = self.client.post(
            "/api/v1/analyze/image",
            files={"image": ("large.jpg", content, "image/jpeg")},
        )
        self.assertEqual(response.status_code, 413)

    def test_model_failure_returns_safe_message(self) -> None:
        app.dependency_overrides[get_detection_service] = lambda: FailingService()
        response = self.client.post(
            "/api/v1/analyze/image",
            files={"image": ("portrait.png", encoded_png(), "image/png")},
        )

        self.assertEqual(response.status_code, 503)
        self.assertNotIn("internal model detail", response.text)


class DirectValidationTests(unittest.TestCase):
    def test_existing_async_endpoint_helpers_still_import(self) -> None:
        # A small smoke guard for projects that run tests without pytest plugins.
        self.assertTrue(inspect.iscoroutinefunction(app.router.routes[-1].endpoint))


if __name__ == "__main__":
    unittest.main()
