from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.detection.service import DetectionService, get_detection_service
from app.schemas import RawDetection
from app.utils.image_validation import DecodedImage
from main import app


class StaticDetector:
    def __init__(self, detections: list[RawDetection]) -> None:
        self.detections = detections

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        return self.detections


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
    def test_response_contains_real_detector_fields_and_clipped_box(self) -> None:
        detector = StaticDetector([RawDetection(0, "person", 0.9234567, -4, 10, 130, 90)])
        response = DetectionService(detector, test_settings()).analyze(
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
        response = DetectionService(detector, test_settings()).analyze(
            DecodedImage(np.zeros((80, 120, 3), dtype=np.uint8), 120, 80, "PNG"), "empty.png"
        )
        self.assertEqual(response.analysis.detections, [])
        self.assertEqual(response.analysis.detection_count, 0)


class AnalysisApiTests(unittest.TestCase):
    def setUp(self) -> None:
        detector = StaticDetector([RawDetection(0, "person", 0.98, 30, 10, 90, 70)])
        app.dependency_overrides[get_detection_service] = lambda: DetectionService(detector, test_settings())
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
