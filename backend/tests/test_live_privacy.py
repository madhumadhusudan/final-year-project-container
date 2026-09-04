from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.detection.face_detector import RawFace
from app.detection.service import DetectionService, get_detection_service
from app.schemas import RawDetection
from main import app


class EmptyObjectDetector:
    def detect(self, _image: np.ndarray) -> list[RawDetection]:
        return []


class StaticFaceDetector:
    name = "test_face_detector"

    def __init__(self, faces=None) -> None:
        self.faces = faces or []

    def detect(self, _image: np.ndarray):
        return self.faces


def test_settings() -> Settings:
    return Settings(yolo_weights_path=Path("unused.pt"))


def encoded_png(width: int = 120, height: int = 80) -> bytes:
    image = np.zeros((height, width, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("test encoding failed")
    return encoded.tobytes()


class LivePrivacyApiTests(unittest.TestCase):
    def setUp(self) -> None:
        faces = [RawFace(0.96, 15, 10, 65, 65), RawFace(0.88, 85, 20, 112, 52)]
        self.service = DetectionService(EmptyObjectDetector(), StaticFaceDetector(faces), test_settings())
        app.dependency_overrides[get_detection_service] = lambda: self.service
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()

    def request(self, **fields):
        data = {
            "frame_id": "7", "captured_at_ms": "1720000000000",
            "modules": "faces", "preserve_main_subject": "true",
            **fields,
        }
        return self.client.post(
            "/analyze-frame", data=data,
            files={"image": ("frame.png", encoded_png(), "image/png")},
        )

    def test_frame_response_preserves_ids_schema_and_safe_metadata(self) -> None:
        response = self.request()
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["frame_id"], 7)
        self.assertEqual(payload["captured_at_ms"], 1720000000000)
        self.assertEqual(payload["image"]["width"], 120)
        self.assertEqual(payload["modules"]["faces"]["status"], "completed")
        self.assertEqual(payload["modules"]["faces"]["detection_count"], 2)
        self.assertEqual(payload["regions"][0]["detection_id"], "face_1")
        self.assertIn(payload["regions"][0]["role"], {"main_subject", "background_face", "unclassified"})
        self.assertNotIn("pixels", response.text)
        self.assertNotIn("raw_text", response.text)

    def test_preserve_off_marks_every_face_protectable(self) -> None:
        response = self.request(preserve_main_subject="false")
        self.assertEqual(response.status_code, 200)
        self.assertTrue(all(item["role"] == "background_face" for item in response.json()["regions"]))

    def test_invalid_frame_id_and_module_are_rejected(self) -> None:
        self.assertEqual(self.request(frame_id="0").status_code, 422)
        invalid = self.request(modules="faces,cloud_detector")
        self.assertEqual(invalid.status_code, 422)
        self.assertIn("supported live modules", invalid.json()["detail"])

    def test_oversized_analysis_frame_is_rejected(self) -> None:
        response = self.client.post(
            "/analyze-frame",
            data={"frame_id": "1", "captured_at_ms": "1", "modules": "faces"},
            files={"image": ("frame.png", encoded_png(2000, 1300), "image/png")},
        )
        self.assertEqual(response.status_code, 413)

    def test_missing_optional_modules_are_honestly_unavailable(self) -> None:
        response = self.request(modules="faces,plates,documents")
        payload = response.json()
        self.assertEqual(payload["modules"]["plates"]["status"], "unavailable")
        self.assertEqual(payload["modules"]["documents"]["status"], "unavailable")

    def test_capabilities_and_no_disk_persistence(self) -> None:
        watched = [Path("uploads"), Path("outputs")]
        before = {str(path): sorted(item.name for item in path.iterdir()) for path in watched}
        capability_response = self.client.get("/live/capabilities")
        self.assertEqual(capability_response.status_code, 200)
        self.assertTrue(capability_response.json()["local_processing_only"])
        self.assertEqual(capability_response.json()["frame_storage"], "none")
        self.assertEqual(self.request().status_code, 200)
        after = {str(path): sorted(item.name for item in path.iterdir()) for path in watched}
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
