from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import Settings
from app.detection.service import DetectionService, get_detection_service
from app.detection.code_detector import CodeDetector, RawCodeDetection
from app.detection.face_detector import RawFace
from app.detection.privacy_object_detector import RawPrivacyObject
from app.ocr.ocr_service import RawOCRText
from app.schemas import RawDetection
from app.video.models import VideoProcessSettings
from app.video.video_analyzer import VideoAnalyzer
from app.video.video_jobs import video_job_manager
from app.video.video_tracker import RegionTracker
from main import app


class EmptyDetector:
    def detect(self, _image):
        return []


class EmptyFaceDetector:
    name = "test_face_detector"

    def detect(self, _image):
        return []


class TwoFaceDetector:
    name = "synthetic_face_detector"

    def detect(self, _image):
        return [RawFace(.99, 28, 8, 76, 62), RawFace(.91, 4, 18, 21, 40)]


class StaticPlateDetector:
    name = "synthetic_plate_detector"

    def detect(self, _image):
        return [RawPrivacyObject("license_plate", .93, 58, 44, 88, 59)]


class StaticCodeDetector:
    qr_name = "synthetic_qr_detector"
    barcode_name = "unavailable"
    qr_available = True
    barcode_available = False
    supported_barcode_formats = ()

    def detect_qr(self, _image):
        return [RawCodeDetection(((8, 45), (25, 45), (25, 62), (8, 62)), 8, 45, 25, 62, "https://safe.example")]


class StaticOCR:
    name = "synthetic_ocr"

    def extract_text(self, _image):
        return [RawOCRText("person@example.com", "person@example.com", .96, 35, 65, 78, 76)]


def test_settings() -> Settings:
    return Settings(yolo_weights_path=Path("unused.pt"))


def synthetic_mp4(frame_count: int = 12, fps: float = 12.0) -> bytes:
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "safe.mp4"
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (96, 64))
        if not writer.isOpened():
            raise unittest.SkipTest("OpenCV MP4 writer is unavailable")
        for index in range(frame_count):
            frame = np.full((64, 96, 3), (index * 13) % 255, dtype=np.uint8)
            cv2.rectangle(frame, (10 + index, 15), (30 + index, 38), (255, 255, 255), -1)
            writer.write(frame)
        writer.release()
        return path.read_bytes()


class VideoTrackerTests(unittest.TestCase):
    def test_ids_persist_with_motion_and_expire_after_grace(self) -> None:
        tracker = RegionTracker(expiry_frames=3)
        first = tracker.update([{"category": "face", "box": (10, 10, 30, 30), "confidence": .9, "role": "main_subject"}], {"face"}, 0)
        second = tracker.update([{"category": "face", "box": (13, 10, 33, 30), "confidence": .9, "role": "main_subject"}], {"face"}, 2)
        self.assertEqual(first[0].track_id, "face_track_1")
        self.assertEqual(second[0].track_id, first[0].track_id)
        self.assertGreater(second[0].predicted_box(3)[0], 13)
        self.assertTrue(tracker.active(4))
        tracker.update([], {"face"}, 6)
        self.assertFalse(tracker.active(6))

    def test_sampled_faces_plate_qr_and_ocr_are_protected_between_detector_frames(self) -> None:
        service = DetectionService(
            EmptyDetector(), TwoFaceDetector(), test_settings(), plate_detector=StaticPlateDetector(),
            ocr_service=StaticOCR(), code_detector=StaticCodeDetector(),
        )
        settings = VideoProcessSettings(
            preserve_main_subject=True, protect_background_faces=True,
            protect_license_plates=True, protect_cards=False, protect_identity_documents=False,
            protect_qr_codes=True, protect_barcodes=False, protect_sensitive_text=True,
            anonymization_method="blackout", quality_profile="balanced",
        )
        analyzer = VideoAnalyzer(service, settings, expiry_frames=18)
        originals = []
        outputs = []
        for index in range(12):
            frame = np.full((80, 100, 3), 180, dtype=np.uint8)
            originals.append(frame.copy())
            outputs.append(analyzer.process_frame(frame, index))
        # The central main face stays clear; the smaller background face stays masked.
        self.assertTrue(np.array_equal(outputs[11][30, 50], originals[11][30, 50]))
        self.assertTrue(np.array_equal(outputs[11][25, 12], np.zeros(3, dtype=np.uint8)))
        # Plate, QR, and OCR tracks cover frame five between their scheduled checks.
        self.assertTrue(np.array_equal(outputs[5][50, 70], np.zeros(3, dtype=np.uint8)))
        self.assertTrue(np.array_equal(outputs[5][53, 16], np.zeros(3, dtype=np.uint8)))
        self.assertTrue(np.array_equal(outputs[5][70, 50], np.zeros(3, dtype=np.uint8)))
        self.assertEqual(analyzer.category_analyzed_frames["license_plate"], 2)
        self.assertEqual(analyzer.category_analyzed_frames["qr_code"], 2)
        self.assertEqual(analyzer.category_analyzed_frames["sensitive_text"], 1)
        self.assertEqual(analyzer.main_subject_track_id, "face_track_1")

    def test_all_three_methods_run_through_video_region_path(self) -> None:
        service = DetectionService(
            EmptyDetector(), EmptyFaceDetector(), test_settings(), plate_detector=StaticPlateDetector(),
        )
        rng = np.random.default_rng(14)
        original = rng.integers(0, 256, (80, 100, 3), dtype=np.uint8)
        for method in ("blur", "pixelate", "blackout"):
            analyzer = VideoAnalyzer(
                service,
                VideoProcessSettings(
                    preserve_main_subject=False, protect_background_faces=False,
                    protect_license_plates=True, protect_cards=False,
                    protect_identity_documents=False, protect_qr_codes=False,
                    protect_barcodes=False, protect_sensitive_text=False,
                    anonymization_method=method, strength="high", quality_profile="balanced",
                ),
                expiry_frames=18,
            )
            protected = analyzer.process_frame(original.copy(), 0)
            self.assertFalse(np.array_equal(protected[44:59, 58:88], original[44:59, 58:88]), method)

    def test_real_qr_detector_runs_through_video_path(self) -> None:
        encoded = cv2.QRCodeEncoder_create().encode("https://example.com/day14-safe")
        scaled = cv2.resize(encoded, None, fx=7, fy=7, interpolation=cv2.INTER_NEAREST)
        frame = np.full((scaled.shape[0] + 80, scaled.shape[1] + 80, 3), 255, dtype=np.uint8)
        frame[40:40 + scaled.shape[0], 40:40 + scaled.shape[1]] = cv2.cvtColor(scaled, cv2.COLOR_GRAY2BGR)
        detector = CodeDetector()
        if not detector.qr_available:
            self.skipTest("OpenCV QR detector unavailable")
        raw = detector.detect_qr(frame)
        self.assertTrue(raw)
        service = DetectionService(EmptyDetector(), EmptyFaceDetector(), test_settings(), code_detector=detector)
        analyzer = VideoAnalyzer(
            service,
            VideoProcessSettings(
                preserve_main_subject=False, protect_background_faces=False,
                protect_license_plates=False, protect_cards=False,
                protect_identity_documents=False, protect_qr_codes=True,
                protect_barcodes=False, protect_sensitive_text=False,
                anonymization_method="blackout", quality_profile="balanced",
            ),
            expiry_frames=18,
        )
        protected = analyzer.process_frame(frame.copy(), 0)
        item = raw[0]
        center = ((item.x1 + item.x2) // 2, (item.y1 + item.y2) // 2)
        self.assertTrue(np.array_equal(protected[center[1], center[0]], np.zeros(3, dtype=np.uint8)))
        self.assertEqual(analyzer.category_present_frames["qr_code"], 1)


class VideoApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.service = DetectionService(EmptyDetector(), EmptyFaceDetector(), test_settings())
        app.dependency_overrides[get_detection_service] = lambda: cls.service
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        app.dependency_overrides.clear()

    def test_invalid_container_is_rejected_and_not_retained(self) -> None:
        response = self.client.post("/video/upload", files={"video": ("fake.mp4", b"not-video", "video/mp4")})
        self.assertEqual(response.status_code, 400)

    def test_metadata_job_progress_result_and_source_cleanup(self) -> None:
        upload = self.client.post(
            "/video/upload", files={"video": ("synthetic.mp4", synthetic_mp4(), "video/mp4")},
        )
        self.assertEqual(upload.status_code, 200, upload.text)
        payload = upload.json()
        self.assertEqual(payload["metadata"]["width"], 96)
        self.assertEqual(payload["metadata"]["height"], 64)
        self.assertEqual(payload["metadata"]["frame_count"], 12)
        self.assertAlmostEqual(payload["metadata"]["duration_seconds"], 1.0, places=1)
        settings = {
            "preserve_main_subject": False, "protect_background_faces": False,
            "protect_license_plates": False, "protect_cards": False,
            "protect_identity_documents": False, "protect_qr_codes": False,
            "protect_barcodes": False, "protect_sensitive_text": False,
            "anonymization_method": "blur", "strength": "medium", "quality_profile": "balanced",
        }
        started = self.client.post("/video/process", json={"upload_id": payload["upload_id"], "settings": settings})
        self.assertEqual(started.status_code, 202, started.text)
        job_id = started.json()["job_id"]
        status_payload = None
        for _ in range(100):
            status_payload = self.client.get(f"/video/status/{job_id}").json()
            if status_payload["state"] not in {"queued", "processing"}:
                break
            time.sleep(.02)
        self.assertEqual(status_payload["state"], "completed", status_payload)
        self.assertEqual(status_payload["progress"], 100)
        self.assertEqual(status_payload["summary"]["total_processed_frames"], 12)
        self.assertEqual(status_payload["summary"]["resolution"], "96x64")
        self.assertIn("audio", status_payload["summary"]["audio_message"].lower())
        result = self.client.get(f"/video/result/{job_id}")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.headers["content-type"], "video/mp4")
        self.assertGreater(len(result.content), 100)
        video_job_manager._jobs[job_id].output_path.unlink(missing_ok=True)

    def test_upload_can_be_explicitly_discarded(self) -> None:
        upload = self.client.post(
            "/video/upload", files={"video": ("discard.mp4", synthetic_mp4(2), "video/mp4")},
        ).json()
        self.assertEqual(self.client.delete(f"/video/upload/{upload['upload_id']}").status_code, 204)
        missing = self.client.post("/video/process", json={"upload_id": upload["upload_id"]})
        self.assertEqual(missing.status_code, 404)

    def test_cancel_marks_job_and_worker_cleans_temporary_files(self) -> None:
        class SlowDetector(EmptyDetector):
            def detect(self, image):
                time.sleep(.08)
                return super().detect(image)

        slow_service = DetectionService(SlowDetector(), EmptyFaceDetector(), test_settings())
        app.dependency_overrides[get_detection_service] = lambda: slow_service
        try:
            upload = self.client.post(
                "/video/upload", files={"video": ("cancel.mp4", synthetic_mp4(80), "video/mp4")},
            ).json()
            settings = {
                "preserve_main_subject": False, "protect_background_faces": False,
                "protect_license_plates": False, "protect_cards": False,
                "protect_identity_documents": False, "protect_qr_codes": False,
                "protect_barcodes": False, "protect_sensitive_text": False,
                "quality_profile": "accuracy",
            }
            started = self.client.post("/video/process", json={"upload_id": upload["upload_id"], "settings": settings}).json()
            cancelled = self.client.delete(f"/video/cancel/{started['job_id']}")
            self.assertEqual(cancelled.status_code, 200)
            self.assertEqual(cancelled.json()["state"], "cancelled")
            job = video_job_manager._jobs[started["job_id"]]
            for _ in range(100):
                if not job.source.path.exists() and not job.output_path.exists():
                    break
                time.sleep(.02)
            self.assertFalse(job.source.path.exists())
            self.assertFalse(job.output_path.exists())
            self.assertEqual(self.client.get(f"/video/result/{started['job_id']}").status_code, 409)
        finally:
            app.dependency_overrides[get_detection_service] = lambda: self.service


if __name__ == "__main__":
    unittest.main()
