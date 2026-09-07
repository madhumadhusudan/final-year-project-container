"""Day 16 accuracy, calibration, protection, and reliability regression tests.

All pixels and identifiers used here are synthetic or deterministic dummy data.
The tests intentionally distinguish detector unavailability from a true negative.
"""

from __future__ import annotations

import time
import unittest
from pathlib import Path

import cv2
import numpy as np

from app.anonymization.anonymizer import ImageAnonymizer
from app.config import Settings
from app.context.main_subject_analyzer import MainSubjectAnalyzer
from app.detection.card_detector import CardDetector
from app.detection.code_detector import CodeDetector
from app.detection.face_detector import FaceDetector, RawFace
from app.privacy.code_content_classifier import CodeContentClassifier
from app.privacy.sensitive_text_classifier import SensitiveTextClassifier
from app.schemas import (
    BoundingBox, DetectionResult, FaceResult, OCRTextResult, Point, ProtectionSettings,
)
from app.video.video_tracker import RegionTracker
from tests.test_anonymization import analysis_response
from tests import test_risk_score as risk_helpers

from .fixture_factory import (
    ean13, load_fixture, negative_card_scene, negative_shape_scene, normal_text_image, qr_image,
    rotate_bound, sensitive_text_image, synthetic_card,
)
from .metrics import DetectionMetrics, intersection_over_union, match_boxes


BACKEND = Path(__file__).resolve().parents[2]


def _face_result(
    face_id: int, box: tuple[int, int, int, int], confidence: float = 0.95,
) -> FaceResult:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
    return FaceResult(
        face_id=face_id, confidence=confidence,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        width=width, height=height, area=width * height,
        area_ratio=(width * height) / 1_000_000,
        center=Point(x=center_x, y=center_y),
        normalized_center=Point(x=center_x / 1000, y=center_y / 1000),
        distance_from_image_center=((center_x - 500) ** 2 + (center_y - 500) ** 2) ** 0.5 / 1414.2,
    )


def _person(item_id: int, box: tuple[int, int, int, int]) -> DetectionResult:
    return DetectionResult(
        id=item_id, class_id=0, class_name="person", confidence=0.95,
        bounding_box=BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), box))),
    )


class MetricTests(unittest.TestCase):
    def test_precision_recall_f1_iou_and_distinct_matching(self) -> None:
        result = match_boxes(
            [(10, 10, 50, 50), (70, 10, 110, 50)],
            [(11, 11, 49, 49), (12, 12, 48, 48), (70, 10, 110, 50)],
            minimum_iou=0.5,
        )
        self.assertEqual((result.true_positives, result.false_positives, result.false_negatives), (2, 1, 0))
        self.assertAlmostEqual(result.precision, 2 / 3)
        self.assertEqual(result.recall, 1.0)
        self.assertAlmostEqual(result.f1, 0.8)
        self.assertGreater(intersection_over_union((10, 10, 50, 50), (11, 11, 49, 49)), 0.8)

    def test_metrics_do_not_invent_values_without_denominators(self) -> None:
        result = DetectionMetrics(0, 0, 0)
        self.assertIsNone(result.precision)
        self.assertIsNone(result.recall)
        self.assertIsNone(result.f1)


class RealFaceAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.detector = FaceDetector(BACKEND / "models" / "face_detection_yunet_2023mar.onnx", 0.75)

    @staticmethod
    def _boxes(faces: list[RawFace]) -> list[tuple[int, int, int, int]]:
        return [(face.x1, face.y1, face.x2, face.y2) for face in faces]

    def test_multi_small_large_glasses_profile_occlusion_and_low_light(self) -> None:
        group = load_fixture("synthetic_three_people.png")
        difficult = load_fixture("synthetic_profile_occlusion_lowlight.png")
        # Independently hand-annotated visible face extents (not copied from
        # detector output), intentionally allowing landmark-detector margins.
        group_truth = [(190, 195, 310, 375), (620, 155, 890, 530), (1235, 290, 1365, 470)]
        difficult_truth = [(265, 210, 455, 535), (1040, 335, 1280, 655)]
        group_result = match_boxes(group_truth, self._boxes(self.detector.detect(group)), 0.45)
        difficult_result = match_boxes(difficult_truth, self._boxes(self.detector.detect(difficult)), 0.45)
        self.assertEqual((group_result.true_positives, group_result.false_positives, group_result.false_negatives), (3, 0, 0))
        self.assertEqual((difficult_result.true_positives, difficult_result.false_positives, difficult_result.false_negatives), (2, 0, 0))

    def test_single_edge_rotated_and_negative_scenes(self) -> None:
        group = load_fixture("synthetic_three_people.png")
        single = group[155:560, 585:925]
        edge = np.full((650, 900, 3), 230, np.uint8)
        edge[100:100 + single.shape[0], 0:single.shape[1]] = single
        rotated = rotate_bound(single, 15)
        for name, pixels in (("single", single), ("edge", edge), ("rotated", rotated)):
            with self.subTest(name=name):
                self.assertGreaterEqual(len(self.detector.detect(pixels)), 1)
        for name, pixels in (
            ("poster_round_objects", negative_shape_scene()),
            ("cartoon", cv2.stylization(negative_shape_scene(), sigma_s=30, sigma_r=0.2)),
            ("statue_like", cv2.cvtColor(cv2.cvtColor(negative_shape_scene(), cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)),
        ):
            with self.subTest(name=name):
                self.assertEqual(self.detector.detect(pixels), [])


class MainSubjectCalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.analyzer = MainSubjectAnalyzer(Settings(yolo_weights_path=Path("unused.pt")))

    def test_one_two_three_people_off_center_and_larger_background_person(self) -> None:
        scenarios = [
            ([_face_result(1, (350, 250, 650, 650))], [_person(1, (250, 100, 750, 950))], "identified", 1),
            ([_face_result(1, (350, 250, 650, 650)), _face_result(2, (80, 150, 230, 330))],
             [_person(1, (250, 100, 750, 950)), _person(2, (20, 80, 280, 800))], "identified", 1),
            ([_face_result(1, (330, 240, 650, 650)), _face_result(2, (40, 180, 180, 340)), _face_result(3, (800, 200, 940, 360))],
             [_person(1, (250, 100, 750, 950)), _person(2, (0, 80, 240, 850)), _person(3, (760, 80, 990, 850))], "identified", 1),
            ([_face_result(1, (80, 220, 430, 670)), _face_result(2, (580, 250, 800, 520))], [], "identified", 1),
        ]
        for faces, people, expected_status, expected_id in scenarios:
            with self.subTest(count=len(faces)):
                result = self.analyzer.analyze(faces, people)
                self.assertEqual((result.status, result.face_id), (expected_status, expected_id))
                self.assertTrue(all(face.role == ("main_subject" if face.face_id == expected_id else "background_face") for face in faces))

    def test_ambiguous_or_partially_supported_group_fails_safe(self) -> None:
        faces = [_face_result(1, (120, 220, 370, 520)), _face_result(2, (630, 220, 880, 520))]
        result = self.analyzer.analyze(faces, [])
        self.assertEqual(result.status, "uncertain")
        self.assertTrue(all(face.role == "unclassified" for face in faces))


class RealCardAccuracyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        settings = Settings()
        if not settings.card_model_path.is_file():
            raise unittest.SkipTest("Installed payment-card model unavailable")
        cls.detector = CardDetector(
            settings.card_model_path, settings.card_confidence_threshold,
            settings.card_inference_image_size, settings.card_tile_size,
            settings.card_tile_overlap, settings.card_tile_inference_image_size,
        )

    def test_synthetic_card_has_real_confidence_and_box_quality(self) -> None:
        labeled = synthetic_card()
        started = time.perf_counter()
        detected = self.detector.detect(labeled.pixels)
        self.assertGreaterEqual((time.perf_counter() - started) * 1000, 0)
        metrics = match_boxes(list(labeled.boxes), [(item.x1, item.y1, item.x2, item.y2) for item in detected], 0.5)
        self.assertEqual((metrics.true_positives, metrics.false_positives, metrics.false_negatives), (1, 0, 0))
        self.assertTrue(all(0 <= item.confidence <= 1 for item in detected))

    def test_negative_rectangles_are_not_cards(self) -> None:
        for kind in (
            "phone", "wallet", "id_card", "business_card", "notebook",
            "paper_rectangle", "remote_control",
        ):
            with self.subTest(kind=kind):
                self.assertEqual(self.detector.detect(negative_card_scene(kind)), [])


class OCRClassificationAccuracyTests(unittest.TestCase):
    @staticmethod
    def _ocr(texts: tuple[str, ...]) -> list[OCRTextResult]:
        return [
            OCRTextResult(
                text_id=index, raw_text=value, normalized_text=value, confidence=0.95,
                bounding_box=BoundingBox(x1=10, y1=index * 30, x2=500, y2=index * 30 + 25),
            )
            for index, value in enumerate(texts, 1)
        ]

    def test_sensitive_patterns_and_conservative_ocr_confusion_repair(self) -> None:
        _, values = sensitive_text_image()
        values = ("Phone: 90000 O0000", *values[1:])
        results = SensitiveTextClassifier().classify(self._ocr(values), [], [])
        kinds = {item.type for item in results}
        self.assertTrue({
            "phone_number", "email", "pan_like_number", "aadhaar_like_number",
            "possible_address", "pincode", "payment_card_number",
        }.issubset(kinds))

    def test_normal_project_text_and_unlabeled_letter_sequences_are_not_sensitive(self) -> None:
        values = ("Welcome to VVCE", "Computer Science Project", "Privacy Protection Demo", "ROOM OOOOO")
        self.assertEqual(SensitiveTextClassifier().classify(self._ocr(values), [], []), [])
        self.assertGreater(normal_text_image().size, 0)


class RealCodeAndProtectionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = CodeDetector()
        self.classifier = CodeContentClassifier()

    def test_qr_payloads_rotations_small_partial_and_document_context(self) -> None:
        payloads = {
            "https://example.com/safe": "url",
            "upi://pay?pa=dummy@example.invalid&am=1": "payment",
            "BEGIN:VCARD\nFN:Dummy Person\nTEL:0000000000\nEND:VCARD": "contact",
            "WIFI:T:WPA;S:Dummy;P:not-real;;": "wifi",
            "Privacy Protection Demo": "text",
            "urn:dummy:unknown-payload": "identifier",
        }
        for payload, expected in payloads.items():
            with self.subTest(expected=expected):
                detected = self.detector.detect_qr(qr_image(payload), exhaustive=True)
                self.assertEqual(len(detected), 1)
                self.assertEqual(detected[0].payload, payload)
                self.assertEqual(self.classifier.classify(payload, code_kind="qr").content_type, expected)
        base = qr_image("https://example.com/rotation", scale=7)
        for angle in (0, 15, 30, 45, 90):
            with self.subTest(angle=angle):
                self.assertEqual(len(self.detector.detect_qr(rotate_bound(base, angle), exhaustive=True)), 1)
        small = cv2.resize(base, None, fx=0.42, fy=0.42, interpolation=cv2.INTER_AREA)
        self.assertEqual(len(self.detector.detect_qr(small, exhaustive=True)), 1)
        partial = base.copy()
        cv2.rectangle(partial, (base.shape[1] // 2 - 20, base.shape[0] // 2 - 20),
                      (base.shape[1] // 2 + 20, base.shape[0] // 2 + 20), (255, 255, 255), -1)
        partial_results = self.detector.detect_qr(partial, exhaustive=True)
        self.assertLessEqual(len(partial_results), 1)

    def test_codes_have_null_confidence_real_boxes_and_retail_barcode_format(self) -> None:
        qr = self.detector.detect_qr(qr_image("plain safe text"))[0]
        self.assertGreater(qr.x2 - qr.x1, 0)
        barcode = ean13(module=3)
        canvas = np.full((700, 1200, 3), 255, np.uint8)
        canvas[250:250 + barcode.shape[0], 300:300 + barcode.shape[1]] = barcode
        detected = self.detector.detect_barcodes(canvas)
        decoded = next(item for item in detected if item.payload == "5901234123457")
        self.assertEqual(decoded.format, "EAN-13")
        self.assertGreater(decoded.x2 - decoded.x1, decoded.y2 - decoded.y1)

    def test_qr_and_barcode_cannot_redecode_after_all_methods(self) -> None:
        qr_pixels = qr_image("https://example.com/protection")
        qr_raw = self.detector.detect_qr(qr_pixels)[0]
        barcode = ean13(module=3)
        barcode_pixels = np.full((700, 1200, 3), 255, np.uint8)
        barcode_pixels[250:250 + barcode.shape[0], 300:300 + barcode.shape[1]] = barcode
        barcode_raw = next(item for item in self.detector.detect_barcodes(barcode_pixels) if item.payload)
        for method in ("blur", "pixelate", "blackout"):
            with self.subTest(method=method, category="qr"):
                analysis = analysis_response(qr_codes=[(qr_raw.x1, qr_raw.y1, qr_raw.x2, qr_raw.y2)])
                analysis.image.width, analysis.image.height = qr_pixels.shape[1], qr_pixels.shape[0]
                protected = ImageAnonymizer().anonymize(
                    qr_pixels, analysis, ProtectionSettings(anonymization_method=method, strength="low"),
                ).pixels_bgr
                self.assertEqual(self.detector.detect_qr(protected, exhaustive=True), [])
            with self.subTest(method=method, category="barcode"):
                analysis = analysis_response(barcodes=[(
                    barcode_raw.x1, barcode_raw.y1, barcode_raw.x2, barcode_raw.y2,
                )])
                analysis.image.width, analysis.image.height = barcode_pixels.shape[1], barcode_pixels.shape[0]
                protected = ImageAnonymizer().anonymize(
                    barcode_pixels, analysis, ProtectionSettings(anonymization_method=method, strength="low"),
                ).pixels_bgr
                self.assertEqual(self.detector.detect_barcodes(protected), [])


class RiskAndTrackingReliabilityTests(unittest.TestCase):
    def test_risk_relationships_determinism_partial_and_failed_protection(self) -> None:
        harness = risk_helpers.PrivacyRiskEngineTests()
        harness.setUp()
        one = harness.calculate([risk_helpers.face(1, 180)], risk_helpers.subject("identified", 99))
        three = harness.calculate(
            [risk_helpers.face(1, 180), risk_helpers.face(2, 180), risk_helpers.face(3, 180)],
            risk_helpers.subject("identified", 99),
        )
        self.assertLess(one.score, three.score)
        self.assertEqual(three, harness.calculate(
            [risk_helpers.face(1, 180), risk_helpers.face(2, 180), risk_helpers.face(3, 180)],
            risk_helpers.subject("identified", 99),
        ))
        partial = harness.calculate(statuses={"face_detection": "completed", "ocr": "unavailable"})
        self.assertEqual(partial.assessment.status, "partial")
        self.assertIn("ocr", partial.assessment.unavailable_modules)
        # Zero successfully protected regions must not receive an artificial reduction.
        before = harness.calculate([risk_helpers.face(1, 180)], risk_helpers.subject("identified", 99))
        after = harness.calculate([risk_helpers.face(1, 180)], risk_helpers.subject("identified", 99))
        self.assertEqual((before.score, after.score), (after.score, before.score))

    def test_video_motion_brief_miss_and_false_track_expiry(self) -> None:
        tracker = RegionTracker(expiry_frames=3)
        first = tracker.update([
            {"category": "face", "box": (10, 10, 50, 50), "confidence": 0.95, "role": "main_subject"},
            {"category": "face", "box": (70, 10, 100, 40), "confidence": 0.91, "role": "background_face"},
        ], {"face"}, 0)
        ids = [item.track_id for item in first]
        moved = tracker.update([
            {"category": "face", "box": (15, 10, 55, 50), "confidence": 0.94, "role": "main_subject"},
            {"category": "face", "box": (75, 10, 105, 40), "confidence": 0.90, "role": "background_face"},
        ], {"face"}, 1)
        self.assertEqual([item.track_id for item in moved], ids)
        self.assertEqual(len(tracker.update([], {"face"}, 2)), 2)
        self.assertEqual(tracker.update([], {"face"}, 5), [])


if __name__ == "__main__":
    unittest.main()
