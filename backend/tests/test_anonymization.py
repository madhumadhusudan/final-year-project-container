from __future__ import annotations

import hashlib
import json
import unittest
from urllib.parse import unquote

import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.anonymization.anonymizer import ImageAnonymizer
from app.anonymization.region_utils import Region, merge_regions, pad_region, sanitize_region
from app.analysis.session_cache import analysis_session_cache
from app.schemas import AnalysisResponse, ProtectionSettings
from main import app


def gradient_image(width: int = 100, height: int = 80) -> np.ndarray:
    x = np.arange(width, dtype=np.uint8)[None, :]
    y = np.arange(height, dtype=np.uint8)[:, None]
    return np.dstack((np.broadcast_to(x, (height, width)), np.broadcast_to(y, (height, width)), (x + y) % 255))


def face(face_id: int, box: tuple[int, int, int, int], role: str) -> dict:
    x1, y1, x2, y2 = box
    width, height = x2 - x1, y2 - y1
    return {
        "face_id": face_id, "confidence": 0.95, "bounding_box": dict(zip(("x1", "y1", "x2", "y2"), box)),
        "width": width, "height": height, "area": width * height,
        "area_ratio": (width * height) / 8000, "center": {"x": (x1 + x2) / 2, "y": (y1 + y2) / 2},
        "normalized_center": {"x": (x1 + x2) / 200, "y": (y1 + y2) / 160},
        "distance_from_image_center": 0.1, "role": role,
    }


def analysis_response(
    *, faces: list[dict] | None = None, subject_status: str = "not_found", subject_face_id: int | None = None,
    plates: list[tuple[int, int, int, int]] | None = None, cards: list[tuple[int, int, int, int]] | None = None,
    documents: list[tuple[int, int, int, int]] | None = None,
    qr_codes: list[tuple[int, int, int, int]] | None = None,
    barcodes: list[tuple[int, int, int, int]] | None = None,
    texts: list[tuple[int, int, int, int]] | None = None,
) -> AnalysisResponse:
    faces, plates, cards, documents = faces or [], plates or [], cards or [], documents or []
    qr_codes, barcodes, texts = qr_codes or [], barcodes or [], texts or []
    box = lambda values: dict(zip(("x1", "y1", "x2", "y2"), values))
    privacy_items = lambda values, name: [
        {"id": index, "class_name": name, "confidence": 0.9, "bounding_box": box(item)}
        for index, item in enumerate(values, 1)
    ]
    text_items = [
        {"id": index, "text_id": index, "type": "phone_number", "masked_value": "******3210",
         "confidence": 0.9, "reason": "test", "bounding_box": box(item)}
        for index, item in enumerate(texts, 1)
    ]
    document_items = [
        {
            "document_id": index, "class_name": "id_card", "confidence": 0.9,
            "bounding_box": box(item), "area_ratio": ((item[2] - item[0]) * (item[3] - item[1])) / 8000,
            "center": {"x": (item[0] + item[2]) / 2, "y": (item[1] + item[3]) / 2},
            "final_document_type": "identity_document", "classification_confidence": 0.675,
            "classification_status": "uncertain", "classification_reasons": ["synthetic test document"],
        }
        for index, item in enumerate(documents, 1)
    ]
    polygon = lambda item: [
        {"x": item[0], "y": item[1]}, {"x": item[2], "y": item[1]},
        {"x": item[2], "y": item[3]}, {"x": item[0], "y": item[3]},
    ]
    qr_items = [
        {
            "qr_id": index, "confidence": None, "bounding_box": box(item), "polygon": polygon(item),
            "decoded": True, "content_type": "url", "masked_preview": "URL encoded (example.com)",
            "privacy_level": "moderate",
        }
        for index, item in enumerate(qr_codes, 1)
    ]
    barcode_items = [
        {
            "barcode_id": index, "confidence": None, "format": "EAN-13",
            "bounding_box": box(item), "polygon": polygon(item), "decoded": True,
            "content_type": "identifier", "masked_preview": "********3457", "privacy_level": "low",
        }
        for index, item in enumerate(barcodes, 1)
    ]
    return AnalysisResponse.model_validate({
        "image": {"filename": "test.png", "width": 100, "height": 80, "format": "PNG"},
        "analysis": {
            "model": "test", "detection_count": 0, "detections": [],
            "object_detection": {"model": "test", "detection_count": 0, "detections": []},
            "face_detection": {"status": "completed", "detector": "test", "face_count": len(faces), "faces": faces},
            "main_subject": {"status": subject_status, "face_id": subject_face_id, "reason": "test"},
            "license_plate_detection": {"status": "completed", "detector": "test", "plate_count": len(plates), "plates": privacy_items(plates, "license_plate")},
            "card_detection": {"status": "completed", "detector": "test", "card_count": len(cards), "cards": privacy_items(cards, "card")},
            "document_detection": {"status": "completed", "detector": "test", "document_count": len(documents), "documents": document_items},
            "qr_detection": {"status": "completed", "detector": "test", "qr_count": len(qr_items), "items": qr_items},
            "barcode_detection": {"status": "completed", "detector": "test", "barcode_count": len(barcode_items), "items": barcode_items},
            "ocr": {"status": "completed", "engine": "test", "languages": ["en"], "text_count": 0, "texts": []},
            "sensitive_text": {"status": "completed", "count": len(text_items), "items": text_items},
        },
        "performance": {
            "inference_time_ms": 0, "object_detection_ms": 0, "face_detection_ms": 0,
            "license_plate_detection_ms": 0, "card_detection_ms": 0, "context_analysis_ms": 0,
            "ocr_detection_ms": 0, "sensitive_text_analysis_ms": 0, "total_analysis_ms": 0,
        },
    })


class RegionUtilityTests(unittest.TestCase):
    def test_sanitize_rejects_invalid_values_and_clamps(self) -> None:
        self.assertEqual(sanitize_region((-4, -2, 120, 90), 100, 80), Region(0, 0, 100, 80))
        for invalid in [(1, 1, 1, 5), (5, 5, 2, 2), (0, 0, float("nan"), 4), "bad"]:
            with self.subTest(invalid=invalid):
                self.assertIsNone(sanitize_region(invalid, 100, 80))

    def test_padding_and_merge_stay_safe(self) -> None:
        self.assertEqual(pad_region(Region(1, 1, 11, 11), 0.5, 12, 12), Region(0, 0, 12, 12))
        self.assertEqual(merge_regions([Region(1, 1, 10, 10), Region(9, 2, 15, 8)]), [Region(1, 1, 15, 10)])


class AnonymizationMethodTests(unittest.TestCase):
    def setUp(self) -> None:
        self.original = gradient_image()
        self.box = Region(20, 15, 70, 60)
        self.mask = np.zeros(self.original.shape[:2], dtype=bool)
        self.mask[15:60, 20:70] = True
        self.anonymizer = ImageAnonymizer()

    def test_each_method_changes_only_roi(self) -> None:
        for method in ("blur", "pixelate", "blackout"):
            with self.subTest(method=method):
                image = self.original.copy()
                changed = getattr(self.anonymizer, f"{method}_region")(image, self.box)
                self.assertTrue(changed)
                self.assertTrue(np.array_equal(image[~self.mask], self.original[~self.mask]))
                self.assertFalse(np.array_equal(image[self.mask], self.original[self.mask]))

    def test_blur_and_pixelate_strengths_are_meaningfully_different(self) -> None:
        for method in ("blur_region", "pixelate_region"):
            outputs = []
            for strength in ("low", "medium", "high"):
                image = self.original.copy()
                getattr(self.anonymizer, method)(image, self.box, strength)
                outputs.append(image)
            self.assertFalse(np.array_equal(outputs[0], outputs[1]))
            self.assertFalse(np.array_equal(outputs[1], outputs[2]))


class PrivacyRuleTests(unittest.TestCase):
    def test_main_subject_is_untouched_and_background_face_is_protected(self) -> None:
        original = gradient_image()
        analysis = analysis_response(
            faces=[face(1, (10, 10, 35, 40), "main_subject"), face(2, (65, 20, 85, 42), "background_face")],
            subject_status="identified", subject_face_id=1,
        )
        result = ImageAnonymizer().anonymize(original, analysis, ProtectionSettings(anonymization_method="blackout"))
        self.assertTrue(np.array_equal(result.pixels_bgr[10:40, 10:35], original[10:40, 10:35]))
        self.assertTrue(np.all(result.pixels_bgr[20:42, 65:85] == 0))
        self.assertEqual(result.protection.breakdown.background_faces, 1)
        self.assertTrue(result.protection.main_subject_preserved)

    def test_uncertain_subject_protects_every_face_and_warns(self) -> None:
        analysis = analysis_response(
            faces=[face(1, (10, 10, 30, 30), "unclassified"), face(2, (60, 10, 80, 30), "unclassified")],
            subject_status="uncertain",
        )
        result = ImageAnonymizer().anonymize(gradient_image(), analysis, ProtectionSettings(anonymization_method="blackout"))
        self.assertEqual(result.protection.breakdown.background_faces, 2)
        self.assertFalse(result.protection.main_subject_preserved)
        self.assertIn("All detected faces were protected", result.protection.warnings[0])

    def test_card_and_plate_take_precedence_over_contained_text(self) -> None:
        analysis = analysis_response(plates=[(5, 50, 35, 65)], cards=[(55, 10, 95, 45)], texts=[(60, 20, 85, 30), (10, 53, 28, 61), (35, 68, 55, 76)])
        result = ImageAnonymizer().anonymize(gradient_image(), analysis, ProtectionSettings(anonymization_method="blackout"))
        self.assertEqual(result.protection.breakdown.cards, 1)
        self.assertEqual(result.protection.breakdown.license_plates, 1)
        self.assertEqual(result.protection.breakdown.sensitive_text, 1)
        self.assertEqual(result.protection.regions_protected, 3)

    def test_disabled_categories_leave_the_image_unchanged(self) -> None:
        original = gradient_image()
        analysis = analysis_response(plates=[(5, 50, 35, 65)], cards=[(55, 10, 95, 45)], texts=[(35, 68, 55, 76)])
        settings = ProtectionSettings(
            protect_background_faces=False, protect_license_plates=False,
            protect_cards=False, protect_sensitive_text=False, anonymization_method="blackout",
        )
        result = ImageAnonymizer().anonymize(original, analysis, settings)
        self.assertTrue(np.array_equal(result.pixels_bgr, original))
        self.assertEqual(result.protection.regions_protected, 0)

    def test_main_subject_restoration_wins_over_overlapping_card(self) -> None:
        original = gradient_image()
        analysis = analysis_response(
            faces=[face(1, (30, 20, 60, 50), "main_subject")], subject_status="identified", subject_face_id=1,
            cards=[(20, 10, 70, 60)],
        )
        result = ImageAnonymizer().anonymize(original, analysis, ProtectionSettings(anonymization_method="blackout"))
        self.assertTrue(np.array_equal(result.pixels_bgr[20:50, 30:60], original[20:50, 30:60]))

    def test_mixed_scene_protects_every_enabled_risk(self) -> None:
        analysis = analysis_response(
            faces=[face(1, (5, 5, 25, 30), "main_subject"), face(2, (75, 5, 92, 24), "background_face")],
            subject_status="identified", subject_face_id=1,
            plates=[(5, 55, 25, 68)], cards=[(65, 45, 95, 75)], texts=[(32, 32, 57, 42)],
        )
        result = ImageAnonymizer().anonymize(gradient_image(), analysis, ProtectionSettings(anonymization_method="blackout"))
        self.assertEqual(result.protection.breakdown.model_dump(), {
            "background_faces": 1, "license_plates": 1, "cards": 1,
            "identity_documents": 0, "qr_codes": 0, "barcodes": 0, "sensitive_text": 1,
        })
        self.assertEqual(result.protection.regions_protected, 4)
        self.assertTrue(result.protection.main_subject_preserved)

    def test_each_toggle_disables_only_its_category(self) -> None:
        analysis = analysis_response(
            faces=[face(1, (5, 5, 25, 30), "main_subject"), face(2, (75, 5, 92, 24), "background_face")],
            subject_status="identified", subject_face_id=1,
            plates=[(5, 55, 25, 68)], cards=[(65, 45, 95, 75)], texts=[(32, 32, 57, 42)],
        )
        toggles = {
            "protect_background_faces": "background_faces", "protect_license_plates": "license_plates",
            "protect_cards": "cards", "protect_sensitive_text": "sensitive_text",
        }
        for setting_name, category in toggles.items():
            with self.subTest(setting=setting_name):
                result = ImageAnonymizer().anonymize(
                    gradient_image(), analysis,
                    ProtectionSettings(**{setting_name: False, "anonymization_method": "blackout"}),
                )
                self.assertEqual(getattr(result.protection.breakdown, category), 0)
                self.assertEqual(result.protection.regions_protected, 3)

    def test_zero_risk_image_is_unchanged(self) -> None:
        original = gradient_image()
        result = ImageAnonymizer().anonymize(original, analysis_response(), ProtectionSettings())
        self.assertTrue(np.array_equal(result.pixels_bgr, original))
        self.assertEqual(result.protection.regions_protected, 0)

    def test_document_whole_region_methods_toggle_and_text_precedence(self) -> None:
        original = gradient_image()
        original[15:60:2, 20:70:2] = 255
        analysis = analysis_response(documents=[(20, 15, 70, 60)], texts=[(30, 25, 60, 35)])
        for method in ("blur", "pixelate", "blackout"):
            with self.subTest(method=method):
                result = ImageAnonymizer().anonymize(
                    original, analysis, ProtectionSettings(anonymization_method=method),
                )
                self.assertEqual(result.protection.breakdown.identity_documents, 1)
                self.assertEqual(result.protection.breakdown.sensitive_text, 0)
                # The padded document ROI changes, while a distant outside corner is untouched.
                self.assertFalse(np.array_equal(result.pixels_bgr[15:60, 20:70], original[15:60, 20:70]))
                self.assertTrue(np.array_equal(result.pixels_bgr[:10, :10], original[:10, :10]))

        disabled = ImageAnonymizer().anonymize(
            original, analysis,
            ProtectionSettings(
                protect_identity_documents=False, protect_sensitive_text=False,
                anonymization_method="blackout",
            ),
        )
        self.assertTrue(np.array_equal(disabled.pixels_bgr, original))
        self.assertEqual(disabled.protection.breakdown.identity_documents, 0)

    def test_qr_and_barcode_methods_padding_and_toggles(self) -> None:
        original = gradient_image()
        original[15:60:2, 15:90:2] = 255
        analysis = analysis_response(qr_codes=[(15, 15, 40, 40)], barcodes=[(55, 42, 90, 58)])
        for method in ("blur", "pixelate", "blackout"):
            with self.subTest(method=method):
                result = ImageAnonymizer().anonymize(
                    original, analysis, ProtectionSettings(anonymization_method=method),
                )
                self.assertEqual(result.protection.breakdown.qr_codes, 1)
                self.assertEqual(result.protection.breakdown.barcodes, 1)
                self.assertEqual(result.protection.regions_protected, 2)
                self.assertFalse(np.array_equal(result.pixels_bgr[15:40, 15:40], original[15:40, 15:40]))
                self.assertFalse(np.array_equal(result.pixels_bgr[42:58, 55:90], original[42:58, 55:90]))

        for disabled_setting, disabled_category, enabled_category in (
            ("protect_qr_codes", "qr_codes", "barcodes"),
            ("protect_barcodes", "barcodes", "qr_codes"),
        ):
            with self.subTest(disabled_setting=disabled_setting):
                toggled = ImageAnonymizer().anonymize(
                    original, analysis,
                    ProtectionSettings(**{disabled_setting: False, "anonymization_method": "blackout"}),
                )
                self.assertEqual(getattr(toggled.protection.breakdown, disabled_category), 0)
                self.assertEqual(getattr(toggled.protection.breakdown, enabled_category), 1)

        disabled = ImageAnonymizer().anonymize(
            original, analysis,
            ProtectionSettings(
                protect_qr_codes=False, protect_barcodes=False, anonymization_method="blackout",
            ),
        )
        self.assertTrue(np.array_equal(disabled.pixels_bgr, original))

    def test_document_precedence_protects_contained_codes_once(self) -> None:
        analysis = analysis_response(
            documents=[(10, 10, 90, 70)], qr_codes=[(25, 20, 42, 38)],
            barcodes=[(50, 45, 75, 57)],
        )
        result = ImageAnonymizer().anonymize(
            gradient_image(), analysis, ProtectionSettings(anonymization_method="blackout"),
        )
        self.assertEqual(result.protection.breakdown.identity_documents, 1)
        self.assertEqual(result.protection.breakdown.qr_codes, 1)
        self.assertEqual(result.protection.breakdown.barcodes, 1)
        self.assertEqual(result.protection.regions_protected, 1)


class ProtectionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)
        ok, encoded = cv2.imencode(".png", gradient_image())
        self.assertTrue(ok)
        self.encoded = encoded.tobytes()
        self.analysis = analysis_response(texts=[(30, 20, 60, 35)])

    def post(self, analysis=None, settings=None):
        return self.client.post("/protect", files={"image": ("test.png", self.encoded, "image/png")}, data={
            "analysis": (analysis or self.analysis).model_dump_json(),
            "settings": json.dumps(settings or {"anonymization_method": "blackout"}),
        })

    def test_returns_image_and_real_metadata_without_storage(self) -> None:
        response = self.post()
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["content-type"], "image/png")
        metadata = json.loads(unquote(response.headers["x-protection-metadata"]))
        self.assertEqual(metadata["regions_protected"], 1)
        self.assertEqual(metadata["breakdown"]["sensitive_text"], 1)
        self.assertGreater(metadata["risk"]["before"]["score"], 0)
        self.assertEqual(metadata["risk"]["after"]["score"], 0)
        self.assertEqual(metadata["risk"]["reduction_percent"], 100.0)
        protected = cv2.imdecode(np.frombuffer(response.content, np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(protected.shape, (80, 100, 3))

    def test_analysis_id_reuses_cached_results_without_resending_payload(self) -> None:
        analysis_id = analysis_session_cache.put(self.analysis, hashlib.sha256(self.encoded).hexdigest())
        response = self.client.post("/protect", files={"image": ("test.png", self.encoded, "image/png")}, data={
            "analysis_id": analysis_id, "settings": json.dumps({"anonymization_method": "blackout"}),
        })
        self.assertEqual(response.status_code, 200)
        metadata = json.loads(unquote(response.headers["x-protection-metadata"]))
        self.assertEqual(metadata["breakdown"]["sensitive_text"], 1)

    def test_invalid_settings_and_mismatched_analysis_are_clean_errors(self) -> None:
        invalid = self.client.post("/protect", files={"image": ("test.png", self.encoded, "image/png")}, data={
            "analysis": self.analysis.model_dump_json(), "settings": '{"anonymization_method":"fake"}',
        })
        self.assertEqual(invalid.status_code, 422)
        extra = self.client.post("/protect", files={"image": ("test.png", self.encoded, "image/png")}, data={
            "analysis": self.analysis.model_dump_json(), "settings": '{"unexpected":true}',
        })
        self.assertEqual(extra.status_code, 422)
        mismatch = self.analysis.model_copy(deep=True)
        mismatch.image.width = 99
        response = self.post(analysis=mismatch)
        self.assertEqual(response.status_code, 409)
        self.assertNotIn("traceback", response.text.lower())

    def test_protect_api_returns_document_count_and_document_risk_reduction(self) -> None:
        response = self.post(analysis=analysis_response(documents=[(20, 15, 70, 60)]))
        self.assertEqual(response.status_code, 200)
        metadata = json.loads(unquote(response.headers["x-protection-metadata"]))
        self.assertEqual(metadata["breakdown"]["identity_documents"], 1)
        self.assertGreater(metadata["risk"]["before"]["score"], 0)
        self.assertEqual(metadata["risk"]["after"]["score"], 0)


if __name__ == "__main__":
    unittest.main()
