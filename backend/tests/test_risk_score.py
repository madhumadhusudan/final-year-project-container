from __future__ import annotations

import unittest

from app.privacy.risk_score import PrivacyRiskEngine
from app.schemas import (
    BoundingBox, CardResult, FaceResult, ImageDetails, MainSubjectDetails, Point,
    PrivacyObjectResult, ProtectionBreakdown, ProtectionMetadata, ProtectionSettings,
    SensitiveTextResult,
)


IMAGE = ImageDetails(filename="synthetic.png", width=1000, height=1000, format="PNG")
COMPLETE = {
    "face_detection": "completed", "license_plate_detection": "completed",
    "card_detection": "completed", "ocr": "completed",
}


def box(values: tuple[int, int, int, int]) -> BoundingBox:
    return BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), values)))


def face(face_id: int, size: int = 100, confidence: float = 0.95, role: str = "background_face") -> FaceResult:
    area = size * size
    return FaceResult(
        face_id=face_id, confidence=confidence, bounding_box=box((10, 10, 10 + size, 10 + size)),
        width=size, height=size, area=area, area_ratio=area / 1_000_000,
        center=Point(x=10 + size / 2, y=10 + size / 2),
        normalized_center=Point(x=(10 + size / 2) / 1000, y=(10 + size / 2) / 1000),
        distance_from_image_center=0.2, role=role,
    )


def privacy_object(item_id: int, values=(100, 100, 300, 200), confidence=0.95, card=False):
    model = CardResult if card else PrivacyObjectResult
    extra = {"card_id": item_id} if card else {}
    return model(
        id=item_id, class_name="card" if card else "license_plate",
        confidence=confidence, bounding_box=box(values), **extra,
    )


def sensitive(item_id: int, kind: str, values=(100, 100, 400, 140), confidence=0.95,
              masked="masked", luhn=None) -> SensitiveTextResult:
    return SensitiveTextResult(
        id=item_id, text_id=item_id, type=kind, masked_value=masked,
        confidence=confidence, reason="synthetic", bounding_box=box(values), luhn_valid=luhn,
    )


def subject(status="not_found", face_id=None) -> MainSubjectDetails:
    return MainSubjectDetails(status=status, face_id=face_id, reason="synthetic")


class PrivacyRiskEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PrivacyRiskEngine()

    def calculate(self, faces=None, main_subject=None, plates=None, cards=None, texts=None, statuses=None,
                  settings=None, protection=None):
        return self.engine.calculate(
            IMAGE, faces or [], main_subject or subject(), plates or [], cards or [], texts or [],
            statuses or COMPLETE, settings, protection,
        )

    def test_zero_risk_and_confident_main_subject_are_not_penalized(self) -> None:
        empty = self.calculate()
        main = face(1, size=250, role="main_subject")
        main_only = self.calculate([main], subject("identified", 1))
        self.assertEqual(empty.score, 0)
        self.assertEqual(main_only.score, 0)
        self.assertEqual(main_only.breakdown.background_faces, 0)

    def test_background_faces_are_monotonic_and_visibility_weighted(self) -> None:
        tiny = self.calculate([face(1, size=30)], subject("identified", 99))
        clear = self.calculate([face(1, size=180)], subject("identified", 99))
        three = self.calculate([face(1, 180), face(2, 180), face(3, 180)], subject("identified", 99))
        self.assertGreater(clear.score, tiny.score)
        self.assertGreater(three.score, clear.score)
        self.assertLessEqual(three.breakdown.background_faces, 35)

    def test_uncertain_subject_scores_context_only_when_faces_exist(self) -> None:
        no_faces = self.calculate(main_subject=subject("uncertain"))
        uncertain = self.calculate([face(1)], subject("uncertain"))
        self.assertEqual(no_faces.breakdown.context_uncertainty, 0)
        self.assertGreater(uncertain.breakdown.context_uncertainty, 0)

    def test_plate_confidence_visibility_and_readability_increase_risk(self) -> None:
        low = self.calculate(plates=[privacy_object(1, (10, 10, 40, 25), 0.2)])
        clear = self.calculate(plates=[privacy_object(1, (10, 10, 310, 110), 0.98)])
        readable = self.calculate(
            plates=[privacy_object(1, (10, 10, 310, 110), 0.98)],
            texts=[sensitive(1, "license_plate_text", (30, 35, 280, 80), 0.98)],
        )
        self.assertGreater(clear.score, low.score)
        self.assertGreater(readable.score, clear.score)
        self.assertEqual(readable.breakdown.sensitive_text, 0)

    def test_card_and_card_number_are_grouped_under_card_cap(self) -> None:
        card = privacy_object(1, (100, 100, 600, 400), 0.98, card=True)
        base = self.calculate(cards=[card])
        readable = self.calculate(
            cards=[card],
            texts=[sensitive(1, "payment_card_number", (150, 180, 550, 230), 0.98, "****1111", True)],
        )
        self.assertGreater(readable.score, base.score)
        self.assertLessEqual(readable.breakdown.payment_cards, 35)
        self.assertEqual(readable.breakdown.sensitive_text, 0)
        self.assertLess(readable.score - base.score, 8)

    def test_pii_types_produce_high_risk_without_raw_values(self) -> None:
        items = [
            sensitive(1, "aadhaar_like_number", (10, 10, 500, 80), masked="XXXX XXXX 9012"),
            sensitive(2, "pan_like_number", (10, 100, 500, 170), masked="ABCDE****F"),
            sensitive(3, "phone_number", (10, 190, 500, 260), masked="******3210"),
            sensitive(4, "email", (10, 280, 500, 350), masked="a***@example.com"),
        ]
        result = self.calculate(texts=items)
        self.assertGreaterEqual(result.score, 60)
        self.assertIn(result.level, {"HIGH", "CRITICAL"})
        explanations = " ".join(factor.reason for factor in result.factors)
        for raw_value in ("9012", "ABCDE", "3210", "example.com"):
            self.assertNotIn(raw_value, explanations)

    def test_duplicate_ocr_and_address_pincode_are_not_double_counted(self) -> None:
        one = sensitive(1, "phone_number", masked="******3210")
        duplicate = sensitive(2, "phone_number", (105, 102, 405, 142), masked="******3210")
        single_score = self.calculate(texts=[one]).score
        self.assertEqual(self.calculate(texts=[one, duplicate]).score, single_score)
        address = sensitive(3, "possible_address", (10, 200, 500, 260), masked="Sensitive location text")
        pincode = sensitive(4, "pincode", (350, 220, 450, 250), masked="******0001")
        grouped = self.calculate(texts=[address, pincode])
        self.assertEqual(len([factor for factor in grouped.factors if factor.type == "pincode"]), 0)

    def test_level_boundaries_determinism_and_total_cap(self) -> None:
        expected = {
            0: "LOW", 19: "LOW", 20: "MODERATE", 39: "MODERATE",
            40: "ELEVATED", 59: "ELEVATED", 60: "HIGH", 79: "HIGH",
            80: "CRITICAL", 100: "CRITICAL",
        }
        for score, level in expected.items():
            self.assertEqual(self.engine.level_for_score(score), level)
        many = [
            sensitive(index, "aadhaar_like_number", (0, index * 30, 900, index * 30 + 20), masked=f"masked-{index}")
            for index in range(1, 30)
        ]
        first = self.calculate(texts=many)
        second = self.calculate(texts=many)
        self.assertEqual(first, second)
        self.assertLessEqual(first.score, 100)
        self.assertEqual(first.breakdown.sensitive_text, 60)

    def test_partial_assessment_lists_failed_or_unavailable_modules(self) -> None:
        result = self.calculate(statuses={**COMPLETE, "card_detection": "unavailable", "ocr": "error"})
        self.assertEqual(result.assessment.status, "partial")
        self.assertEqual(result.assessment.unavailable_modules, ["card_detection", "ocr"])

    def test_residual_risk_respects_actual_protection_and_toggle(self) -> None:
        phone = sensitive(1, "phone_number", masked="******3210")
        before = self.calculate(texts=[phone])
        protected_metadata = ProtectionMetadata(
            method="blackout", strength="medium", regions_protected=1,
            breakdown=ProtectionBreakdown(sensitive_text=1), main_subject_preserved=False,
        )
        after = self.calculate(
            texts=[phone], settings=ProtectionSettings(anonymization_method="blackout"),
            protection=protected_metadata,
        )
        comparison = self.engine.compare(before, after)
        self.assertEqual(after.score, 0)
        self.assertEqual(comparison.reduction, before.score)
        self.assertEqual(comparison.reduction_percent, 100.0)

        disabled = self.calculate(
            texts=[phone],
            settings=ProtectionSettings(protect_sensitive_text=False, anonymization_method="blackout"),
            protection=ProtectionMetadata(
                method="blackout", strength="medium", regions_protected=0,
                breakdown=ProtectionBreakdown(), main_subject_preserved=False,
            ),
        )
        self.assertEqual(disabled.score, before.score)

    def test_zero_before_score_has_safe_reduction_percent(self) -> None:
        zero = self.calculate()
        comparison = self.engine.compare(zero, zero)
        self.assertEqual(comparison.reduction, 0)
        self.assertEqual(comparison.reduction_percent, 0.0)


if __name__ == "__main__":
    unittest.main()
