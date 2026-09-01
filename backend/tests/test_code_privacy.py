from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

from app.context.code_context import associate_code
from app.detection.code_detector import CodeDetector
from app.privacy.code_content_classifier import CodeContentClassifier
from app.privacy.risk_score import PrivacyRiskEngine
from app.schemas import (
    BarcodeResult, BoundingBox, DocumentResult, ImageDetails, MainSubjectDetails, Point,
    ProtectionBreakdown, ProtectionMetadata, ProtectionSettings, QRCodeResult,
)


def make_qr(payload: str, scale: int = 8) -> np.ndarray:
    encoded = cv2.QRCodeEncoder_create().encode(payload)
    scaled = cv2.resize(encoded, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    canvas = np.full((scaled.shape[0] + 80, scaled.shape[1] + 80, 3), 255, np.uint8)
    canvas[40:40 + scaled.shape[0], 40:40 + scaled.shape[1]] = cv2.cvtColor(
        scaled, cv2.COLOR_GRAY2BGR
    )
    return canvas


def make_multiple_qr(payloads: list[str]) -> np.ndarray:
    images = [make_qr(payload, 7) for payload in payloads]
    height = max(image.shape[0] for image in images)
    canvas = np.full((height, sum(image.shape[1] for image in images), 3), 255, np.uint8)
    offset = 0
    for image in images:
        canvas[:image.shape[0], offset:offset + image.shape[1]] = image
        offset += image.shape[1]
    return canvas


def make_ean13(value: str = "5901234123457", module: int = 5) -> np.ndarray:
    left = {
        "0": "0001101", "1": "0011001", "2": "0010011", "3": "0111101", "4": "0100011",
        "5": "0110001", "6": "0101111", "7": "0111011", "8": "0110111", "9": "0001011",
    }
    alternate = {
        "0": "0100111", "1": "0110011", "2": "0011011", "3": "0100001", "4": "0011101",
        "5": "0111001", "6": "0000101", "7": "0010001", "8": "0001001", "9": "0010111",
    }
    right = {digit: pattern.translate(str.maketrans("01", "10")) for digit, pattern in left.items()}
    parity = {
        "0": "LLLLLL", "1": "LLGLGG", "2": "LLGGLG", "3": "LLGGGL", "4": "LGLLGG",
        "5": "LGGLLG", "6": "LGGGLL", "7": "LGLGLG", "8": "LGLGGL", "9": "LGGLGL",
    }
    self_check = (10 - sum((3 if index % 2 else 1) * int(digit) for index, digit in enumerate(value[:12])) % 10) % 10
    if len(value) != 13 or int(value[-1]) != self_check:
        raise ValueError("EAN-13 test value must include a valid check digit")
    left_bits = "".join(
        (left if encoding == "L" else alternate)[digit]
        for digit, encoding in zip(value[1:7], parity[value[0]])
    )
    bits = "101" + left_bits + "01010" + "".join(right[digit] for digit in value[7:]) + "101"
    quiet = 14
    image = np.full((210, (len(bits) + quiet * 2) * module, 3), 255, np.uint8)
    for index, bit in enumerate(bits):
        if bit == "1":
            x1 = (quiet + index) * module
            cv2.rectangle(image, (x1, 20), (x1 + module - 1, 175), (0, 0, 0), -1)
    cv2.putText(image, value, (quiet * module, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return image


def box(values=(100, 100, 300, 300)) -> BoundingBox:
    return BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), values)))


def polygon(values=(100, 100, 300, 300)) -> list[Point]:
    x1, y1, x2, y2 = values
    return [Point(x=x1, y=y1), Point(x=x2, y=y1), Point(x=x2, y=y2), Point(x=x1, y=y2)]


def qr_result(content_type="payment", parent_type=None, parent_id=None) -> QRCodeResult:
    return QRCodeResult(
        qr_id=1, bounding_box=box(), polygon=polygon(), decoded=True,
        content_type=content_type, masked_preview="Safe preview", privacy_level="high",
        parent_type=parent_type, parent_id=parent_id,
    )


def barcode_result(code_format="EAN-13", parent_type=None, parent_id=None) -> BarcodeResult:
    return BarcodeResult(
        barcode_id=1, format=code_format, bounding_box=box(), polygon=polygon(), decoded=True,
        content_type="identifier", masked_preview="********3457", privacy_level="low",
        parent_type=parent_type, parent_id=parent_id,
    )


class CodeContentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = CodeContentClassifier()

    def test_sensitive_categories_are_classified_without_payload_disclosure(self) -> None:
        cases = {
            "https://example.com/private?token=super-secret": ("url", "example.com"),
            "upi://pay?pa=dummy-user@example&am=1": ("payment", "Payment information encoded"),
            "BEGIN:VCARD\nFN:Test Person\nTEL:+910000000000\nEND:VCARD": ("contact", "Contact information encoded"),
            "WIFI:T:WPA;S:DummyNetwork;P:fake-password;;": ("wifi", "Wi-Fi credentials encoded"),
        }
        for payload, (expected_type, expected_preview) in cases.items():
            with self.subTest(expected_type=expected_type):
                result = self.classifier.classify(payload, code_kind="qr")
                self.assertEqual(result.content_type, expected_type)
                self.assertIn(expected_preview, result.masked_preview)
                for private_part in ("super-secret", "dummy-user", "+910000000000", "fake-password"):
                    self.assertNotIn(private_part, result.masked_preview)

    def test_unknown_and_barcode_values_are_safely_masked(self) -> None:
        unknown = self.classifier.classify(None, code_kind="qr")
        barcode = self.classifier.classify("5901234123457", code_kind="barcode", barcode_format="EAN-13")
        self.assertEqual((unknown.content_type, unknown.masked_preview), (
            "unknown", "QR code content could not be decoded",
        ))
        self.assertEqual(barcode.content_type, "identifier")
        self.assertTrue(barcode.masked_preview.endswith("3457"))
        self.assertNotIn("590123412", barcode.masked_preview)

    def test_malicious_looking_data_is_never_executed_or_requested(self) -> None:
        payloads = [
            "https://example.com/?next=javascript:alert(1)&token=hidden",
            "powershell -Command Remove-Item fake-file",
            "file:///C:/private.txt",
        ]
        with patch("subprocess.run") as run, patch("urllib.request.urlopen") as request:
            results = [self.classifier.classify(payload, code_kind="qr") for payload in payloads]
        run.assert_not_called()
        request.assert_not_called()
        self.assertEqual(results[0].content_type, "url")
        self.assertNotIn("token", results[0].masked_preview)


class RealCodeDetectorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.detector = CodeDetector()

    def test_real_qr_decode_for_all_safe_categories(self) -> None:
        cases = [
            ("https://example.com", "url"),
            ("upi://pay?pa=dummy@example&am=1", "payment"),
            ("BEGIN:VCARD\nFN:Test Person\nTEL:0000000000\nEND:VCARD", "contact"),
            ("WIFI:T:WPA;S:Dummy;P:not-a-real-password;;", "wifi"),
        ]
        classifier = CodeContentClassifier()
        for payload, expected in cases:
            with self.subTest(expected=expected):
                detected = self.detector.detect_qr(make_qr(payload))
                self.assertEqual(len(detected), 1)
                self.assertEqual(detected[0].payload, payload)
                self.assertEqual(classifier.classify(detected[0].payload, code_kind="qr").content_type, expected)

    def test_multiple_qr_codes_have_independent_real_regions(self) -> None:
        payloads = ["https://example.com", "upi://pay?pa=dummy@example&am=1"]
        detected = self.detector.detect_qr(make_multiple_qr(payloads))
        self.assertEqual(len(detected), 2)
        self.assertEqual({item.payload for item in detected}, set(payloads))
        self.assertNotEqual((detected[0].x1, detected[0].x2), (detected[1].x1, detected[1].x2))

    def test_partly_obscured_qr_never_crashes(self) -> None:
        image = make_qr("https://example.com")
        height, width = image.shape[:2]
        cv2.rectangle(
            image, (round(width * 0.4), round(height * 0.4)),
            (round(width * 0.6), round(height * 0.6)), (255, 255, 255), -1,
        )
        results = self.detector.detect_qr(image)
        self.assertEqual(len(results), 1)
        self.assertIsNone(results[0].payload)

    def test_real_ean13_barcode_decode_and_actual_format(self) -> None:
        barcode = make_ean13(module=3)
        image = np.full((700, 1200, 3), 255, np.uint8)
        image[250:250 + barcode.shape[0], 300:300 + barcode.shape[1]] = barcode
        detected = self.detector.detect_barcodes(image)
        self.assertGreaterEqual(len(detected), 1)
        decoded = next(item for item in detected if item.payload == "5901234123457")
        self.assertEqual(decoded.format, "EAN-13")
        self.assertGreater(decoded.x2 - decoded.x1, decoded.y2 - decoded.y1)

    def test_multiple_barcodes_have_independent_real_regions(self) -> None:
        first = make_ean13("5901234123457", module=3)
        second = make_ean13("4006381333931", module=3)
        image = np.full((1000, 1400, 3), 255, np.uint8)
        image[150:150 + first.shape[0], 250:250 + first.shape[1]] = first
        image[600:600 + second.shape[0], 700:700 + second.shape[1]] = second
        detected = self.detector.detect_barcodes(image)
        self.assertEqual(len(detected), 2)
        self.assertEqual({item.payload for item in detected}, {"5901234123457", "4006381333931"})
        self.assertTrue(all(item.format == "EAN-13" for item in detected))

    def test_unavailable_barcode_decoder_is_explicit(self) -> None:
        self.detector._barcode = None
        with self.assertRaisesRegex(RuntimeError, "unavailable"):
            self.detector.detect_barcodes(np.zeros((100, 100, 3), np.uint8))


class CodeContextAndRiskTests(unittest.TestCase):
    def test_document_and_card_association_requires_spatial_evidence(self) -> None:
        document = SimpleNamespace(document_id=7, bounding_box=box((50, 50, 400, 400)))
        card = SimpleNamespace(id=9, card_id=9, bounding_box=box((500, 50, 900, 400)))
        self.assertEqual(associate_code(box((100, 100, 200, 200)), [document], [card]), ("identity_document", 7))
        self.assertEqual(associate_code(box((550, 100, 700, 250)), [document], [card]), ("payment_card", 9))
        self.assertEqual(associate_code(box((420, 500, 520, 600)), [document], [card]), (None, None))

    def test_product_barcode_is_low_risk_and_unknown_qr_is_not_safe(self) -> None:
        engine = PrivacyRiskEngine()
        image = ImageDetails(filename="synthetic.png", width=1000, height=1000, format="PNG")
        subject = MainSubjectDetails(status="not_found", reason="test")
        statuses = {"qr_detection": "completed", "barcode_detection": "completed"}
        product = engine.calculate(
            image, [], subject, [], [], [], statuses, barcodes=[barcode_result()],
        )
        unknown = qr_result("unknown")
        unknown.decoded = False
        unknown.masked_preview = "QR code content could not be decoded"
        undecoded = engine.calculate(image, [], subject, [], [], [], statuses, qr_codes=[unknown])
        self.assertLessEqual(product.score, 3)
        self.assertGreater(undecoded.score, product.score)

    def test_document_qr_adds_only_limited_bonus_and_protection_reduces_it(self) -> None:
        engine = PrivacyRiskEngine()
        image = ImageDetails(filename="synthetic.png", width=1000, height=1000, format="PNG")
        subject = MainSubjectDetails(status="not_found", reason="test")
        document = DocumentResult(
            document_id=1, class_name="id_card", confidence=0.95,
            bounding_box=box((50, 50, 500, 500)), area_ratio=0.2025,
            center=Point(x=275, y=275), final_document_type="identity_document",
            classification_confidence=0.9, classification_status="model_confirmed",
            classification_reasons=["synthetic"],
        )
        statuses = {"document_detection": "completed", "qr_detection": "completed", "barcode_detection": "completed"}
        base = engine.calculate(image, [], subject, [], [], [], statuses, documents=[document])
        contextual = qr_result(parent_type="identity_document", parent_id=1)
        combined = engine.calculate(
            image, [], subject, [], [], [], statuses, documents=[document], qr_codes=[contextual],
        )
        self.assertGreater(combined.score, base.score)
        self.assertLessEqual(combined.score - base.score, 7)
        self.assertIn("document_qr", {factor.type for factor in combined.factors})
        contextual_barcode = barcode_result(parent_type="identity_document", parent_id=1)
        with_barcode = engine.calculate(
            image, [], subject, [], [], [], statuses,
            documents=[document], barcodes=[contextual_barcode],
        )
        self.assertGreater(with_barcode.score, base.score)
        self.assertLessEqual(with_barcode.score - base.score, 5)
        self.assertIn("document_barcode", {factor.type for factor in with_barcode.factors})
        protected = engine.calculate(
            image, [], subject, [], [], [], statuses,
            ProtectionSettings(anonymization_method="blackout"),
            ProtectionMetadata(
                method="blackout", strength="medium", regions_protected=1,
                breakdown=ProtectionBreakdown(identity_documents=1, qr_codes=1),
                main_subject_preserved=False,
            ),
            documents=[document], qr_codes=[contextual],
        )
        self.assertEqual(protected.score, 0)


if __name__ == "__main__":
    unittest.main()
