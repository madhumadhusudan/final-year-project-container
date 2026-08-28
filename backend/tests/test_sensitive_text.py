import unittest

from app.privacy.sensitive_text_classifier import SensitiveTextClassifier, luhn_valid
from app.schemas import BoundingBox, OCRTextResult, PrivacyObjectResult


def text(text_id: int, value: str, x1: int = 10, y1: int = 10, x2: int = 210, y2: int = 40) -> OCRTextResult:
    return OCRTextResult(
        text_id=text_id, raw_text=value, normalized_text=value, confidence=0.96,
        bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def region(class_name: str) -> PrivacyObjectResult:
    return PrivacyObjectResult(
        id=1, class_name=class_name, confidence=0.9,
        bounding_box=BoundingBox(x1=0, y1=0, x2=240, y2=80),
    )


class SensitiveTextClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = SensitiveTextClassifier()

    def classify(self, value: str, plates=None, cards=None):
        return self.classifier.classify([text(1, value)], plates or [], cards or [])

    def test_phone_email_pan_and_aadhaar_are_masked(self) -> None:
        cases = [
            ("Contact +91 98765 43210", "phone_number", "******3210"),
            ("hello@example.com", "email", "h***@example.com"),
            ("ABCDE1234F", "pan_like_number", "ABCDE****F"),
            ("1234 5678 9012", "aadhaar_like_number", "XXXX XXXX 9012"),
        ]
        for value, expected_type, expected_mask in cases:
            with self.subTest(value=value):
                item = self.classify(value)[0]
                self.assertEqual(item.type, expected_type)
                self.assertEqual(item.masked_value, expected_mask)

    def test_payment_card_requires_luhn_or_confirmed_card_context(self) -> None:
        self.assertTrue(luhn_valid("4242 4242 4242 4242"))
        item = self.classify("4242 4242 4242 4242")[0]
        self.assertEqual(item.type, "payment_card_number")
        self.assertTrue(item.luhn_valid)
        self.assertEqual(item.masked_value, "**** **** **** 4242")
        self.assertEqual(self.classify("1234 5678 9012 3456"), [])
        contextual = self.classify("1234 5678 9012 3456", cards=[region("card")])[0]
        self.assertFalse(contextual.luhn_valid)

    def test_expiry_and_plate_text_require_detector_context(self) -> None:
        self.assertEqual(self.classify("08/29"), [])
        self.assertEqual(self.classify("08/29", cards=[region("card")])[0].type, "possible_expiry_date")
        self.assertEqual(self.classify("KA01AB1234"), [])
        self.assertEqual(self.classify("KA01AB1234", plates=[region("license_plate")])[0].type, "license_plate_text")

    def test_address_and_pincode_use_conservative_context(self) -> None:
        texts = [text(1, "123 Example Main Road"), text(2, "Mysuru", 10, 45, 100, 70), text(3, "570001", 10, 75, 90, 100)]
        items = self.classifier.classify(texts, [], [])
        self.assertEqual([item.type for item in items], ["possible_address", "pincode"])
        self.assertEqual(self.classify("570001"), [])

    def test_normal_text_is_not_sensitive(self) -> None:
        for value in ["Happy Birthday", "Coffee Shop", "Welcome"]:
            self.assertEqual(self.classify(value), [])


if __name__ == "__main__":
    unittest.main()
