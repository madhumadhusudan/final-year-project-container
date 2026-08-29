from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.config import Settings
from app.detection.document_detector import DocumentDetector
from app.privacy.document_classifier import DocumentClassifier
from app.privacy.risk_score import PrivacyRiskEngine
from app.schemas import (
    BoundingBox, DocumentResult, ImageDetails, MainSubjectDetails, OCRTextResult, Point,
    ProtectionBreakdown, ProtectionMetadata, ProtectionSettings, SensitiveTextResult,
)


def box(values=(100, 100, 700, 500)) -> BoundingBox:
    return BoundingBox(**dict(zip(("x1", "y1", "x2", "y2"), values)))


def ocr(text_id: int, value: str, values=(130, 130, 650, 180)) -> OCRTextResult:
    return OCRTextResult(
        text_id=text_id, raw_text=value, normalized_text=value, confidence=0.96,
        bounding_box=box(values),
    )


def sensitive(item_id: int, kind: str, values=(150, 210, 600, 260)) -> SensitiveTextResult:
    return SensitiveTextResult(
        id=item_id, text_id=item_id, type=kind, masked_value="MASKED TEST VALUE",
        confidence=0.94, reason="safe synthetic fixture", bounding_box=box(values),
    )


class DocumentClassifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classifier = DocumentClassifier()

    def classify(self, model_class: str, texts=None, sensitive_items=None):
        return self.classifier.classify(
            1, model_class, 0.88, box(), 1000, 800, texts or [], sensitive_items or [],
        )

    def test_aadhaar_and_pan_model_labels_gain_masked_pattern_support(self) -> None:
        aadhaar = self.classify(
            "aadhaar_card", [ocr(1, "Government of India Aadhaar")],
            [sensitive(1, "aadhaar_like_number")],
        )
        pan = self.classify(
            "pan_card", [ocr(1, "Income Tax Department Permanent Account Number")],
            [sensitive(1, "pan_like_number")],
        )
        self.assertEqual(aadhaar.final_document_type, "aadhaar_card")
        self.assertEqual(pan.final_document_type, "pan_card")
        self.assertEqual(aadhaar.classification_status, "context_supported")
        self.assertEqual(pan.classification_status, "context_supported")
        self.assertGreater(aadhaar.classification_confidence, aadhaar.confidence)
        self.assertNotIn("MASKED TEST VALUE", " ".join(aadhaar.classification_reasons))

    def test_passport_and_driving_licence_explicit_labels_are_not_ocr_dependent(self) -> None:
        passport = self.classify("passport")
        licence = self.classify("driving_licence")
        self.assertEqual(passport.final_document_type, "passport")
        self.assertEqual(licence.final_document_type, "driving_license")
        self.assertEqual(passport.classification_status, "model_confirmed")
        self.assertEqual(licence.classification_status, "model_confirmed")

    def test_generic_document_requires_multiple_clear_context_clues(self) -> None:
        weak = self.classify("document", [ocr(1, "PASSPORT")])
        supported = self.classify("document", [ocr(1, "PASSPORT REPUBLIC OF INDIA NATIONALITY")])
        unrelated = self.classify("id_card", [ocr(1, "Sample Club Member")])
        self.assertEqual(weak.final_document_type, "identity_document")
        self.assertEqual(weak.classification_status, "uncertain")
        self.assertEqual(supported.final_document_type, "passport")
        self.assertEqual(supported.classification_status, "context_supported")
        self.assertEqual(unrelated.final_document_type, "identity_document")


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    xyxy = FakeTensor([[10, 12, 100, 70], [20, 20, 80, 60], [5, 5, 40, 40]])
    conf = FakeTensor([0.91, 0.86, 0.80])
    cls = FakeTensor([0, 1, 2])

    def __len__(self) -> int:
        return 3


class FakeDocumentYolo:
    names = {0: "aadhaarCard", 1: "passport", 2: "paper"}

    def __init__(self, _path: str) -> None:
        self.predict_kwargs = None

    def predict(self, _image, **kwargs):
        self.predict_kwargs = kwargs
        return [SimpleNamespace(boxes=FakeBoxes(), names=self.names)]


class DocumentDetectorContractTests(unittest.TestCase):
    def test_document_settings_are_independent_and_conservative(self) -> None:
        configured = Settings(yolo_weights_path=Path("unused.pt"))
        self.assertEqual(configured.document_confidence_threshold, 0.35)
        self.assertEqual(configured.document_inference_image_size, 960)
        self.assertEqual(configured.document_model_path.name, "document_detector.pt")

    def test_detector_uses_only_actual_supported_model_classes(self) -> None:
        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=FakeDocumentYolo)}):
            detector = DocumentDetector(Path(__file__), 0.35, 960)
            detections = detector.detect(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertEqual([item.class_name for item in detections], ["aadhaar_card", "passport"])
        self.assertEqual(detector.model_class_names, ["aadhaar_card", "passport", "paper"])
        self.assertEqual(detector._model.predict_kwargs["classes"], [0, 1])
        self.assertEqual(detector._model.predict_kwargs["conf"], 0.35)
        self.assertEqual(detector._model.predict_kwargs["imgsz"], 960)

    def test_book_phone_paper_and_rectangle_model_is_rejected(self) -> None:
        class GenericYolo(FakeDocumentYolo):
            names = {0: "book", 1: "cell phone", 2: "paper", 3: "rectangle"}

        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=GenericYolo)}):
            with self.assertRaisesRegex(RuntimeError, "class labels"):
                DocumentDetector(Path(__file__), 0.35, 960)


class DocumentRiskTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PrivacyRiskEngine()
        self.image = ImageDetails(filename="safe-synthetic.png", width=1000, height=800, format="PNG")
        self.statuses = {
            "face_detection": "completed", "license_plate_detection": "completed",
            "card_detection": "completed", "document_detection": "completed", "ocr": "completed",
        }

    @staticmethod
    def document(*, text_ids=None, sensitive_ids=None, size_box=(100, 100, 700, 500)) -> DocumentResult:
        item_box = box(size_box)
        area = (item_box.x2 - item_box.x1) * (item_box.y2 - item_box.y1)
        return DocumentResult(
            document_id=1, class_name="aadhaar_card", confidence=0.94,
            bounding_box=item_box, area_ratio=area / 800_000,
            center=Point(x=(item_box.x1 + item_box.x2) / 2, y=(item_box.y1 + item_box.y2) / 2),
            final_document_type="aadhaar_card", classification_confidence=0.98,
            classification_status="context_supported", classification_reasons=["safe fixture"],
            ocr_text_ids=text_ids or [], sensitive_text_ids=sensitive_ids or [],
        )

    def calculate(self, document, texts=None, statuses=None, settings=None, metadata=None):
        return self.engine.calculate(
            self.image, [], MainSubjectDetails(status="not_found", reason="no external face"),
            [], [], texts or [], statuses or self.statuses, settings, metadata,
            documents=[document] if document else [],
        )

    def test_document_risk_visibility_grouping_partial_status_and_reduction(self) -> None:
        aadhaar_text = sensitive(1, "aadhaar_like_number")
        tiny = self.calculate(self.document(size_box=(10, 10, 70, 45)))
        clear = self.calculate(self.document(text_ids=[1, 2], sensitive_ids=[1]), [aadhaar_text])
        self.assertGreater(clear.score, tiny.score)
        self.assertGreater(clear.breakdown.identity_documents, 0)
        self.assertEqual(clear.breakdown.sensitive_text, 0)

        partial = self.calculate(None, statuses={**self.statuses, "document_detection": "unavailable"})
        self.assertEqual(partial.assessment.status, "partial")
        self.assertIn("document_detection", partial.assessment.unavailable_modules)

        metadata = ProtectionMetadata(
            method="blackout", strength="medium", regions_protected=1,
            breakdown=ProtectionBreakdown(identity_documents=1), main_subject_preserved=False,
        )
        after = self.calculate(
            self.document(text_ids=[1], sensitive_ids=[1]), [aadhaar_text],
            settings=ProtectionSettings(anonymization_method="blackout"), metadata=metadata,
        )
        self.assertEqual(after.score, 0)

        unchanged = self.calculate(
            self.document(text_ids=[1, 2], sensitive_ids=[1]), [aadhaar_text],
            settings=ProtectionSettings(
                protect_identity_documents=False, anonymization_method="blackout",
            ), metadata=ProtectionMetadata(
                method="blackout", strength="medium", regions_protected=0,
                breakdown=ProtectionBreakdown(), main_subject_preserved=False,
            ),
        )
        self.assertEqual(unchanged.score, clear.score)


if __name__ == "__main__":
    unittest.main()
