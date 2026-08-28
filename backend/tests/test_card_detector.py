from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from app.config import Settings
from app.detection.card_detector import CardDetector
from app.detection.privacy_object_detector import LocalYoloPrivacyDetector, normalize_model_label


class FakeTensor:
    def __init__(self, values) -> None:
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


class FakeBoxes:
    def __init__(self) -> None:
        self.xyxy = FakeTensor([
            [11.2, 12.4, 101.8, 71.6], [5, 5, 70, 75],
            [20, 25, 30, 35], [30, 20, 50, 25], [10, 20, 60, 30], [20, 40, 40, 45],
        ])
        self.conf = FakeTensor([0.91, 0.84, 0.80, 0.78, 0.88, 0.72])
        self.cls = FakeTensor([0, 1, 2, 3, 4, 3])

    def __len__(self) -> int:
        return 6


class FakeYolo:
    names = {0: "creditCardFront", 1: "creditCardBack", 2: "chip", 3: "text", 4: "band"}

    def __init__(self, _path: str) -> None:
        self.predict_kwargs = None

    def predict(self, _image, **kwargs):
        self.predict_kwargs = kwargs
        return [SimpleNamespace(boxes=FakeBoxes(), names=self.names)]


class CardDetectorContractTests(unittest.TestCase):
    def test_camel_case_labels_preserve_actual_trained_semantics(self) -> None:
        self.assertEqual(normalize_model_label("creditCardFront"), "credit_card_front")
        self.assertEqual(normalize_model_label("Credit Card Back"), "credit_card_back")
        self.assertEqual(normalize_model_label("rectangular-paper"), "rectangular_paper")

    def test_card_detector_filters_to_real_card_classes_and_uses_small_object_settings(self) -> None:
        fake_module = SimpleNamespace(YOLO=FakeYolo)
        with patch.dict(sys.modules, {"ultralytics": fake_module}):
            detector = CardDetector(Path(__file__), 0.25, 960)
            run = detector.detect_with_diagnostics(np.zeros((80, 120, 3), dtype=np.uint8))
        self.assertEqual([item.class_name for item in run.detections], ["credit_card_front", "credit_card_back"])
        self.assertEqual(run.detections[0].x1, 11)
        self.assertEqual(run.raw_detection_count, 6)
        self.assertEqual(run.accepted_detection_count, 2)
        self.assertEqual(detector._model.predict_kwargs["conf"], 0.25)
        self.assertEqual(detector._model.predict_kwargs["imgsz"], 960)
        self.assertEqual(detector._model.predict_kwargs["classes"], [0, 1, 2, 3, 4])

    def test_generic_rectangle_model_is_rejected(self) -> None:
        class RectangleYolo(FakeYolo):
            names = {0: "rectangle", 1: "paper", 2: "book"}

        with patch.dict(sys.modules, {"ultralytics": SimpleNamespace(YOLO=RectangleYolo)}):
            with self.assertRaisesRegex(RuntimeError, "class labels"):
                CardDetector(Path(__file__), 0.25, 960)

    def test_card_settings_are_independent_from_other_privacy_detectors(self) -> None:
        configured = Settings(yolo_weights_path=Path("unused.pt"))
        self.assertEqual(configured.card_confidence_threshold, 0.25)
        self.assertEqual(configured.card_inference_image_size, 960)
        self.assertEqual(configured.card_tile_size, 640)
        self.assertEqual(configured.card_tile_inference_image_size, 640)
        self.assertEqual(configured.card_tile_overlap, 0.4)


class InstalledCardModelTests(unittest.TestCase):
    model_path = Settings(yolo_weights_path=Path("unused.pt")).card_model_path

    @unittest.skipUnless(model_path.is_file(), "Local card detector weights are not installed")
    def test_installed_model_loads_with_verified_payment_card_face_classes(self) -> None:
        detector = CardDetector(self.model_path, 0.25, 960)
        self.assertEqual(detector.model_name, "card_detector.pt")
        self.assertIn("credit_card_front", detector.model_class_names)
        self.assertIn("credit_card_back", detector.model_class_names)
        self.assertNotIn("rectangle", detector.model_class_names)


if __name__ == "__main__":
    unittest.main()
