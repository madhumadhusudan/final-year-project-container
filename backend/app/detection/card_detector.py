"""Dedicated payment-card detector contract."""

from pathlib import Path

from .privacy_object_detector import LocalYoloPrivacyDetector


class CardDetector(LocalYoloPrivacyDetector):
    name = "local_yolo_payment_card"
    supported_classes = {"card", "credit_card", "debit_card", "payment_card"}

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        super().__init__(model_path, confidence_threshold, self.supported_classes)
