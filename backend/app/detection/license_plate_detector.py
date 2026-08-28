"""Dedicated license-plate detector contract."""

from pathlib import Path

from .privacy_object_detector import LocalYoloPrivacyDetector


class LicensePlateDetector(LocalYoloPrivacyDetector):
    name = "local_yolo_license_plate"
    supported_classes = {"license_plate", "licence_plate"}

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        super().__init__(model_path, confidence_threshold, self.supported_classes)
