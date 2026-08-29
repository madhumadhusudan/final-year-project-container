"""Dedicated local identity-document detector contract."""

from pathlib import Path

from .privacy_object_detector import LocalYoloPrivacyDetector


class DocumentDetector(LocalYoloPrivacyDetector):
    """Accept only model-native identity/document labels from a dedicated model."""

    name = "local_yolo_identity_document"
    supported_classes = {
        "aadhaar", "aadhar", "aadhaar_card", "aadhar_card",
        "pan", "pan_card", "permanent_account_number_card",
        "passport", "indian_passport",
        "driving_license", "driving_licence", "driver_license", "driver_licence",
        "drivers_license", "drivers_licence",
        "id_card", "identity_card", "identity_document", "document",
    }

    def __init__(self, model_path: Path, confidence_threshold: float, inference_image_size: int = 960) -> None:
        super().__init__(
            model_path, confidence_threshold, self.supported_classes,
            inference_image_size=inference_image_size,
        )
