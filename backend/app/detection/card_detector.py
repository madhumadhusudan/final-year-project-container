"""Dedicated payment-card detector contract."""

from pathlib import Path

from .privacy_object_detector import LocalYoloPrivacyDetector, RawPrivacyObject


class CardDetector(LocalYoloPrivacyDetector):
    name = "local_yolo_payment_card"
    supported_classes = {
        "card", "credit_card", "debit_card", "payment_card",
        "credit_card_front", "credit_card_back",
    }

    def __init__(
        self, model_path: Path, confidence_threshold: float, inference_image_size: int = 960,
        tile_size: int = 640, tile_overlap: float = 0.4, tile_inference_image_size: int = 640,
    ) -> None:
        component_classes = {"chip", "text", "band"}
        super().__init__(
            model_path, confidence_threshold, self.supported_classes, inference_image_size,
            tile_size=tile_size, tile_overlap=tile_overlap,
            prediction_classes=self.supported_classes | component_classes,
            tile_inference_image_size=tile_inference_image_size,
        )
        self._component_confirmation = component_classes.issubset(set(self.model_class_names))

    @staticmethod
    def _center_is_inside(component: RawPrivacyObject, candidate: RawPrivacyObject) -> bool:
        center_x = (component.x1 + component.x2) / 2
        center_y = (component.y1 + component.y2) / 2
        return candidate.x1 <= center_x <= candidate.x2 and candidate.y1 <= center_y <= candidate.y2

    def _candidate_is_supported(
        self, candidate: RawPrivacyObject, prediction_objects: list[RawPrivacyObject]
    ) -> bool:
        width, height = candidate.x2 - candidate.x1, candidate.y2 - candidate.y1
        aspect_ratio = max(width, height) / max(1, min(width, height))
        # ISO/IEC 7810 payment cards are roughly 1.59:1. Allow generous
        # perspective/partial-view distortion while rejecting very elongated
        # remotes and phones that the installed model can confuse with a card.
        if aspect_ratio > 2.35:
            return False
        if not self._component_confirmation:
            return True
        inside = [
            item for item in prediction_objects
            if item is not candidate and self._center_is_inside(item, candidate)
        ]
        text_count = sum(item.class_name == "text" for item in inside)
        if candidate.class_name == "credit_card_front":
            return text_count >= 2 or any(item.class_name == "chip" for item in inside)
        if candidate.class_name == "credit_card_back":
            return text_count >= 1 and any(item.class_name == "band" for item in inside)
        return True
