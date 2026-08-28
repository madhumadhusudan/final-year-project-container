"""Shared adapter for optional dedicated local Ultralytics privacy models."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class RawPrivacyObject:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class LocalYoloPrivacyDetector:
    def __init__(self, model_path: Path, confidence_threshold: float, allowed_classes: set[str]) -> None:
        if not model_path.is_file():
            raise RuntimeError("Dedicated model weights are unavailable.")
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError("Dedicated model weights could not be loaded.") from exc
        labels = {str(label).lower().replace(" ", "_") for label in self._model.names.values()}
        if not labels.intersection(allowed_classes):
            raise RuntimeError("Dedicated model class labels are not supported.")
        self._allowed_classes = allowed_classes
        self._confidence_threshold = confidence_threshold

    def detect(self, image_bgr: np.ndarray) -> list[RawPrivacyObject]:
        try:
            predictions = self._model.predict(image_bgr, conf=self._confidence_threshold, verbose=False)
        except Exception as exc:
            raise RuntimeError("Dedicated privacy-object inference failed.") from exc
        results = []
        for prediction in predictions:
            if prediction.boxes is None:
                continue
            for xyxy, confidence, class_id in zip(
                prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.conf.cpu().tolist(),
                prediction.boxes.cls.cpu().tolist(), strict=False,
            ):
                label = str(prediction.names[int(class_id)]).lower().replace(" ", "_")
                if label not in self._allowed_classes:
                    continue
                x1, y1, x2, y2 = (round(value) for value in xyxy)
                results.append(RawPrivacyObject(label, float(confidence), x1, y1, x2, y2))
        return results
