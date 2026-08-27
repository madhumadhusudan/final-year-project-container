"""Optional Ultralytics YOLO adapter for base or custom weights."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.schemas import RawDetection

from .base import Detector


class YoloDetector(Detector):
    """Detects the COCO person class; custom weights can add privacy classes later."""

    name = "ultralytics_yolo"

    def __init__(self, weights_path: Path) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "DETECTOR_BACKEND=yolo requires the optional 'ultralytics' package"
            ) from exc
        if not weights_path.is_file():
            raise RuntimeError(f"YOLO weights were not found at {weights_path}")
        self._model = YOLO(str(weights_path))

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        predictions = self._model.predict(image_bgr, classes=[0], verbose=False)
        results: list[RawDetection] = []
        for prediction in predictions:
            if prediction.boxes is None:
                continue
            for xyxy, confidence in zip(
                prediction.boxes.xyxy.cpu().tolist(),
                prediction.boxes.conf.cpu().tolist(),
                strict=False,
            ):
                x1, y1, x2, y2 = (round(value) for value in xyxy)
                results.append(
                    RawDetection(
                        category="person",
                        confidence=float(confidence),
                        x=x1,
                        y=y1,
                        width=max(0, x2 - x1),
                        height=max(0, y2 - y1),
                        source=self.name,
                    )
                )
        return results
