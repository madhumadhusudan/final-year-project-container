"""Ultralytics YOLOv8 detector adapter."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import numpy as np

from app.schemas import RawDetection


class YoloDetector:
    """Loads one local YOLO model and returns genuine COCO detections."""

    def __init__(
        self, weights_path: Path, confidence_threshold: float, *, device: str = "cpu",
        inference_image_size: int = 640, max_detections: int = 100,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("The local YOLO detection package is unavailable.") from exc
        if not weights_path.is_file():
            raise RuntimeError("The local YOLO model is unavailable.")
        try:
            self._model = YOLO(str(weights_path))
        except Exception as exc:
            raise RuntimeError("The local YOLO model could not be loaded.") from exc
        self._confidence_threshold = confidence_threshold
        self.device = device
        self.inference_image_size = inference_image_size
        self.max_detections = max_detections
        self._lock = Lock()

    def detect(self, image_bgr: np.ndarray, inference_image_size: int | None = None) -> list[RawDetection]:
        try:
            with self._lock:
                predict_options = {
                    "source": image_bgr, "conf": self._confidence_threshold,
                    "imgsz": inference_image_size or self.inference_image_size,
                    "max_det": self.max_detections, "device": self.device, "verbose": False,
                }
                if self.device == "cuda":
                    predict_options["half"] = True
                predictions = self._model.predict(**predict_options)
        except Exception as exc:
            raise RuntimeError("YOLO inference failed.") from exc

        detections: list[RawDetection] = []
        for prediction in predictions:
            if prediction.boxes is None:
                continue
            names = prediction.names
            for xyxy, confidence, class_id in zip(
                prediction.boxes.xyxy.cpu().tolist(),
                prediction.boxes.conf.cpu().tolist(),
                prediction.boxes.cls.cpu().tolist(),
                strict=False,
            ):
                x1, y1, x2, y2 = (round(value) for value in xyxy)
                numeric_class_id = int(class_id)
                detections.append(
                    RawDetection(
                        class_id=numeric_class_id,
                        class_name=str(names[numeric_class_id]),
                        confidence=float(confidence),
                        x1=x1,
                        y1=y1,
                        x2=x2,
                        y2=y2,
                    )
                )
        return detections
