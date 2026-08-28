"""Dedicated, local face detection using OpenCV YuNet."""

from __future__ import annotations

from pathlib import Path
from threading import Lock

import cv2
import numpy as np


class RawFace:
    def __init__(self, confidence: float, x1: int, y1: int, x2: int, y2: int) -> None:
        self.confidence = confidence
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2


class UnavailableFaceDetector:
    """Keeps object analysis usable when the local face model cannot initialize."""

    name = "opencv_yunet_unavailable"

    def detect(self, _image_bgr: np.ndarray) -> list[RawFace]:
        raise RuntimeError("The local face detection model is unavailable.")


class FaceDetector:
    """Loads the bundled model once and returns boxes in original-image pixels."""

    name = "opencv_yunet"

    def __init__(self, model_path: Path, confidence_threshold: float) -> None:
        if not model_path.is_file():
            raise RuntimeError("The local YuNet face detection model is unavailable.")
        try:
            self._model = cv2.FaceDetectorYN.create(
                str(model_path), "", (320, 320), confidence_threshold, 0.3, 5000
            )
        except Exception as exc:
            raise RuntimeError("The local YuNet face detection model could not be loaded.") from exc
        self._lock = Lock()

    def detect(self, image_bgr: np.ndarray) -> list[RawFace]:
        try:
            height, width = image_bgr.shape[:2]
            # YuNet requires the current input size. The lock protects the reused
            # model when FastAPI handles multiple analysis requests concurrently.
            with self._lock:
                self._model.setInputSize((width, height))
                _status, detections = self._model.detect(image_bgr)
        except Exception as exc:
            raise RuntimeError("Local face detection failed.") from exc

        faces = []
        if detections is None:
            return faces
        for detection in detections:
            x, y, box_width, box_height = detection[:4]
            confidence = float(detection[-1])
            x1, y1 = round(float(x)), round(float(y))
            faces.append(RawFace(confidence, x1, y1, round(float(x + box_width)), round(float(y + box_height))))
        return faces
