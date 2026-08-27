"""Local OpenCV model adapters for faces and people."""

from __future__ import annotations

import math

import cv2
import numpy as np

from app.schemas import RawDetection

from .base import Detector


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, value))))


class OpenCVFaceDetector(Detector):
    """Uses OpenCV's bundled pretrained frontal-face cascade."""

    name = "opencv_haar_face"

    def __init__(self) -> None:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            raise RuntimeError("OpenCV face model could not be loaded")

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        min_side = max(20, min(image_bgr.shape[:2]) // 40)

        rectangles, _reject_levels, level_weights = self._cascade.detectMultiScale3(
            gray,
            scaleFactor=1.08,
            minNeighbors=5,
            minSize=(min_side, min_side),
            flags=cv2.CASCADE_SCALE_IMAGE,
            outputRejectLevels=True,
        )

        results: list[RawDetection] = []
        for rectangle, level_weight in zip(rectangles, level_weights, strict=False):
            x, y, width, height = (int(value) for value in rectangle)
            # Cascade level weights are margins rather than calibrated probabilities.
            confidence = _sigmoid(float(level_weight) - 1.5)
            results.append(
                RawDetection(
                    category="face",
                    confidence=confidence,
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    source=self.name,
                )
            )
        return results


class OpenCVPersonDetector(Detector):
    """Uses OpenCV's pretrained HOG/SVM pedestrian model."""

    name = "opencv_hog_person"

    def __init__(self) -> None:
        self._hog = cv2.HOGDescriptor()
        self._hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        height, width = image_bgr.shape[:2]
        scale = min(1.0, 960.0 / max(height, width))
        inference_image = image_bgr
        if scale < 1.0:
            inference_image = cv2.resize(
                image_bgr,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )

        rectangles, weights = self._hog.detectMultiScale(
            inference_image,
            winStride=(8, 8),
            padding=(8, 8),
            scale=1.05,
        )
        if len(rectangles) == 0:
            return []

        boxes: list[list[int]] = []
        scores: list[float] = []
        inverse_scale = 1.0 / scale
        for rectangle, weight in zip(rectangles, weights, strict=False):
            x, y, box_width, box_height = (int(round(float(value) * inverse_scale)) for value in rectangle)
            boxes.append([x, y, box_width, box_height])
            scores.append(_sigmoid(float(weight)))

        retained = cv2.dnn.NMSBoxes(boxes, scores, score_threshold=0.01, nms_threshold=0.35)
        return [
            RawDetection(
                category="person",
                confidence=scores[index],
                x=boxes[index][0],
                y=boxes[index][1],
                width=boxes[index][2],
                height=boxes[index][3],
                source=self.name,
            )
            for index in np.array(retained).reshape(-1).tolist()
        ]
