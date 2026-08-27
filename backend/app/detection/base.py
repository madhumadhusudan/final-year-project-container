"""Detector contracts that isolate model-specific inference."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from app.schemas import RawDetection


class Detector(ABC):
    name: str

    @abstractmethod
    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        """Return genuine model predictions mapped to the supplied image pixels."""


class CompositeDetector(Detector):
    name = "composite"

    def __init__(self, detectors: list[Detector]) -> None:
        self._detectors = detectors

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        detections: list[RawDetection] = []
        for detector in self._detectors:
            detections.extend(detector.detect(image_bgr))
        return detections
