"""Reusable detector-sized image views and original-coordinate conversion."""

from __future__ import annotations

import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class AnalysisView:
    pixels_bgr: np.ndarray
    scale_x: float
    scale_y: float


def analysis_to_original_coordinates(
    box: tuple[float, float, float, float], scale_x: float, scale_y: float,
) -> tuple[int, int, int, int]:
    return (
        round(box[0] * scale_x), round(box[1] * scale_y),
        round(box[2] * scale_x), round(box[3] * scale_y),
    )


class AnalysisImageCache:
    """Create each required downscaled BGR representation at most once."""

    def __init__(self, original: np.ndarray) -> None:
        self.original = original
        self._views: dict[int, AnalysisView] = {}
        self.preprocess_ms = 0

    def get(self, max_side: int) -> AnalysisView:
        height, width = self.original.shape[:2]
        if max(height, width) <= max_side:
            return AnalysisView(self.original, 1.0, 1.0)
        if max_side in self._views:
            return self._views[max_side]
        started = time.perf_counter()
        ratio = max_side / max(height, width)
        target_width, target_height = max(1, round(width * ratio)), max(1, round(height * ratio))
        pixels = cv2.resize(self.original, (target_width, target_height), interpolation=cv2.INTER_AREA)
        self.preprocess_ms += max(0, round((time.perf_counter() - started) * 1000))
        view = AnalysisView(pixels, width / target_width, height / target_height)
        self._views[max_side] = view
        return view
