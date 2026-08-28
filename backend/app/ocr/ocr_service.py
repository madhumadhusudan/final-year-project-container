"""One-time EasyOCR reader lifecycle and original-coordinate text extraction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np

from .text_normalizer import normalize_text


@dataclass(frozen=True)
class RawOCRText:
    raw_text: str
    normalized_text: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


class OCRService:
    name = "easyocr"

    def __init__(self, languages: list[str], model_directory: Path) -> None:
        try:
            import easyocr
            model_directory.mkdir(parents=True, exist_ok=True)
            self._reader = easyocr.Reader(
                languages, gpu=False, model_storage_directory=str(model_directory),
                download_enabled=True, verbose=False,
            )
        except Exception as exc:
            raise RuntimeError("The local OCR engine could not be initialized.") from exc
        self._lock = Lock()

    def extract_text(self, image_bgr: np.ndarray) -> list[RawOCRText]:
        try:
            with self._lock:
                predictions = self._reader.readtext(image_bgr, detail=1, paragraph=False)
        except Exception as exc:
            raise RuntimeError("Local OCR inference failed.") from exc
        results = []
        for polygon, raw_text, confidence in predictions:
            normalized = normalize_text(str(raw_text))
            if not normalized:
                continue
            xs = [float(point[0]) for point in polygon]
            ys = [float(point[1]) for point in polygon]
            results.append(RawOCRText(
                str(raw_text), normalized, float(confidence),
                round(min(xs)), round(min(ys)), round(max(xs)), round(max(ys)),
            ))
        return results


class UnavailableOCRService:
    name = "easyocr_unavailable"

    def extract_text(self, _image_bgr: np.ndarray) -> list[RawOCRText]:
        raise RuntimeError("The local OCR engine is unavailable.")
