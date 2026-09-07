"""Real local QR-code and one-dimensional barcode detection with OpenCV."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class RawCodeDetection:
    polygon: tuple[tuple[float, float], ...]
    x1: int
    y1: int
    x2: int
    y2: int
    payload: str | None
    format: str | None = None


class CodeDetector:
    """Keep decoded payloads transient; callers must return only classified/masked data."""

    qr_name = "opencv_qrcode_detector"
    barcode_name = "opencv_barcode_detector"
    # OpenCV's BarcodeDetector documentation limits decoding to these 1-D families.
    supported_barcode_formats = ("EAN-8", "EAN-13", "UPC-A", "UPC-E")
    supports_exhaustive_qr = True

    def __init__(self) -> None:
        try:
            self._qr = cv2.QRCodeDetector()
        except (AttributeError, cv2.error):
            self._qr = None
        try:
            barcode_type = getattr(cv2, "barcode_BarcodeDetector")
            self._barcode = barcode_type()
        except (AttributeError, cv2.error):
            self._barcode = None

    @property
    def qr_available(self) -> bool:
        return self._qr is not None

    @property
    def barcode_available(self) -> bool:
        return self._barcode is not None

    @staticmethod
    def _polygons(points: object) -> list[np.ndarray]:
        if points is None:
            return []
        values = np.asarray(points, dtype=np.float64)
        if values.size == 0 or values.size % 8:
            return []
        try:
            return [polygon for polygon in values.reshape(-1, 4, 2)]
        except ValueError:
            return []

    @staticmethod
    def _result(
        polygon: np.ndarray, payload: str | None, code_format: str | None,
        image_width: int, image_height: int,
    ) -> RawCodeDetection | None:
        if polygon.shape != (4, 2) or not np.isfinite(polygon).all():
            return None
        safe_points = tuple(
            (
                max(0.0, min(float(point[0]), float(image_width))),
                max(0.0, min(float(point[1]), float(image_height))),
            )
            for point in polygon
        )
        xs, ys = [point[0] for point in safe_points], [point[1] for point in safe_points]
        x1, y1 = max(0, math.floor(min(xs))), max(0, math.floor(min(ys)))
        x2, y2 = min(image_width, math.ceil(max(xs))), min(image_height, math.ceil(max(ys)))
        if x2 <= x1 or y2 <= y1:
            return None
        return RawCodeDetection(
            polygon=safe_points, x1=x1, y1=y1, x2=x2, y2=y2,
            payload=payload if payload else None,
            format=code_format.strip().replace("_", "-") if code_format else None,
        )

    def _detect_qr_once(self, image_bgr: np.ndarray) -> list[RawCodeDetection]:
        if self._qr is None:
            raise RuntimeError("Local QR detector unavailable.")
        height, width = image_bgr.shape[:2]
        try:
            detected, points = self._qr.detectMulti(image_bgr)
            if not detected or not self._polygons(points):
                detected, points = self._qr.detect(image_bgr)
            if not detected:
                return []
            polygons = self._polygons(points)
            decoded: tuple[str, ...] = ()
            try:
                _, decoded_values, _ = self._qr.decodeMulti(image_bgr, np.asarray(points))
                decoded = tuple(decoded_values or ())
            except (cv2.error, UnicodeDecodeError, ValueError, TypeError):
                decoded = ()
        except cv2.error as exc:
            raise RuntimeError("Local QR detection failed.") from exc
        results = []
        for index, polygon in enumerate(polygons):
            item = self._result(
                polygon, decoded[index] if index < len(decoded) else None, None, width, height,
            )
            if item is not None:
                results.append(item)
        return results

    def detect_qr(self, image_bgr: np.ndarray, exhaustive: bool = False) -> list[RawCodeDetection]:
        """Detect locally, with a bounded upscale retry for small codes when requested.

        Responsive live/video callers retain the single-pass default. Balanced and
        Accuracy still-image analysis can opt into the retry when the native pass
        finds nothing. Payloads remain transient in both paths.
        """
        results = self._detect_qr_once(image_bgr)
        if results or not exhaustive:
            return results
        height, width = image_bgr.shape[:2]
        pixel_count = max(1, width * height)
        scale = min(2.0, (12_000_000 / pixel_count) ** 0.5)
        if scale < 1.25:
            return results
        enlarged = cv2.resize(image_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        enlarged_results = self._detect_qr_once(enlarged)
        recovered: list[RawCodeDetection] = []
        for item in enlarged_results:
            polygon = np.asarray(
                [(point[0] / scale, point[1] / scale) for point in item.polygon], dtype=np.float64,
            )
            mapped = self._result(polygon, item.payload, item.format, width, height)
            if mapped is not None:
                recovered.append(mapped)
        return recovered

    def detect_barcodes(self, image_bgr: np.ndarray) -> list[RawCodeDetection]:
        if self._barcode is None:
            raise RuntimeError("Local barcode detector unavailable.")
        height, width = image_bgr.shape[:2]
        try:
            detected, points = self._barcode.detect(image_bgr)
            if not detected:
                return []
            polygons = self._polygons(points)
            decoded: tuple[str, ...] = ()
            formats: tuple[str, ...] = ()
            try:
                _, decoded_values, decoded_types = self._barcode.decodeWithType(
                    image_bgr, np.asarray(points)
                )
                decoded = tuple(decoded_values or ())
                formats = tuple(decoded_types or ())
            except (cv2.error, UnicodeDecodeError, ValueError, TypeError):
                pass
        except cv2.error as exc:
            raise RuntimeError("Local barcode detection failed.") from exc
        results = []
        for index, polygon in enumerate(polygons):
            item = self._result(
                polygon,
                decoded[index] if index < len(decoded) else None,
                formats[index] if index < len(formats) else None,
                width,
                height,
            )
            if item is not None:
                results.append(item)
        return results
