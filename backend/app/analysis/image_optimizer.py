"""Small, reusable image-analysis optimizations with original-coordinate output."""

from __future__ import annotations

import time
from dataclasses import replace

import cv2
import numpy as np

from app.ocr.ocr_service import RawOCRText
from app.utils.analysis_coordinates import AnalysisImageCache, analysis_to_original_coordinates


# 480 px lost small but clearly rendered sensitive text in the Day 16 composite.
# 640 remains bounded while matching the minimum size that retained recall.
OCR_MAX_SIDE = {"fast": 640, "balanced": 640, "accuracy": 1280}
YOLO_IMAGE_SIZE = {"fast": 512, "balanced": 640, "accuracy": 960}
FACE_MAX_SIDE = {"fast": 640, "balanced": 800, "accuracy": 960}
PRIVACY_MAX_SIDE = {"fast": 768, "balanced": 960, "accuracy": 1280}


def scaled_detect(
    detector, image_cache: AnalysisImageCache, max_side: int, method: str = "detect", method_args=(),
):
    """Run one detector on a bounded image and map its boxes back to source pixels."""
    view = image_cache.get(max_side)
    started = time.perf_counter()
    values = getattr(detector, method)(view.pixels_bgr, *method_args)
    elapsed = max(0, round((time.perf_counter() - started) * 1000))
    return scale_values(values, view.scale_x, view.scale_y), elapsed


def scale_values(values, scale_x: float, scale_y: float):
    mapped = []
    for value in values:
        x1, y1, x2, y2 = analysis_to_original_coordinates(
            (value.x1, value.y1, value.x2, value.y2), scale_x, scale_y,
        )
        if hasattr(value, "polygon"):
            polygon = tuple((point[0] * scale_x, point[1] * scale_y) for point in value.polygon)
            mapped.append(replace(value, x1=x1, y1=y1, x2=x2, y2=y2, polygon=polygon))
        elif hasattr(value, "class_id") or hasattr(value, "class_name"):
            mapped.append(replace(value, x1=x1, y1=y1, x2=x2, y2=y2))
        else:
            mapped.append(type(value)(value.confidence, x1, y1, x2, y2))
    return mapped


def text_regions(image_bgr: np.ndarray, *, limit: int = 6) -> list[tuple[int, int, int, int]]:
    """Find conservative text-like ROIs cheaply; no pixels or extracted text are retained."""
    height, width = image_bgr.shape[:2]
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    _, binary = cv2.threshold(gradient, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3)))
    candidates = []
    for contour in cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        if w < 28 or h < 8 or area < 300 or area > width * height * 0.55 or w / max(h, 1) < 1.5:
            continue
        pad_x, pad_y = max(6, round(w * 0.08)), max(5, round(h * 0.25))
        candidates.append((max(0, x - pad_x), max(0, y - pad_y), min(width, x + w + pad_x), min(height, y + h + pad_y)))
    return sorted(candidates, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]), reverse=True)[:limit]


def extract_text_optimized(
    ocr_service, image_cache: AnalysisImageCache, profile: str, primary_boxes=(),
) -> tuple[list[RawOCRText], int]:
    """Use bounded full-frame OCR for accuracy and text ROIs for responsive modes."""
    view = image_cache.get(OCR_MAX_SIDE[profile])
    started = time.perf_counter()
    if profile == "accuracy" or max(view.pixels_bgr.shape[:2]) <= 320:
        regions = [(0, 0, view.pixels_bgr.shape[1], view.pixels_bgr.shape[0])]
    else:
        regions = []
        view_height, view_width = view.pixels_bgr.shape[:2]
        for box in primary_boxes:
            pad_x, pad_y = round((box.x2 - box.x1) * 0.04), round((box.y2 - box.y1) * 0.06)
            regions.append((
                max(0, round((box.x1 - pad_x) / view.scale_x)),
                max(0, round((box.y1 - pad_y) / view.scale_y)),
                min(view_width, round((box.x2 + pad_x) / view.scale_x)),
                min(view_height, round((box.y2 + pad_y) / view.scale_y)),
            ))
        regions.extend(text_regions(view.pixels_bgr, limit=4 if profile == "fast" else 6))
        # Avoid repeat OCR for the same primary/text-like region.
        unique = []
        for candidate in regions:
            area = max(1, (candidate[2] - candidate[0]) * (candidate[3] - candidate[1]))
            duplicate = False
            for kept in unique:
                intersection = max(0, min(candidate[2], kept[2]) - max(candidate[0], kept[0])) * max(
                    0, min(candidate[3], kept[3]) - max(candidate[1], kept[1])
                )
                if intersection / area >= 0.8:
                    duplicate = True
                    break
            if not duplicate and candidate[2] > candidate[0] and candidate[3] > candidate[1]:
                unique.append(candidate)
        regions = unique[:6 if profile == "fast" else 10]
    output: list[RawOCRText] = []
    for x1, y1, x2, y2 in regions:
        crop = view.pixels_bgr[y1:y2, x1:x2]
        for raw in ocr_service.extract_text(crop):
            ox1, oy1, ox2, oy2 = analysis_to_original_coordinates(
                (raw.x1 + x1, raw.y1 + y1, raw.x2 + x1, raw.y2 + y1), view.scale_x, view.scale_y,
            )
            output.append(replace(raw, x1=ox1, y1=oy1, x2=ox2, y2=oy2))
    return output, max(0, round((time.perf_counter() - started) * 1000))
