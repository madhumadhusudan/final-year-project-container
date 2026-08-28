"""Shared adapter for optional dedicated local Ultralytics privacy models."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock

import numpy as np


@dataclass(frozen=True)
class RawPrivacyObject:
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass(frozen=True)
class PrivacyDetectionRun:
    detections: list[RawPrivacyObject]
    raw_detection_count: int
    accepted_detection_count: int
    inference_time_ms: int
    tiles_processed: int = 1


def normalize_model_label(label: object) -> str:
    """Normalize model metadata without changing the trained class semantics."""
    value = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", str(label).strip())
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


class LocalYoloPrivacyDetector:
    def __init__(
        self, model_path: Path, confidence_threshold: float, allowed_classes: set[str],
        inference_image_size: int = 640, tile_size: int | None = None, tile_overlap: float = 0.2,
        prediction_classes: set[str] | None = None, tile_inference_image_size: int | None = None,
    ) -> None:
        if not model_path.is_file():
            raise RuntimeError("Dedicated model weights are unavailable.")
        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError("Dedicated model weights could not be loaded.") from exc
        raw_names = self._model.names
        name_items = raw_names.items() if isinstance(raw_names, dict) else enumerate(raw_names)
        self._class_names = {int(class_id): normalize_model_label(label) for class_id, label in name_items}
        labels = set(self._class_names.values())
        if not labels.intersection(allowed_classes):
            raise RuntimeError("Dedicated model class labels are not supported.")
        self._allowed_classes = frozenset(allowed_classes)
        self._allowed_class_ids = [
            class_id for class_id, label in self._class_names.items() if label in self._allowed_classes
        ]
        predicted_labels = prediction_classes or allowed_classes
        self._prediction_class_ids = [
            class_id for class_id, label in self._class_names.items() if label in predicted_labels
        ]
        self._confidence_threshold = confidence_threshold
        self._inference_image_size = inference_image_size
        self._model_name = model_path.name
        self._tile_size = tile_size
        self._tile_overlap = max(0.0, min(tile_overlap, 0.5))
        self._tile_inference_image_size = tile_inference_image_size or inference_image_size
        self._lock = Lock()

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def model_class_names(self) -> list[str]:
        return [self._class_names[index] for index in sorted(self._class_names)]

    @property
    def confidence_threshold(self) -> float:
        return self._confidence_threshold

    @property
    def inference_image_size(self) -> int:
        return self._inference_image_size

    @property
    def tile_size(self) -> int | None:
        return self._tile_size

    @property
    def tile_inference_image_size(self) -> int:
        return self._tile_inference_image_size

    @staticmethod
    def _tile_positions(length: int, tile_size: int, overlap: float) -> list[int]:
        if length <= tile_size:
            return [0]
        step = max(1, round(tile_size * (1 - overlap)))
        positions = list(range(0, length - tile_size + 1, step))
        final = length - tile_size
        if positions[-1] != final:
            positions.append(final)
        return positions

    def _sources(self, image_bgr: np.ndarray) -> tuple[list[np.ndarray], list[tuple[int, int]]]:
        height, width = image_bgr.shape[:2]
        if self._tile_size is None or max(height, width) < round(self._tile_size * 1.5):
            return [image_bgr], [(0, 0)]
        sources, offsets = [], []
        for y in self._tile_positions(height, self._tile_size, self._tile_overlap):
            for x in self._tile_positions(width, self._tile_size, self._tile_overlap):
                sources.append(image_bgr[y:min(y + self._tile_size, height), x:min(x + self._tile_size, width)])
                offsets.append((x, y))
        return sources, offsets

    @staticmethod
    def _touches_tile_edge(
        box: tuple[int, int, int, int], offset: tuple[int, int], tile_shape: tuple[int, int],
        image_shape: tuple[int, int], margin: int = 3,
    ) -> bool:
        x1, y1, x2, y2 = box
        tile_height, tile_width = tile_shape
        # Tile-boundary boxes are commonly clipped fragments and image-edge artifacts.
        # A fully visible card will occur in a neighbouring overlapping tile.
        return x1 <= margin or y1 <= margin or x2 >= tile_width - margin or y2 >= tile_height - margin

    def _candidate_is_supported(
        self, candidate: RawPrivacyObject, prediction_objects: list[RawPrivacyObject]
    ) -> bool:
        """Subclasses may require model-native contextual evidence for a candidate."""
        return True

    @staticmethod
    def _intersection_over_union(first: RawPrivacyObject, second: RawPrivacyObject) -> float:
        x1, y1 = max(first.x1, second.x1), max(first.y1, second.y1)
        x2, y2 = min(first.x2, second.x2), min(first.y2, second.y2)
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        if intersection == 0:
            return 0.0
        first_area = (first.x2 - first.x1) * (first.y2 - first.y1)
        second_area = (second.x2 - second.x1) * (second.y2 - second.y1)
        return intersection / (first_area + second_area - intersection)

    @classmethod
    def _deduplicate(cls, detections: list[RawPrivacyObject], threshold: float = 0.5) -> list[RawPrivacyObject]:
        accepted: list[RawPrivacyObject] = []
        for candidate in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if any(
                candidate.class_name == existing.class_name
                and cls._intersection_over_union(candidate, existing) >= threshold
                for existing in accepted
            ):
                continue
            accepted.append(candidate)
        return accepted

    def detect(self, image_bgr: np.ndarray) -> list[RawPrivacyObject]:
        return self.detect_with_diagnostics(image_bgr).detections

    def detect_with_diagnostics(self, image_bgr: np.ndarray) -> PrivacyDetectionRun:
        started = time.perf_counter()
        sources, offsets = self._sources(image_bgr)
        prediction_size = (
            self._tile_inference_image_size if len(sources) > 1 else self._inference_image_size
        )
        try:
            with self._lock:
                predictions = self._model.predict(
                    sources if len(sources) > 1 else sources[0],
                    conf=self._confidence_threshold,
                    imgsz=prediction_size,
                    classes=self._prediction_class_ids,
                    verbose=False,
                )
        except Exception as exc:
            raise RuntimeError("Dedicated privacy-object inference failed.") from exc
        results = []
        raw_detection_count = 0
        image_height, image_width = image_bgr.shape[:2]
        for prediction, source, (offset_x, offset_y) in zip(predictions, sources, offsets, strict=False):
            if prediction.boxes is None:
                continue
            raw_detection_count += len(prediction.boxes)
            prediction_objects = []
            for xyxy, confidence, class_id in zip(
                prediction.boxes.xyxy.cpu().tolist(), prediction.boxes.conf.cpu().tolist(),
                prediction.boxes.cls.cpu().tolist(), strict=False,
                ):
                label = normalize_model_label(prediction.names[int(class_id)])
                x1, y1, x2, y2 = (round(value) for value in xyxy)
                prediction_objects.append(RawPrivacyObject(label, float(confidence), x1, y1, x2, y2))
            for candidate in prediction_objects:
                if candidate.class_name not in self._allowed_classes:
                    continue
                x1, y1, x2, y2 = candidate.x1, candidate.y1, candidate.x2, candidate.y2
                if len(sources) > 1 and self._touches_tile_edge(
                    (x1, y1, x2, y2), (offset_x, offset_y), source.shape[:2], image_bgr.shape[:2]
                ):
                    continue
                if not self._candidate_is_supported(candidate, prediction_objects):
                    continue
                x1, x2 = max(0, x1 + offset_x), min(image_width, x2 + offset_x)
                y1, y2 = max(0, y1 + offset_y), min(image_height, y2 + offset_y)
                if x2 <= x1 or y2 <= y1:
                    continue
                results.append(RawPrivacyObject(candidate.class_name, candidate.confidence, x1, y1, x2, y2))
        results = self._deduplicate(results)
        return PrivacyDetectionRun(
            detections=results,
            raw_detection_count=raw_detection_count,
            accepted_detection_count=len(results),
            inference_time_ms=max(0, round((time.perf_counter() - started) * 1000)),
            tiles_processed=len(sources),
        )
