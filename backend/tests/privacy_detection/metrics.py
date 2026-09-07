"""Small ground-truth metric helpers used by the Day 16 validation suite."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import mean


Box = tuple[int, int, int, int]


def intersection_over_union(first: Box, second: Box) -> float:
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    first_area = max(0, first[2] - first[0]) * max(0, first[3] - first[1])
    second_area = max(0, second[2] - second[0]) * max(0, second[3] - second[1])
    return intersection / max(1, first_area + second_area - intersection)


@dataclass(frozen=True)
class DetectionMetrics:
    true_positives: int
    false_positives: int
    false_negatives: int
    matched_ious: tuple[float, ...] = ()

    @property
    def precision(self) -> float | None:
        denominator = self.true_positives + self.false_positives
        return self.true_positives / denominator if denominator else None

    @property
    def recall(self) -> float | None:
        denominator = self.true_positives + self.false_negatives
        return self.true_positives / denominator if denominator else None

    @property
    def f1(self) -> float | None:
        precision, recall = self.precision, self.recall
        if precision is None or recall is None or precision + recall == 0:
            return None if precision is None or recall is None else 0.0
        return 2 * precision * recall / (precision + recall)

    @property
    def average_iou(self) -> float | None:
        return mean(self.matched_ious) if self.matched_ious else None


def match_boxes(truth: list[Box], predicted: list[Box], minimum_iou: float = 0.3) -> DetectionMetrics:
    """Greedily match predictions to distinct labels by highest IoU."""
    candidates = sorted(
        (
            (intersection_over_union(expected, actual), truth_index, prediction_index)
            for truth_index, expected in enumerate(truth)
            for prediction_index, actual in enumerate(predicted)
        ),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    ious: list[float] = []
    for iou, truth_index, prediction_index in candidates:
        if iou < minimum_iou:
            break
        if truth_index in matched_truth or prediction_index in matched_predictions:
            continue
        matched_truth.add(truth_index)
        matched_predictions.add(prediction_index)
        ious.append(iou)
    return DetectionMetrics(
        true_positives=len(ious),
        false_positives=len(predicted) - len(ious),
        false_negatives=len(truth) - len(ious),
        matched_ious=tuple(ious),
    )


def combine(metrics: list[DetectionMetrics]) -> DetectionMetrics:
    return DetectionMetrics(
        true_positives=sum(item.true_positives for item in metrics),
        false_positives=sum(item.false_positives for item in metrics),
        false_negatives=sum(item.false_negatives for item in metrics),
        matched_ious=tuple(value for item in metrics for value in item.matched_ious),
    )
