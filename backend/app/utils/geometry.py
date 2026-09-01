"""Shared, side-effect-free geometry helpers for detected regions."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def box_coordinates(value: object) -> tuple[float, float, float, float] | None:
    """Read finite x1/y1/x2/y2 coordinates from a box or result object."""
    box = getattr(value, "bounding_box", value)
    if isinstance(box, Mapping):
        raw = tuple(box.get(key) for key in ("x1", "y1", "x2", "y2"))
    elif isinstance(box, Sequence) and not isinstance(box, (str, bytes)) and len(box) == 4:
        raw = tuple(box)
    else:
        raw = tuple(getattr(box, key, None) for key in ("x1", "y1", "x2", "y2"))
    try:
        coordinates = tuple(float(item) for item in raw)
    except (TypeError, ValueError):
        return None
    if len(coordinates) != 4 or not all(math.isfinite(item) for item in coordinates):
        return None
    x1, y1, x2, y2 = coordinates
    return coordinates if x2 > x1 and y2 > y1 else None


def area(value: object) -> float:
    coordinates = box_coordinates(value)
    if coordinates is None:
        return 0.0
    x1, y1, x2, y2 = coordinates
    return (x2 - x1) * (y2 - y1)


def intersection_area(first: object, second: object) -> float:
    first_box, second_box = box_coordinates(first), box_coordinates(second)
    if first_box is None or second_box is None:
        return 0.0
    ax1, ay1, ax2, ay2 = first_box
    bx1, by1, bx2, by2 = second_box
    return max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0.0, min(ay2, by2) - max(ay1, by1)
    )


def overlap_ratio(candidate: object, container: object) -> float:
    """Return the fraction of the candidate covered by the container."""
    candidate_area = area(candidate)
    return intersection_area(candidate, container) / candidate_area if candidate_area else 0.0


def intersection_over_union(first: object, second: object) -> float:
    intersection = intersection_area(first, second)
    union = area(first) + area(second) - intersection
    return intersection / union if union else 0.0


def center_inside(candidate: object, container: object) -> bool:
    candidate_box, container_box = box_coordinates(candidate), box_coordinates(container)
    if candidate_box is None or container_box is None:
        return False
    x1, y1, x2, y2 = candidate_box
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    ax1, ay1, ax2, ay2 = container_box
    return ax1 <= cx <= ax2 and ay1 <= cy <= ay2
