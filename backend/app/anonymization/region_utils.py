"""Safe bounding-box operations for selective image anonymization."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Region:
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1


def _coordinates(box: object) -> tuple[object, object, object, object] | None:
    if isinstance(box, Mapping):
        return tuple(box.get(key) for key in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
    if isinstance(box, Sequence) and not isinstance(box, (str, bytes)) and len(box) == 4:
        return tuple(box)  # type: ignore[return-value]
    values = tuple(getattr(box, key, None) for key in ("x1", "y1", "x2", "y2"))
    return values if all(value is not None for value in values) else None


def sanitize_region(box: object, image_width: int, image_height: int) -> Region | None:
    """Convert, clamp, and validate a box before it is used for array slicing."""
    if image_width <= 0 or image_height <= 0:
        return None
    values = _coordinates(box)
    if values is None:
        return None
    try:
        numeric = tuple(float(value) for value in values)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) for value in numeric):
        return None
    x1 = max(0, min(math.floor(numeric[0]), image_width))
    y1 = max(0, min(math.floor(numeric[1]), image_height))
    x2 = max(0, min(math.ceil(numeric[2]), image_width))
    y2 = max(0, min(math.ceil(numeric[3]), image_height))
    return Region(x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def pad_region(region: Region, ratio: float, image_width: int, image_height: int) -> Region | None:
    """Expand a valid region by a percentage on each edge, then clamp it."""
    if not math.isfinite(ratio):
        return None
    ratio = max(0.0, min(ratio, 1.0))
    padding_x = math.ceil(region.width * ratio)
    padding_y = math.ceil(region.height * ratio)
    return sanitize_region(
        (region.x1 - padding_x, region.y1 - padding_y, region.x2 + padding_x, region.y2 + padding_y),
        image_width,
        image_height,
    )


def contains_region(container: Region, candidate: Region) -> bool:
    return (
        container.x1 <= candidate.x1
        and container.y1 <= candidate.y1
        and container.x2 >= candidate.x2
        and container.y2 >= candidate.y2
    )


def _should_merge(first: Region, second: Region, adjacency: int) -> bool:
    horizontal_gap = max(0, max(first.x1, second.x1) - min(first.x2, second.x2))
    vertical_gap = max(0, max(first.y1, second.y1) - min(first.y2, second.y2))
    horizontal_overlap = min(first.x2, second.x2) > max(first.x1, second.x1)
    vertical_overlap = min(first.y2, second.y2) > max(first.y1, second.y1)
    overlaps = horizontal_gap == 0 and vertical_gap == 0
    adjacent_in_line = (
        horizontal_gap <= adjacency and vertical_overlap
    ) or (
        vertical_gap <= adjacency and horizontal_overlap
    )
    return overlaps or adjacent_in_line


def merge_regions(regions: Iterable[Region], adjacency: int = 3) -> list[Region]:
    """Merge overlapping or directly adjacent regions until the result is stable."""
    merged = list(regions)
    adjacency = max(0, adjacency)
    changed = True
    while changed:
        changed = False
        output: list[Region] = []
        while merged:
            current = merged.pop(0)
            match_index = next(
                (index for index, other in enumerate(merged) if _should_merge(current, other, adjacency)),
                None,
            )
            if match_index is None:
                output.append(current)
                continue
            other = merged.pop(match_index)
            merged.insert(0, Region(
                min(current.x1, other.x1), min(current.y1, other.y1),
                max(current.x2, other.x2), max(current.y2, other.y2),
            ))
            changed = True
        merged = output
    return merged
