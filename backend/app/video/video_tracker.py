"""Category-aware temporal association without biometric identification."""

from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot


@dataclass
class TrackedRegion:
    track_id: str
    category: str
    box: tuple[float, float, float, float]
    confidence: float | None
    role: str | None
    last_seen_frame: int
    miss_count: int = 0
    age: int = 1
    velocity: tuple[float, float, float, float] = field(default_factory=lambda: (0.0, 0.0, 0.0, 0.0))

    def predicted_box(self, frame_index: int) -> tuple[float, float, float, float]:
        delta = max(0, frame_index - self.last_seen_frame)
        return tuple(value + speed * delta for value, speed in zip(self.box, self.velocity))


def _area(box: tuple[float, float, float, float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def intersection_over_union(first, second) -> float:
    intersection = max(0.0, min(first[2], second[2]) - max(first[0], second[0])) * max(
        0.0, min(first[3], second[3]) - max(first[1], second[1])
    )
    return intersection / max(1.0, _area(first) + _area(second) - intersection)


def _association_score(first, second) -> float | None:
    iou = intersection_over_union(first, second)
    first_center = ((first[0] + first[2]) / 2, (first[1] + first[3]) / 2)
    second_center = ((second[0] + second[2]) / 2, (second[1] + second[3]) / 2)
    scale = max(1.0, hypot(first[2] - first[0], first[3] - first[1]), hypot(second[2] - second[0], second[3] - second[1]))
    center_distance = hypot(first_center[0] - second_center[0], first_center[1] - second_center[1]) / scale
    size_similarity = min(_area(first), _area(second)) / max(1.0, max(_area(first), _area(second)))
    if iou < 0.08 and (center_distance > 0.85 or size_similarity < 0.25):
        return None
    return iou * 0.65 + max(0.0, 1.0 - center_distance) * 0.25 + size_similarity * 0.10


class RegionTracker:
    """Assign session-only IDs and predict boxes between sampled detections."""

    def __init__(self, expiry_frames: int = 18) -> None:
        self.expiry_frames = expiry_frames
        self.tracks: dict[str, TrackedRegion] = {}
        self._counters: dict[str, int] = {}

    def _new_id(self, category: str) -> str:
        self._counters[category] = self._counters.get(category, 0) + 1
        prefix = {"license_plate": "plate", "identity_document": "document", "payment_card": "card"}.get(category, category)
        return f"{prefix}_track_{self._counters[category]}"

    def update(self, detections: list[dict], observed_categories: set[str], frame_index: int) -> list[TrackedRegion]:
        candidates: list[tuple[float, str, int]] = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                if detection["category"] != track.category:
                    continue
                score = _association_score(track.predicted_box(frame_index), detection["box"])
                if score is not None:
                    candidates.append((score, track_id, index))
        matched_tracks: set[str] = set()
        matched_detections: set[int] = set()
        for _, track_id, index in sorted(candidates, reverse=True):
            if track_id in matched_tracks or index in matched_detections:
                continue
            track = self.tracks[track_id]
            detection = detections[index]
            elapsed = max(1, frame_index - track.last_seen_frame)
            measured = tuple((new - old) / elapsed for new, old in zip(detection["box"], track.box))
            track.velocity = tuple(previous * 0.45 + current * 0.55 for previous, current in zip(track.velocity, measured))
            track.box = detection["box"]
            track.confidence = detection.get("confidence")
            track.role = detection.get("role")
            track.last_seen_frame = frame_index
            track.miss_count = 0
            track.age += elapsed
            matched_tracks.add(track_id)
            matched_detections.add(index)

        for index, detection in enumerate(detections):
            if index in matched_detections:
                continue
            track_id = self._new_id(detection["category"])
            self.tracks[track_id] = TrackedRegion(
                track_id=track_id, category=detection["category"], box=detection["box"],
                confidence=detection.get("confidence"), role=detection.get("role"), last_seen_frame=frame_index,
            )

        for track in self.tracks.values():
            if track.category in observed_categories and track.track_id not in matched_tracks and track.last_seen_frame != frame_index:
                track.miss_count += 1
        expired = [track_id for track_id, track in self.tracks.items() if frame_index - track.last_seen_frame > self.expiry_frames]
        for track_id in expired:
            del self.tracks[track_id]
        return self.active(frame_index)

    def active(self, frame_index: int) -> list[TrackedRegion]:
        return [track for track in self.tracks.values() if frame_index - track.last_seen_frame <= self.expiry_frames]
