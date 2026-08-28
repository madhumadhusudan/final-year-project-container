"""Explainable, geometry-only main-subject analysis (no identity recognition)."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import Settings
from app.schemas import DetectionResult, FaceResult, MainSubjectDetails


@dataclass(frozen=True)
class SubjectCandidate:
    face: FaceResult
    matched_person_id: int | None
    size_score: float
    center_score: float
    person_context_score: float
    subject_score: float


class MainSubjectAnalyzer:
    def __init__(self, app_settings: Settings) -> None:
        self._settings = app_settings

    @staticmethod
    def _match_person(face: FaceResult, people: list[DetectionResult]) -> int | None:
        x, y = face.center.x, face.center.y
        containing = [
            person for person in people
            if person.bounding_box.x1 <= x <= person.bounding_box.x2
            and person.bounding_box.y1 <= y <= person.bounding_box.y2
        ]
        if not containing:
            return None
        # The smallest containing box is normally the least ambiguous match.
        matched = min(
            containing,
            key=lambda person: (person.bounding_box.x2 - person.bounding_box.x1)
            * (person.bounding_box.y2 - person.bounding_box.y1),
        )
        return matched.id

    def analyze(self, faces: list[FaceResult], detections: list[DetectionResult]) -> MainSubjectDetails:
        if not faces:
            return MainSubjectDetails(status="not_found", reason="No face was available for context analysis.")

        people = [detection for detection in detections if detection.class_name == "person"]
        largest_area = max(face.area for face in faces)
        candidates: list[SubjectCandidate] = []
        for face in faces:
            matched_person_id = self._match_person(face, people)
            size_score = face.area / largest_area
            # Day 5 distance uses the full image diagonal; 0.5 is the farthest
            # possible center-to-center distance in that normalization.
            center_score = max(0.0, 1.0 - min(1.0, face.distance_from_image_center / 0.5))
            person_score = 1.0 if matched_person_id is not None else 0.0
            score = (
                self._settings.subject_size_weight * size_score
                + self._settings.subject_center_weight * center_score
                + self._settings.subject_confidence_weight * face.confidence
                + self._settings.subject_person_weight * person_score
            )
            candidates.append(SubjectCandidate(
                face, matched_person_id, size_score, center_score, person_score, round(score, 6)
            ))

        candidates.sort(key=lambda candidate: candidate.subject_score, reverse=True)
        winner = candidates[0]
        score_gap = winner.subject_score - candidates[1].subject_score if len(candidates) > 1 else 1.0
        threshold = (
            self._settings.subject_single_threshold if len(candidates) == 1
            else self._settings.subject_score_threshold
        )
        if winner.subject_score < threshold:
            status, reason = "uncertain", "No face reached the minimum subject score."
        elif len(candidates) > 1 and score_gap < self._settings.subject_ambiguity_margin:
            status, reason = "uncertain", "The leading faces have similarly strong subject scores."
        else:
            status, reason = "identified", "Selected using face prominence, centrality, confidence, and person context."

        if status == "identified":
            for face in faces:
                face.role = "main_subject" if face.face_id == winner.face.face_id else "background_face"
                face.matched_person_id = self._match_person(face, people)
        else:
            for face in faces:
                face.role = "unclassified"
                face.matched_person_id = self._match_person(face, people)

        return MainSubjectDetails(
            status=status,
            face_id=winner.face.face_id if status == "identified" else None,
            matched_person_id=winner.matched_person_id if status == "identified" else None,
            subject_score=winner.subject_score,
            size_score=round(winner.size_score, 6),
            center_score=round(winner.center_score, 6),
            confidence_score=winner.face.confidence,
            person_context_score=winner.person_context_score,
            face_area_ratio=winner.face.area_ratio,
            score_gap=round(score_gap, 6) if len(candidates) > 1 else None,
            reason=reason,
        )
