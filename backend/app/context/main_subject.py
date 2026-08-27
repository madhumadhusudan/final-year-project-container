"""Explainable main-subject classification for detected faces and people."""

from __future__ import annotations

import math
from dataclasses import replace

from app.config import Settings
from app.schemas import RawDetection


def _relative_area(detection: RawDetection, image_width: int, image_height: int) -> float:
    return (detection.width * detection.height) / (image_width * image_height)


def _center_proximity(detection: RawDetection, image_width: int, image_height: int) -> float:
    center_x = (detection.x + detection.width / 2) / image_width
    center_y = (detection.y + detection.height / 2) / image_height
    distance = math.hypot(center_x - 0.5, center_y - 0.5)
    return max(0.0, 1.0 - distance / math.sqrt(0.5))


def _subject_score(
    detection: RawDetection,
    image_width: int,
    image_height: int,
    area_threshold: float,
) -> float:
    area_score = min(1.0, _relative_area(detection, image_width, image_height) / area_threshold)
    center_score = _center_proximity(detection, image_width, image_height)
    return min(1.0, 0.50 * area_score + 0.30 * center_score + 0.20 * detection.confidence)


def _main_explanation(
    detection: RawDetection,
    image_width: int,
    image_height: int,
) -> str:
    area_percent = _relative_area(detection, image_width, image_height) * 100
    center_score = _center_proximity(detection, image_width, image_height)
    reasons = [f"occupies {area_percent:.1f}% of the image"]
    if center_score >= 0.65:
        reasons.append("is positioned near the image center")
    if detection.confidence >= 0.8:
        reasons.append("has high detection confidence")
    return "Selected as the main subject because it " + ", ".join(reasons) + "."


def _background_face_explanation(
    detection: RawDetection,
    image_width: int,
    image_height: int,
    settings: Settings,
) -> str:
    area_percent = _relative_area(detection, image_width, image_height) * 100
    threshold_percent = settings.main_subject_area_threshold * 100
    if area_percent < threshold_percent:
        return (
            f"Classified as background because the face occupies {area_percent:.1f}% of the image, "
            f"below the {threshold_percent:.1f}% main-subject size threshold."
        )
    return "Classified as background because another face received a stronger size, position and confidence score."


def _contains_face_center(person: RawDetection, face: RawDetection) -> bool:
    face_center_x = face.x + face.width / 2
    face_center_y = face.y + face.height / 2
    return person.x <= face_center_x <= person.x2 and person.y <= face_center_y <= person.y2


def assign_subject_context(
    detections: list[RawDetection],
    image_width: int,
    image_height: int,
    settings: Settings,
) -> list[RawDetection]:
    faces = [detection for detection in detections if detection.category == "face"]
    scored_faces = [
        (
            detection,
            _relative_area(detection, image_width, image_height),
            _subject_score(
                detection,
                image_width,
                image_height,
                max(settings.main_subject_area_threshold, 0.000001),
            ),
        )
        for detection in faces
    ]
    eligible = [
        item
        for item in scored_faces
        if item[1] >= settings.main_subject_area_threshold
        and item[2] >= settings.main_subject_score_threshold
    ]
    main_face = max(eligible, key=lambda item: item[2])[0] if eligible else None

    contextualized: list[RawDetection] = []
    for detection in detections:
        if detection.category == "face":
            score = next(score for face, _area, score in scored_faces if face is detection)
            is_main = detection is main_face
            explanation = (
                _main_explanation(detection, image_width, image_height)
                if is_main
                else _background_face_explanation(detection, image_width, image_height, settings)
            )
            contextualized.append(
                replace(
                    detection,
                    is_main_subject=is_main,
                    subject_score=score,
                    explanation=explanation,
                )
            )
            continue

        is_main_person = main_face is not None and _contains_face_center(detection, main_face)
        contextualized.append(
            replace(
                detection,
                is_main_subject=is_main_person,
                subject_score=None,
                explanation=(
                    "Associated with the main subject because the selected main face is inside this person region."
                    if is_main_person
                    else "Classified as a background person because this region is not associated with the selected main face."
                ),
            )
        )
    return contextualized
