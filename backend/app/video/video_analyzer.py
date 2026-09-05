"""Sampled detector orchestration, tracking, anonymization, and risk aggregation."""

from __future__ import annotations

import time
from collections import defaultdict

import cv2

from app.anonymization.anonymizer import (
    BARCODE_PADDING, CARD_PADDING, DOCUMENT_PADDING, FACE_PADDING, PLATE_PADDING,
    QR_PADDING, TEXT_PADDING, ImageAnonymizer,
)
from app.anonymization.region_utils import Region, pad_region, sanitize_region
from app.detection.live_service import LiveDetectionService
from app.schemas import ProtectionSettings
from app.utils.image_validation import DecodedImage

from .models import VideoProcessSettings
from .video_tracker import RegionTracker, TrackedRegion


QUALITY_PROFILES = {
    "performance": {"analysis_width": 640, "general": 10, "faces": 5, "plates": 15, "cards": 15, "documents": 15, "codes": 15, "ocr": 30},
    "balanced": {"analysis_width": 960, "general": 7, "faces": 3, "plates": 10, "cards": 10, "documents": 10, "codes": 10, "ocr": 20},
    "accuracy": {"analysis_width": 1280, "general": 4, "faces": 2, "plates": 5, "cards": 6, "documents": 6, "codes": 6, "ocr": 12},
}

CATEGORY_MODULE = {
    "face": "faces", "license_plate": "plates", "payment_card": "cards",
    "identity_document": "documents", "qr_code": "codes", "barcode": "codes",
    "sensitive_text": "ocr",
}
SETTING_BY_CATEGORY = {
    "face": "protect_background_faces", "license_plate": "protect_license_plates",
    "payment_card": "protect_cards", "identity_document": "protect_identity_documents",
    "qr_code": "protect_qr_codes", "barcode": "protect_barcodes",
    "sensitive_text": "protect_sensitive_text",
}
PADDING = {
    "face": FACE_PADDING, "license_plate": PLATE_PADDING, "payment_card": CARD_PADDING,
    "identity_document": DOCUMENT_PADDING, "qr_code": QR_PADDING,
    "barcode": BARCODE_PADDING, "sensitive_text": TEXT_PADDING,
}
RISK_WEIGHTS = {
    "face": 30, "license_plate": 22, "payment_card": 38, "identity_document": 42,
    "qr_code": 18, "barcode": 14, "sensitive_text": 34,
}
DISPLAY_NAMES = {
    "face": "Background Faces", "license_plate": "License Plates", "payment_card": "Payment Cards",
    "identity_document": "Identity Documents", "qr_code": "QR Codes", "barcode": "Barcodes",
    "sensitive_text": "Sensitive Text",
}


def risk_level(score: int) -> str:
    if score >= 85:
        return "CRITICAL"
    if score >= 65:
        return "HIGH"
    if score >= 40:
        return "ELEVATED"
    if score >= 20:
        return "MODERATE"
    return "LOW"


class VideoAnalyzer:
    def __init__(self, detection_service, settings: VideoProcessSettings, expiry_frames: int) -> None:
        self.service = detection_service
        self.settings = settings
        self.profile = QUALITY_PROFILES[settings.quality_profile]
        self.live = LiveDetectionService(detection_service)
        self.tracker = RegionTracker(expiry_frames)
        self.anonymizer = ImageAnonymizer()
        self.main_subject_track_id: str | None = None
        self.analyzed_frame_indexes: set[int] = set()
        self.category_analyzed_frames: dict[str, int] = defaultdict(int)
        self.category_present_frames: dict[str, int] = defaultdict(int)
        self.category_region_count: dict[str, int] = defaultdict(int)
        self.unavailable_modules: set[str] = set()
        self.protected_regions = 0
        self.peak_risk = 0
        self.face_times: list[int] = []
        self.heavy_times: list[int] = []
        self.ocr_times: list[int] = []

    def _analysis_frame(self, frame):
        height, width = frame.shape[:2]
        target = min(width, self.profile["analysis_width"])
        if target == width:
            return frame, 1.0, 1.0
        target_height = max(1, round(height * target / width))
        resized = cv2.resize(frame, (target, target_height), interpolation=cv2.INTER_AREA)
        return resized, width / target, height / target_height

    def _scheduled_modules(self, frame_index: int) -> set[str]:
        modules = set()
        face_needed = self.settings.protect_background_faces or self.settings.preserve_main_subject
        if face_needed and frame_index % self.profile["faces"] == 0:
            modules.add("faces")
        for module, setting_name in (
            ("plates", "protect_license_plates"), ("cards", "protect_cards"),
            ("documents", "protect_identity_documents"), ("ocr", "protect_sensitive_text"),
        ):
            if getattr(self.settings, setting_name) and frame_index % self.profile[module] == 0:
                modules.add(module)
        if (self.settings.protect_qr_codes or self.settings.protect_barcodes) and frame_index % self.profile["codes"] == 0:
            modules.add("codes")
        return modules

    @staticmethod
    def _scaled_box(box, scale_x: float, scale_y: float) -> tuple[float, float, float, float]:
        return (box.x1 * scale_x, box.y1 * scale_y, box.x2 * scale_x, box.y2 * scale_y)

    def _run_general_detector(self, decoded: DecodedImage, frame_index: int) -> None:
        if frame_index % self.profile["general"]:
            return
        started = time.perf_counter()
        try:
            self.service._detector.detect(decoded.pixels_bgr)
        except (RuntimeError, cv2.error):
            self.unavailable_modules.add("general_object_detection")
        self.heavy_times.append(round((time.perf_counter() - started) * 1000))

    def _update_metrics(self, result, requested: set[str], detections: list[dict], frame_index: int) -> None:
        if requested:
            self.analyzed_frame_indexes.add(frame_index)
        for module in requested:
            module_result = result.modules[module]
            if module_result.status in {"unavailable", "error"}:
                self.unavailable_modules.add(module)
            if module == "faces":
                self.face_times.append(module_result.duration_ms)
            elif module == "ocr":
                self.ocr_times.append(module_result.duration_ms)
            else:
                self.heavy_times.append(module_result.duration_ms)
        observed_categories = {category for category, module in CATEGORY_MODULE.items() if module in requested}
        for category in observed_categories:
            self.category_analyzed_frames[category] += 1
            items = [item for item in detections if item["category"] == category]
            if category == "face" and self.settings.preserve_main_subject:
                items = [item for item in items if item.get("role") != "main_subject"]
            if items:
                self.category_present_frames[category] += 1
                self.category_region_count[category] += len(items)

    def _stable_main_subject(self, tracks: list[TrackedRegion], face_was_analyzed: bool, frame_index: int) -> str | None:
        if not self.settings.preserve_main_subject:
            self.main_subject_track_id = None
            return None
        by_id = {track.track_id: track for track in tracks}
        current = by_id.get(self.main_subject_track_id or "")
        if current:
            if not face_was_analyzed or current.last_seen_frame == frame_index:
                return current.track_id
            # A scheduled face check missed this track: fail safely on this frame.
            return None
        self.main_subject_track_id = None
        candidates = [
            track for track in tracks
            if track.category == "face" and track.last_seen_frame == frame_index and track.role == "main_subject"
        ]
        if candidates:
            self.main_subject_track_id = max(
                candidates, key=lambda item: (item.box[2] - item.box[0]) * (item.box[3] - item.box[1]),
            ).track_id
        return self.main_subject_track_id

    def _apply(self, frame, tracks: list[TrackedRegion], main_track_id: str | None, frame_index: int) -> None:
        height, width = frame.shape[:2]
        protection_settings = ProtectionSettings(
            anonymization_method=self.settings.anonymization_method, strength=self.settings.strength,
        )
        for track in tracks:
            if not getattr(self.settings, SETTING_BY_CATEGORY[track.category]):
                continue
            if track.category == "face" and main_track_id == track.track_id:
                continue
            predicted = track.predicted_box(frame_index)
            region = sanitize_region(Region(*map(round, predicted)), width, height)
            if region is None:
                continue
            region = pad_region(region, PADDING[track.category], width, height)
            if region is not None and self.anonymizer._apply(frame, region, protection_settings):
                self.protected_regions += 1

    def process_frame(self, frame, frame_index: int):
        analysis_frame, scale_x, scale_y = self._analysis_frame(frame)
        height, width = analysis_frame.shape[:2]
        decoded = DecodedImage(analysis_frame, width, height, "VIDEO_FRAME")
        requested = self._scheduled_modules(frame_index)
        self._run_general_detector(decoded, frame_index)
        detections: list[dict] = []
        if requested:
            try:
                result = self.live.analyze(decoded, frame_index + 1, 0, requested, self.settings.preserve_main_subject)
                for item in result.regions:
                    detections.append({
                        "category": item.category,
                        "box": self._scaled_box(item.bounding_box, scale_x, scale_y),
                        "confidence": item.confidence,
                        "role": item.role,
                    })
                self._update_metrics(result, requested, detections, frame_index)
            except Exception:
                # One sampled detector pass must not destroy the output stream. With
                # no new boxes, existing tracks receive their short grace period.
                self.analyzed_frame_indexes.add(frame_index)
                self.unavailable_modules.update(requested)
        observed = {category for category, module in CATEGORY_MODULE.items() if module in requested}
        tracks = self.tracker.update(detections, observed, frame_index)
        main_track = self._stable_main_subject(tracks, "faces" in requested, frame_index)
        risk_tracks = [track for track in tracks if not (track.category == "face" and track.track_id == main_track)]
        frame_score = min(100, sum(RISK_WEIGHTS[track.category] for track in {item.track_id: item for item in risk_tracks}.values()))
        self.peak_risk = max(self.peak_risk, frame_score)
        self._apply(frame, tracks, main_track, frame_index)
        return frame

    @staticmethod
    def _average(values: list[int]) -> float:
        return round(sum(values) / len(values), 2) if values else 0.0

    def risk_summary(self) -> dict:
        category_summary = {}
        persistent_score = 0.0
        for category, display in DISPLAY_NAMES.items():
            analyzed = self.category_analyzed_frames[category]
            present = self.category_present_frames[category]
            frequency = round((present / analyzed * 100), 1) if analyzed else 0.0
            if present:
                persistent_score += RISK_WEIGHTS[category] * min(1.0, frequency / 50)
            category_summary[display] = {
                "analyzed_frames": analyzed, "frames_present": present,
                "presence_percent": frequency, "detections": self.category_region_count[category],
                "basis": "sampled detector frames",
            }
        overall = min(100, round(self.peak_risk * 0.65 + min(100, persistent_score) * 0.35))
        effectiveness = {"blur": 0.78, "pixelate": 0.86, "blackout": 0.96}[self.settings.anonymization_method]
        residual = max(1 if overall else 0, round(overall * (1 - effectiveness)))
        top = sorted(
            (category for category in DISPLAY_NAMES if self.category_present_frames[category]),
            key=lambda category: RISK_WEIGHTS[category] * self.category_present_frames[category], reverse=True,
        )[:3]
        return {
            "overall_video_score": overall, "level": risk_level(overall), "peak_score": self.peak_risk,
            "residual_score": residual, "residual_level": risk_level(residual),
            "risk_reduction": max(0, overall - residual), "top_risks": [DISPLAY_NAMES[item] for item in top],
            "category_summary": category_summary, "based_on_sampled_frames": True,
        }

    def timing_summary(self) -> dict[str, float]:
        return {
            "average_face_detector_ms": self._average(self.face_times),
            "average_heavy_detector_ms": self._average(self.heavy_times),
            "average_ocr_ms": self._average(self.ocr_times),
        }
