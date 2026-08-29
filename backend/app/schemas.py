"""Typed API and internal detection models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


@dataclass(frozen=True)
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
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

    def clipped(self, image_width: int, image_height: int) -> RawDetection:
        x1 = max(0, min(self.x1, image_width))
        y1 = max(0, min(self.y1, image_height))
        x2 = max(x1, min(self.x2, image_width))
        y2 = max(y1, min(self.y2, image_height))
        return replace(self, x1=x1, y1=y1, x2=x2, y2=y2)


class BoundingBox(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)


class DetectionResult(BaseModel):
    id: int = Field(gt=0)
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox


class Point(BaseModel):
    x: float
    y: float


class FaceResult(BaseModel):
    face_id: int = Field(gt=0)
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    area: int = Field(gt=0)
    area_ratio: float = Field(ge=0, le=1)
    center: Point
    normalized_center: Point
    distance_from_image_center: float = Field(ge=0)
    matched_person_id: int | None = None
    role: Literal["main_subject", "background_face", "unclassified"] = "unclassified"


class ImageDetails(BaseModel):
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str


class ObjectDetectionDetails(BaseModel):
    model: str
    detection_count: int = Field(ge=0)
    detections: list[DetectionResult]


class FaceDetectionDetails(BaseModel):
    status: Literal["completed", "error"] = "completed"
    detector: str
    face_count: int = Field(ge=0)
    faces: list[FaceResult]
    error: str | None = None


class MainSubjectDetails(BaseModel):
    status: Literal["identified", "uncertain", "not_found"]
    face_id: int | None = None
    matched_person_id: int | None = None
    subject_score: float | None = Field(default=None, ge=0, le=1)
    size_score: float | None = Field(default=None, ge=0, le=1)
    center_score: float | None = Field(default=None, ge=0, le=1)
    confidence_score: float | None = Field(default=None, ge=0, le=1)
    person_context_score: float | None = Field(default=None, ge=0, le=1)
    face_area_ratio: float | None = Field(default=None, ge=0, le=1)
    score_gap: float | None = Field(default=None, ge=0, le=1)
    reason: str


class PrivacyObjectResult(BaseModel):
    id: int = Field(gt=0)
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox
    matched_vehicle_id: int | None = None


class CardResult(PrivacyObjectResult):
    card_id: int | None = Field(default=None, gt=0)


class DetectorDiagnostics(BaseModel):
    loaded: bool
    model_name: str
    model_class_names: list[str]
    inference_image_size: int = Field(gt=0)
    confidence_threshold: float = Field(ge=0, le=1)
    raw_detection_count: int = Field(ge=0)
    accepted_detection_count: int = Field(ge=0)
    inference_time_ms: int = Field(ge=0)
    tile_size: int | None = Field(default=None, gt=0)
    tile_inference_image_size: int | None = Field(default=None, gt=0)
    tiles_processed: int = Field(default=1, gt=0)


class LicensePlateDetectionDetails(BaseModel):
    status: Literal["completed", "error", "unavailable"]
    detector: str
    plate_count: int = Field(ge=0)
    plates: list[PrivacyObjectResult]
    message: str | None = None
    model_source: str | None = None
    supported_classes: list[str] = Field(default_factory=list)


class CardDetectionDetails(BaseModel):
    status: Literal["completed", "error", "unavailable"]
    detector: str
    card_count: int = Field(ge=0)
    cards: list[CardResult]
    message: str | None = None
    model_source: str | None = None
    supported_classes: list[str] = Field(default_factory=list)
    diagnostics: DetectorDiagnostics | None = None


class OCRTextResult(BaseModel):
    text_id: int = Field(gt=0)
    raw_text: str
    normalized_text: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox


class OCRDetails(BaseModel):
    status: Literal["completed", "error", "unavailable"]
    engine: str
    languages: list[str]
    text_count: int = Field(ge=0)
    texts: list[OCRTextResult]
    message: str | None = None


class SensitiveTextResult(BaseModel):
    id: int = Field(gt=0)
    text_id: int = Field(gt=0)
    type: Literal[
        "phone_number", "email", "url", "possible_address", "pincode",
        "payment_card_number", "possible_expiry_date", "license_plate_text",
        "aadhaar_like_number", "pan_like_number",
    ]
    masked_value: str
    confidence: float = Field(ge=0, le=1)
    reason: str
    bounding_box: BoundingBox
    luhn_valid: bool | None = None


class SensitiveTextDetails(BaseModel):
    status: Literal["completed", "error", "unavailable"]
    count: int = Field(ge=0)
    items: list[SensitiveTextResult]
    message: str | None = None


RiskLevel = Literal["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]


class RiskBreakdown(BaseModel):
    background_faces: int = Field(default=0, ge=0)
    license_plates: int = Field(default=0, ge=0)
    payment_cards: int = Field(default=0, ge=0)
    sensitive_text: int = Field(default=0, ge=0)
    context_uncertainty: int = Field(default=0, ge=0)


class RiskFactor(BaseModel):
    type: str
    category: Literal[
        "background_faces", "license_plates", "payment_cards",
        "sensitive_text", "context_uncertainty",
    ]
    severity: Literal["low", "medium", "high", "critical"]
    contribution: float = Field(ge=0, le=100)
    reason: str


class RiskAssessment(BaseModel):
    status: Literal["complete", "partial"]
    unavailable_modules: list[str] = Field(default_factory=list)


class PrivacyRiskDetails(BaseModel):
    score: int = Field(ge=0, le=100)
    level: RiskLevel
    summary: str
    breakdown: RiskBreakdown
    factors: list[RiskFactor]
    top_risks: list[str]
    recommendations: list[str]
    assessment: RiskAssessment


class AnalysisDetails(BaseModel):
    status: Literal["completed"] = "completed"
    # Day 4 fields remain available while clients migrate to the grouped output.
    model: str
    detection_count: int = Field(ge=0)
    detections: list[DetectionResult]
    object_detection: ObjectDetectionDetails
    face_detection: FaceDetectionDetails
    main_subject: MainSubjectDetails
    license_plate_detection: LicensePlateDetectionDetails
    card_detection: CardDetectionDetails
    ocr: OCRDetails
    sensitive_text: SensitiveTextDetails
    # Optional on input so Day 8 analysis payloads remain valid for /protect.
    # Every new /analyze response populates this field.
    privacy_risk: PrivacyRiskDetails | None = None


class PerformanceDetails(BaseModel):
    inference_time_ms: int = Field(ge=0)
    object_detection_ms: int = Field(ge=0)
    face_detection_ms: int = Field(ge=0)
    license_plate_detection_ms: int = Field(ge=0)
    card_detection_ms: int = Field(ge=0)
    context_analysis_ms: int = Field(ge=0)
    ocr_detection_ms: int = Field(ge=0)
    sensitive_text_analysis_ms: int = Field(ge=0)
    total_analysis_ms: int = Field(ge=0)


class AnalysisResponse(BaseModel):
    status: Literal["success"] = "success"
    image: ImageDetails
    analysis: AnalysisDetails
    performance: PerformanceDetails


class ProtectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protect_background_faces: bool = True
    protect_license_plates: bool = True
    protect_cards: bool = True
    protect_sensitive_text: bool = True
    anonymization_method: Literal["blur", "pixelate", "blackout"] = "blur"
    strength: Literal["low", "medium", "high"] = "medium"


class ProtectionBreakdown(BaseModel):
    background_faces: int = Field(default=0, ge=0)
    license_plates: int = Field(default=0, ge=0)
    cards: int = Field(default=0, ge=0)
    sensitive_text: int = Field(default=0, ge=0)


class RiskScoreSummary(BaseModel):
    score: int = Field(ge=0, le=100)
    level: RiskLevel


class RiskReductionDetails(BaseModel):
    before: RiskScoreSummary
    after: RiskScoreSummary
    reduction: int = Field(ge=0, le=100)
    reduction_percent: float = Field(ge=0, le=100)


class ProtectionMetadata(BaseModel):
    status: Literal["completed"] = "completed"
    method: Literal["blur", "pixelate", "blackout"]
    strength: Literal["low", "medium", "high"]
    regions_protected: int = Field(ge=0)
    breakdown: ProtectionBreakdown
    main_subject_preserved: bool
    warnings: list[str] = Field(default_factory=list)
    risk: RiskReductionDetails | None = None
