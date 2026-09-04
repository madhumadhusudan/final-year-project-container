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
    role: Literal["main_subject", "background_face", "document_face", "unclassified"] = "unclassified"


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


DocumentType = Literal[
    "aadhaar_card", "pan_card", "passport", "driving_license", "identity_document",
]


class DocumentResult(BaseModel):
    document_id: int = Field(gt=0)
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox
    area_ratio: float = Field(ge=0, le=1)
    center: Point
    final_document_type: DocumentType
    classification_confidence: float = Field(ge=0, le=1)
    classification_status: Literal["model_confirmed", "context_supported", "uncertain"]
    classification_reasons: list[str]
    ocr_text_ids: list[int] = Field(default_factory=list)
    sensitive_text_ids: list[int] = Field(default_factory=list)


class DocumentDetectionDetails(BaseModel):
    status: Literal["completed", "error", "unavailable"]
    detector: str
    document_count: int = Field(ge=0)
    documents: list[DocumentResult]
    message: str | None = None
    model_source: str | None = None
    model_license: str | None = None
    model_class_names: list[str] = Field(default_factory=list)
    confidence_threshold: float = Field(default=0.35, ge=0, le=1)
    inference_image_size: int = Field(default=960, gt=0)


CodeContentType = Literal["url", "payment", "contact", "wifi", "text", "identifier", "unknown"]
CodePrivacyLevel = Literal["low", "moderate", "high", "critical"]
CodeParentType = Literal["identity_document", "payment_card"]


class QRCodeResult(BaseModel):
    qr_id: int = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    bounding_box: BoundingBox
    polygon: list[Point] = Field(min_length=4, max_length=4)
    decoded: bool
    content_type: CodeContentType
    masked_preview: str
    privacy_level: CodePrivacyLevel
    parent_type: CodeParentType | None = None
    parent_id: int | None = Field(default=None, gt=0)


class BarcodeResult(BaseModel):
    barcode_id: int = Field(gt=0)
    confidence: float | None = Field(default=None, ge=0, le=1)
    format: str | None = None
    bounding_box: BoundingBox
    polygon: list[Point] = Field(min_length=4, max_length=4)
    decoded: bool
    content_type: CodeContentType
    masked_preview: str
    privacy_level: CodePrivacyLevel
    parent_type: CodeParentType | None = None
    parent_id: int | None = Field(default=None, gt=0)


class QRDetectionDetails(BaseModel):
    status: Literal["completed", "error", "unavailable"]
    detector: str
    qr_count: int = Field(ge=0)
    items: list[QRCodeResult]
    message: str | None = None


class BarcodeDetectionDetails(BaseModel):
    status: Literal["completed", "error", "unavailable"]
    detector: str
    barcode_count: int = Field(ge=0)
    items: list[BarcodeResult]
    supported_formats: list[str] = Field(default_factory=list)
    message: str | None = None


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
    identity_documents: int = Field(default=0, ge=0)
    qr_codes: int = Field(default=0, ge=0)
    barcodes: int = Field(default=0, ge=0)
    sensitive_text: int = Field(default=0, ge=0)
    context_uncertainty: int = Field(default=0, ge=0)


class RiskFactor(BaseModel):
    type: str
    category: Literal[
        "background_faces", "license_plates", "payment_cards",
        "identity_documents", "qr_codes", "barcodes", "sensitive_text", "context_uncertainty",
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


class PrivacySensitiveElements(BaseModel):
    background_faces: int = Field(default=0, ge=0)
    license_plates: int = Field(default=0, ge=0)
    payment_cards: int = Field(default=0, ge=0)
    identity_documents: int = Field(default=0, ge=0)
    qr_codes: int = Field(default=0, ge=0)
    barcodes: int = Field(default=0, ge=0)
    sensitive_text: int = Field(default=0, ge=0)


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
    document_detection: DocumentDetectionDetails = Field(default_factory=lambda: DocumentDetectionDetails(
        status="unavailable", detector="model_required", document_count=0, documents=[],
        message="Dedicated document model required.",
    ))
    qr_detection: QRDetectionDetails = Field(default_factory=lambda: QRDetectionDetails(
        status="unavailable", detector="opencv_qrcode_detector", qr_count=0, items=[],
        message="Local QR detector unavailable.",
    ))
    barcode_detection: BarcodeDetectionDetails = Field(default_factory=lambda: BarcodeDetectionDetails(
        status="unavailable", detector="opencv_barcode_detector", barcode_count=0, items=[],
        message="Local barcode detector unavailable.",
    ))
    ocr: OCRDetails
    sensitive_text: SensitiveTextDetails
    # Optional on input so Day 8 analysis payloads remain valid for /protect.
    # Every new /analyze response populates this field.
    privacy_risk: PrivacyRiskDetails | None = None
    privacy_sensitive_elements: PrivacySensitiveElements = Field(default_factory=PrivacySensitiveElements)


class PerformanceDetails(BaseModel):
    inference_time_ms: int = Field(ge=0)
    object_detection_ms: int = Field(ge=0)
    face_detection_ms: int = Field(ge=0)
    license_plate_detection_ms: int = Field(ge=0)
    card_detection_ms: int = Field(ge=0)
    document_detection_ms: int = Field(default=0, ge=0)
    document_classification_ms: int = Field(default=0, ge=0)
    qr_detection_ms: int = Field(default=0, ge=0)
    barcode_detection_ms: int = Field(default=0, ge=0)
    code_classification_ms: int = Field(default=0, ge=0)
    context_analysis_ms: int = Field(ge=0)
    ocr_detection_ms: int = Field(ge=0)
    sensitive_text_analysis_ms: int = Field(ge=0)
    total_analysis_ms: int = Field(ge=0)


class AnalysisResponse(BaseModel):
    status: Literal["success"] = "success"
    image: ImageDetails
    analysis: AnalysisDetails
    performance: PerformanceDetails


LiveRegionCategory = Literal[
    "face", "license_plate", "payment_card", "identity_document",
    "qr_code", "barcode", "sensitive_text",
]


class LiveRegion(BaseModel):
    """A privacy region in analysis-frame pixel coordinates.

    Stable temporal track IDs intentionally remain a browser concern. The backend
    IDs identify detections only within this response and never identify people.
    """

    detection_id: str
    category: LiveRegionCategory
    bounding_box: BoundingBox
    confidence: float | None = Field(default=None, ge=0, le=1)
    role: Literal["main_subject", "background_face", "unclassified"] | None = None


class LiveModuleResult(BaseModel):
    status: Literal["completed", "unavailable", "error", "skipped"]
    duration_ms: int = Field(default=0, ge=0)
    detection_count: int = Field(default=0, ge=0)
    message: str | None = None


class LiveRiskSummary(BaseModel):
    score: int = Field(ge=0, le=100)
    level: RiskLevel


class LiveFramePerformance(BaseModel):
    total_analysis_ms: int = Field(ge=0)
    server_received_at_ms: int = Field(ge=0)
    server_completed_at_ms: int = Field(ge=0)


class LiveFrameResponse(BaseModel):
    status: Literal["success"] = "success"
    frame_id: int = Field(gt=0)
    captured_at_ms: int = Field(ge=0)
    image: ImageDetails
    regions: list[LiveRegion]
    main_subject: MainSubjectDetails
    modules: dict[str, LiveModuleResult]
    risk: LiveRiskSummary
    performance: LiveFramePerformance


class LiveCapabilitiesResponse(BaseModel):
    analysis_widths: list[int]
    modules: dict[str, bool]
    local_processing_only: Literal[True] = True
    frame_storage: Literal["none"] = "none"


class ProtectionSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    protect_background_faces: bool = True
    protect_license_plates: bool = True
    protect_cards: bool = True
    protect_identity_documents: bool = True
    protect_qr_codes: bool = True
    protect_barcodes: bool = True
    protect_sensitive_text: bool = True
    anonymization_method: Literal["blur", "pixelate", "blackout"] = "blur"
    strength: Literal["low", "medium", "high"] = "medium"


class ProtectionBreakdown(BaseModel):
    background_faces: int = Field(default=0, ge=0)
    license_plates: int = Field(default=0, ge=0)
    cards: int = Field(default=0, ge=0)
    identity_documents: int = Field(default=0, ge=0)
    qr_codes: int = Field(default=0, ge=0)
    barcodes: int = Field(default=0, ge=0)
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
