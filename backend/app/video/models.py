"""Typed contracts shared by the video routes and worker."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


VideoJobState = Literal["queued", "processing", "completed", "failed", "cancelled"]


class VideoMetadata(BaseModel):
    filename: str
    duration_seconds: float = Field(gt=0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0)
    frame_count: int | None = Field(default=None, gt=0)
    file_size_bytes: int = Field(gt=0)
    container: str
    fps_fallback_used: bool = False


class VideoCapabilities(BaseModel):
    formats: list[str]
    max_size_bytes: int
    max_duration_seconds: int
    quality_profiles: dict[str, dict[str, int]]
    modules: dict[str, bool]
    ffmpeg_available: bool
    h264_available: bool
    audio_preservation_available: bool
    local_processing_only: Literal[True] = True


class VideoUploadResponse(BaseModel):
    upload_id: str
    metadata: VideoMetadata
    capabilities: VideoCapabilities


class VideoProcessSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    preserve_main_subject: bool = True
    protect_background_faces: bool = True
    protect_license_plates: bool = True
    protect_cards: bool = True
    protect_identity_documents: bool = True
    protect_qr_codes: bool = True
    protect_barcodes: bool = True
    protect_sensitive_text: bool = True
    anonymization_method: Literal["blur", "pixelate", "blackout"] = "blur"
    strength: Literal["low", "medium", "high"] = "medium"
    quality_profile: Literal["performance", "balanced", "accuracy"] = "balanced"


class VideoProcessResponse(BaseModel):
    job_id: str
    state: VideoJobState
    metadata: VideoMetadata


class VideoRiskSummary(BaseModel):
    overall_video_score: int = Field(ge=0, le=100)
    level: Literal["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]
    peak_score: int = Field(ge=0, le=100)
    residual_score: int = Field(ge=0, le=100)
    residual_level: Literal["LOW", "MODERATE", "ELEVATED", "HIGH", "CRITICAL"]
    risk_reduction: int = Field(ge=0, le=100)
    top_risks: list[str]
    category_summary: dict[str, dict[str, float | int | str]]
    based_on_sampled_frames: Literal[True] = True


class VideoPerformance(BaseModel):
    processing_time_seconds: float = Field(ge=0)
    processing_fps: float = Field(ge=0)
    average_face_detector_ms: float = Field(ge=0)
    average_heavy_detector_ms: float = Field(ge=0)
    average_ocr_ms: float = Field(ge=0)
    encoding_time_seconds: float = Field(ge=0)


class VideoSummary(BaseModel):
    duration_seconds: float
    resolution: str
    fps: float
    input_frames: int | None
    analyzed_frames: int
    total_processed_frames: int
    detected_privacy_categories: list[str]
    protected_regions: int
    original_risk: int
    residual_risk: int
    processing_time_seconds: float
    output_size_bytes: int
    codec: str
    audio_preserved: bool
    audio_message: str
    assessment: Literal["complete", "partial"]
    unavailable_modules: list[str]
    performance: VideoPerformance
    risk: VideoRiskSummary


class VideoJobStatus(BaseModel):
    job_id: str
    state: VideoJobState
    progress: float | None = Field(default=None, ge=0, le=100)
    stage: Literal["Preparing", "Analyzing", "Protecting", "Encoding", "Finalizing"]
    error: str | None = None
    output_available: bool = False
    summary: VideoSummary | None = None
    created_at: float
    updated_at: float
