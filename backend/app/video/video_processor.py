"""Secure upload validation and streaming video processing."""

from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2
from fastapi import HTTPException, UploadFile, status

from app.config import ALLOWED_VIDEO_EXTENSIONS, BACKEND_DIR, VIDEO_MIME_TYPES, Settings

from .models import VideoMetadata, VideoPerformance, VideoProcessSettings, VideoSummary
from .video_analyzer import VideoAnalyzer
from .video_writer import EncodingCapabilities, VideoFrameWriter, finalize_video


UPLOAD_DIRECTORY = BACKEND_DIR / "uploads"
OUTPUT_DIRECTORY = BACKEND_DIR / "outputs"


@dataclass(frozen=True)
class StoredVideo:
    upload_id: str
    path: Path
    metadata: VideoMetadata
    created_at: float


def _has_valid_signature(extension: str, header: bytes) -> bool:
    if extension in {".mp4", ".mov"}:
        return len(header) >= 12 and header[4:8] == b"ftyp"
    if extension == ".avi":
        return len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"AVI "
    if extension == ".webm":
        return header.startswith(b"\x1aE\xdf\xa3")
    return False


def inspect_video(path: Path, filename: str, size: int, extension: str, app_settings: Settings) -> VideoMetadata:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The video is corrupt or uses an unsupported codec.")
        width = round(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        raw_fps = float(capture.get(cv2.CAP_PROP_FPS))
        raw_count = float(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_count = round(raw_count) if math.isfinite(raw_count) and raw_count >= 1 else None
        fallback = not math.isfinite(raw_fps) or raw_fps <= 0.01 or raw_fps > 240
        fps = float(app_settings.video_default_fps) if fallback else raw_fps
        if width <= 0 or height <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The video has invalid dimensions.")
        if width * height > 33_177_600:
            raise HTTPException(status_code=status.HTTP_413_CONTENT_TOO_LARGE, detail="Video resolution must not exceed 8K.")
        if frame_count is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The video frame count could not be read safely.")
        duration = frame_count / fps
        if duration <= 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The video has zero readable frames.")
        if duration > app_settings.max_video_duration_seconds:
            raise HTTPException(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                detail=f"Video duration must be {app_settings.max_video_duration_seconds} seconds or shorter.",
            )
        capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
        readable, _ = capture.read()
        if not readable:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The video contains no decodable frames.")
        return VideoMetadata(
            filename=filename, duration_seconds=round(duration, 3), width=width, height=height,
            fps=round(fps, 3), frame_count=frame_count, file_size_bytes=size,
            container=extension.removeprefix(".").upper(), fps_fallback_used=fallback,
        )
    finally:
        capture.release()


async def store_and_validate_video(upload: UploadFile, app_settings: Settings) -> StoredVideo:
    filename = Path((upload.filename or "video").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    declared_mime = (upload.content_type or "").lower()
    if extension not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Choose an MP4, MOV, AVI or WEBM video.")
    if declared_mime not in VIDEO_MIME_TYPES[extension]:
        raise HTTPException(status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="The declared video type does not match the file extension.")
    UPLOAD_DIRECTORY.mkdir(parents=True, exist_ok=True)
    upload_id = uuid.uuid4().hex
    path = (UPLOAD_DIRECTORY / f"video_{upload_id}{extension}").resolve()
    if path.parent != UPLOAD_DIRECTORY.resolve():
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid upload path.")
    size = 0
    header = b""
    try:
        with path.open("xb") as destination:
            while chunk := await upload.read(1024 * 1024):
                size += len(chunk)
                if size > app_settings.max_video_bytes:
                    raise HTTPException(
                        status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                        detail=f"Video must be {app_settings.max_video_bytes // (1024 * 1024)} MB or smaller.",
                    )
                if len(header) < 32:
                    header += chunk[:32 - len(header)]
                destination.write(chunk)
        if size == 0:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded video is empty.")
        if not _has_valid_signature(extension, header):
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The video content does not match its container type.")
        metadata = inspect_video(path, filename, size, extension, app_settings)
        return StoredVideo(upload_id, path, metadata, time.time())
    except Exception:
        path.unlink(missing_ok=True)
        raise
    finally:
        await upload.close()


def process_video(
    source: StoredVideo, output_path: Path, detection_service, process_settings: VideoProcessSettings,
    app_settings: Settings, capabilities: EncodingCapabilities, cancel_event, update,
) -> VideoSummary:
    started = time.perf_counter()
    intermediate = OUTPUT_DIRECTORY / f"frames_{uuid.uuid4().hex}.mp4"
    capture = cv2.VideoCapture(str(source.path))
    writer = None
    processed = 0
    analyzer = VideoAnalyzer(detection_service, process_settings, app_settings.video_track_expiry_frames)
    try:
        if not capture.isOpened():
            raise RuntimeError("The uploaded video could not be reopened for processing.")
        update("processing", 0.0, "Preparing")
        writer = VideoFrameWriter(intermediate, source.metadata.fps, source.metadata.width, source.metadata.height)
        total = source.metadata.frame_count
        while True:
            if cancel_event.is_set():
                raise InterruptedError("Video processing was cancelled.")
            readable, frame = capture.read()
            if not readable:
                break
            if frame.shape[1] != source.metadata.width or frame.shape[0] != source.metadata.height:
                frame = cv2.resize(frame, (source.metadata.width, source.metadata.height))
            protected = analyzer.process_frame(frame, processed)
            writer.write(protected)
            processed += 1
            progress = min(100.0, processed / total * 100) if total else None
            update("processing", progress, "Protecting" if processed > 1 else "Analyzing")
        if processed == 0:
            raise RuntimeError("The video contained no processable frames.")
        writer.close()
        writer = None
        capture.release()
        update("processing", 100.0, "Encoding")
        finalized = finalize_video(intermediate, source.path, output_path, capabilities, cancel_event)
        update("processing", 100.0, "Finalizing")
        elapsed = time.perf_counter() - started
        risk = analyzer.risk_summary()
        timings = analyzer.timing_summary()
        unavailable = sorted(analyzer.unavailable_modules)
        enabled_module_names = {
            "faces": process_settings.protect_background_faces,
            "plates": process_settings.protect_license_plates,
            "cards": process_settings.protect_cards,
            "documents": process_settings.protect_identity_documents,
            "codes": process_settings.protect_qr_codes or process_settings.protect_barcodes,
            "ocr": process_settings.protect_sensitive_text,
        }
        expected_unavailable = [name for name, enabled in enabled_module_names.items() if enabled and name in unavailable]
        performance = VideoPerformance(
            processing_time_seconds=round(elapsed, 3), processing_fps=round(processed / elapsed, 3) if elapsed else 0,
            encoding_time_seconds=round(finalized.encoding_seconds, 3), **timings,
        )
        detected = [name for name, details in risk["category_summary"].items() if details["frames_present"]]
        return VideoSummary(
            duration_seconds=source.metadata.duration_seconds,
            resolution=f"{source.metadata.width}x{source.metadata.height}", fps=source.metadata.fps,
            input_frames=source.metadata.frame_count, analyzed_frames=len(analyzer.analyzed_frame_indexes),
            total_processed_frames=processed, detected_privacy_categories=detected,
            protected_regions=analyzer.protected_regions, original_risk=risk["overall_video_score"],
            residual_risk=risk["residual_score"], processing_time_seconds=round(elapsed, 3),
            output_size_bytes=output_path.stat().st_size, codec=finalized.codec,
            audio_preserved=finalized.audio_preserved, audio_message=finalized.audio_message,
            assessment="partial" if expected_unavailable else "complete", unavailable_modules=expected_unavailable,
            performance=performance, risk=risk,
        )
    finally:
        capture.release()
        if writer is not None:
            writer.close()
        intermediate.unlink(missing_ok=True)
        source.path.unlink(missing_ok=True)
