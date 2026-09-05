"""Bounded in-memory job and temporary-upload registry."""

from __future__ import annotations

import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from app.config import Settings, settings
from app.detection.live_service import LiveDetectionService

from .models import VideoCapabilities, VideoJobStatus, VideoProcessSettings
from .video_analyzer import QUALITY_PROFILES
from .video_processor import OUTPUT_DIRECTORY, StoredVideo, process_video
from .video_writer import detect_encoding_capabilities


@dataclass
class VideoJob:
    job_id: str
    source: StoredVideo
    output_path: Path
    process_settings: VideoProcessSettings
    state: str = "queued"
    progress: float | None = 0.0
    stage: str = "Preparing"
    error: str | None = None
    summary: object | None = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    cancel_event: threading.Event = field(default_factory=threading.Event)


class VideoJobManager:
    def __init__(self, app_settings: Settings = settings) -> None:
        self.settings = app_settings
        self.capabilities = detect_encoding_capabilities()
        self._uploads: dict[str, StoredVideo] = {}
        self._jobs: dict[str, VideoJob] = {}
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(
            max_workers=app_settings.max_concurrent_video_jobs, thread_name_prefix="privacy-video",
        )

    def capability_response(self, detection_service) -> VideoCapabilities:
        intervals = {
            name: {key: value for key, value in profile.items()}
            for name, profile in QUALITY_PROFILES.items()
        }
        return VideoCapabilities(
            formats=sorted(item.removeprefix(".").upper() for item in {".mp4", ".mov", ".avi", ".webm"}),
            max_size_bytes=self.settings.max_video_bytes,
            max_duration_seconds=self.settings.max_video_duration_seconds,
            quality_profiles=intervals,
            modules=LiveDetectionService(detection_service).capabilities(),
            ffmpeg_available=self.capabilities.ffmpeg_available,
            h264_available=self.capabilities.h264_available,
            audio_preservation_available=self.capabilities.ffmpeg_available,
        )

    def cleanup_stale(self) -> None:
        now = time.time()
        with self._lock:
            stale_uploads = [key for key, item in self._uploads.items() if now - item.created_at > 1800]
            for key in stale_uploads:
                self._uploads.pop(key).path.unlink(missing_ok=True)
            stale_jobs = [
                key for key, item in self._jobs.items()
                if item.state in {"completed", "failed", "cancelled"}
                and now - item.updated_at > self.settings.video_output_ttl_seconds
            ]
            for key in stale_jobs:
                self._jobs.pop(key).output_path.unlink(missing_ok=True)

    def add_upload(self, source: StoredVideo) -> None:
        self.cleanup_stale()
        with self._lock:
            self._uploads[source.upload_id] = source

    def discard_upload(self, upload_id: str) -> bool:
        with self._lock:
            source = self._uploads.pop(upload_id, None)
            if source is None:
                return False
            source.path.unlink(missing_ok=True)
            return True

    def start(self, upload_id: str, process_settings: VideoProcessSettings, detection_service) -> VideoJob:
        self.cleanup_stale()
        with self._lock:
            source = self._uploads.pop(upload_id, None)
            if source is None:
                raise KeyError("The temporary upload was not found or has already been used.")
            OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
            job_id = uuid.uuid4().hex
            output = (OUTPUT_DIRECTORY / f"privacy_{job_id}.mp4").resolve()
            if output.parent != OUTPUT_DIRECTORY.resolve():
                source.path.unlink(missing_ok=True)
                raise RuntimeError("Invalid output path.")
            job = VideoJob(job_id, source, output, process_settings)
            self._jobs[job_id] = job
            self._executor.submit(self._run, job, detection_service)
            return job

    def _update(self, job: VideoJob, state: str, progress: float | None, stage: str) -> None:
        with self._lock:
            if job.state in {"cancelled", "failed", "completed"}:
                return
            job.state = state
            job.progress = progress
            job.stage = stage
            job.updated_at = time.time()

    def _run(self, job: VideoJob, detection_service) -> None:
        try:
            if job.cancel_event.is_set():
                raise InterruptedError("Video processing was cancelled.")
            summary = process_video(
                job.source, job.output_path, detection_service, job.process_settings,
                self.settings, self.capabilities, job.cancel_event,
                lambda state, progress, stage: self._update(job, state, progress, stage),
            )
            with self._lock:
                if job.cancel_event.is_set():
                    raise InterruptedError("Video processing was cancelled.")
                job.summary = summary
                job.state = "completed"
                job.progress = 100.0
                job.stage = "Finalizing"
                job.updated_at = time.time()
        except InterruptedError:
            job.source.path.unlink(missing_ok=True)
            job.output_path.unlink(missing_ok=True)
            with self._lock:
                job.state = "cancelled"
                job.error = None
                job.updated_at = time.time()
        except Exception:
            job.source.path.unlink(missing_ok=True)
            job.output_path.unlink(missing_ok=True)
            with self._lock:
                job.state = "failed"
                job.error = "Video processing failed locally. The file may use an unsupported codec or a detector could not continue safely."
                job.updated_at = time.time()

    def status(self, job_id: str) -> VideoJobStatus:
        self.cleanup_stale()
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            return VideoJobStatus(
                job_id=job.job_id, state=job.state, progress=job.progress, stage=job.stage,
                error=job.error, output_available=job.state == "completed" and job.output_path.exists(),
                summary=job.summary, created_at=job.created_at, updated_at=job.updated_at,
            )

    def result_path(self, job_id: str) -> Path:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state != "completed" or not job.output_path.exists():
                raise RuntimeError("The protected video is not ready.")
            return job.output_path

    def cancel(self, job_id: str) -> VideoJobStatus:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                raise KeyError(job_id)
            if job.state in {"completed", "failed", "cancelled"}:
                return self.status(job_id)
            job.cancel_event.set()
            job.state = "cancelled"
            job.error = None
            job.updated_at = time.time()
            return self.status(job_id)


video_job_manager = VideoJobManager()
