"""Video upload, background processing, status, cancellation, and result routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.config import settings
from app.detection.service import DetectionService, get_detection_service
from app.video.models import (
    VideoCapabilities, VideoJobStatus, VideoProcessResponse, VideoProcessSettings, VideoUploadResponse,
)
from app.video.video_jobs import video_job_manager
from app.video.video_processor import store_and_validate_video


router = APIRouter(prefix="/video", tags=["video privacy"])


class StartVideoRequest(BaseModel):
    upload_id: str
    settings: VideoProcessSettings = VideoProcessSettings()


@router.get("/capabilities", response_model=VideoCapabilities)
async def video_capabilities(service: DetectionService = Depends(get_detection_service)) -> VideoCapabilities:
    return video_job_manager.capability_response(service)


@router.post("/upload", response_model=VideoUploadResponse)
async def upload_video(
    video: UploadFile = File(...), service: DetectionService = Depends(get_detection_service),
) -> VideoUploadResponse:
    source = await store_and_validate_video(video, settings)
    video_job_manager.add_upload(source)
    return VideoUploadResponse(
        upload_id=source.upload_id, metadata=source.metadata,
        capabilities=video_job_manager.capability_response(service),
    )


@router.post("/process", response_model=VideoProcessResponse, status_code=status.HTTP_202_ACCEPTED)
async def start_video_processing(
    request: StartVideoRequest, service: DetectionService = Depends(get_detection_service),
) -> VideoProcessResponse:
    try:
        job = video_job_manager.start(request.upload_id, request.settings, service)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc.args[0])) from exc
    return VideoProcessResponse(job_id=job.job_id, state="queued", metadata=job.source.metadata)


@router.delete("/upload/{upload_id}", status_code=status.HTTP_204_NO_CONTENT)
async def discard_video_upload(upload_id: str) -> None:
    video_job_manager.discard_upload(upload_id)


@router.get("/status/{job_id}", response_model=VideoJobStatus)
async def video_status(job_id: str) -> VideoJobStatus:
    try:
        return video_job_manager.status(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video job not found.") from exc


@router.get("/result/{job_id}")
async def video_result(job_id: str) -> FileResponse:
    try:
        path = video_job_manager.result_path(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video job not found.") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc)) from exc
    return FileResponse(path, media_type="video/mp4", filename="privacy-protected-video.mp4")


@router.delete("/cancel/{job_id}", response_model=VideoJobStatus)
@router.post("/cancel/{job_id}", response_model=VideoJobStatus, include_in_schema=False)
async def cancel_video(job_id: str) -> VideoJobStatus:
    try:
        return video_job_manager.cancel(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Video job not found.") from exc
