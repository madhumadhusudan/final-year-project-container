"""Transient real-time camera-frame analysis endpoints."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.detection.live_service import LIVE_MODULES, LiveDetectionService
from app.detection.service import DetectionService, get_detection_service
from app.schemas import LiveCapabilitiesResponse, LiveFrameResponse
from app.utils.image_validation import read_and_validate_image

router = APIRouter(tags=["live privacy"])


@router.get("/live/capabilities", response_model=LiveCapabilitiesResponse)
async def live_capabilities(
    service: DetectionService = Depends(get_detection_service),
) -> LiveCapabilitiesResponse:
    return LiveCapabilitiesResponse(
        analysis_widths=[480, 640, 768],
        modules=LiveDetectionService(service).capabilities(),
    )


@router.post("/analyze-frame", response_model=LiveFrameResponse)
async def analyze_frame(
    image: UploadFile = File(...),
    frame_id: int = Form(..., gt=0),
    captured_at_ms: int = Form(..., ge=0),
    modules: str = Form("faces"),
    preserve_main_subject: bool = Form(True),
    service: DetectionService = Depends(get_detection_service),
) -> LiveFrameResponse:
    requested = {item.strip().lower() for item in modules.split(",") if item.strip()}
    unknown = requested - LIVE_MODULES
    if not requested or unknown:
        supported = ", ".join(sorted(LIVE_MODULES))
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Choose one or more supported live modules: {supported}.",
        )

    decoded = await read_and_validate_image(image)
    # This endpoint is for downscaled analysis frames, not full camera captures.
    if decoded.width * decoded.height > 2_500_000 or max(decoded.width, decoded.height) > 1920:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Live analysis frames must be downscaled before upload.",
        )
    try:
        return await asyncio.to_thread(
            LiveDetectionService(service).analyze,
            decoded, frame_id, captured_at_ms, requested, preserve_main_subject,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local live detection engine could not analyze this frame.",
        ) from exc
