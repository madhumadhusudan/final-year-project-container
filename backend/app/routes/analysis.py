"""Image analysis API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from app.detection.service import DetectionService, get_detection_service
from app.schemas import AnalysisResponse
from app.utils.image_validation import read_and_validate_image

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
@router.post("/api/v1/analyze/image", response_model=AnalysisResponse, include_in_schema=False)
async def analyze_image(
    image: UploadFile = File(...),
    service: DetectionService = Depends(get_detection_service),
) -> AnalysisResponse:
    decoded = await read_and_validate_image(image)
    try:
        return await asyncio.to_thread(service.analyze, decoded, image.filename or "image")
    except HTTPException:
        raise
    except Exception as exc:
        # Loggers can capture the chained exception without exposing internals to clients.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local detection engine could not analyze this image. Please try another image.",
        ) from exc
