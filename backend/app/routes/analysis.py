"""Image analysis API."""

from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from pydantic import ValidationError

from app.analysis.session_cache import analysis_session_cache
from app.detection.service import DetectionService, get_detection_service
from app.schemas import AnalysisOptions, AnalysisResponse
from app.utils.image_validation import read_and_validate_image

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalysisResponse)
@router.post("/api/v1/analyze/image", response_model=AnalysisResponse, include_in_schema=False)
async def analyze_image(
    image: UploadFile = File(...),
    analysis_options: str | None = Form(None),
    performance_profile: str = Form("balanced"),
    service: DetectionService = Depends(get_detection_service),
) -> AnalysisResponse:
    decoded = await read_and_validate_image(image)
    try:
        options = AnalysisOptions.model_validate_json(analysis_options) if analysis_options else AnalysisOptions(
            performance_profile=performance_profile,
        )
        result = await asyncio.to_thread(service.analyze, decoded, image.filename or "image", options)
        result.analysis_id = analysis_session_cache.put(result, decoded.content_sha256)
        return result
    except ValidationError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_CONTENT, detail="Invalid analysis options.") from exc
    except HTTPException:
        raise
    except Exception as exc:
        # Loggers can capture the chained exception without exposing internals to clients.
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The local detection engine could not analyze this image. Please try another image.",
        ) from exc
