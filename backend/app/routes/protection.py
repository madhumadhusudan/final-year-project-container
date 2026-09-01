"""In-memory selective image protection API."""

from __future__ import annotations

import asyncio
from urllib.parse import quote

import cv2
from fastapi import APIRouter, File, Form, HTTPException, Response, UploadFile, status
from pydantic import ValidationError

from app.anonymization import ImageAnonymizer
from app.privacy.risk_score import PrivacyRiskEngine
from app.schemas import AnalysisResponse, ProtectionSettings
from app.utils.image_validation import DecodedImage, read_and_validate_image

router = APIRouter(tags=["protection"])


def _parse_payload(model_type, raw_value: str, label: str):
    try:
        return model_type.model_validate_json(raw_value)
    except (ValidationError, ValueError, TypeError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_CONTENT,
            detail=f"Invalid {label} payload.",
        ) from exc


def _encode_image(decoded: DecodedImage, pixels_bgr) -> tuple[bytes, str, str]:
    formats = {
        "JPEG": (".jpg", "image/jpeg", [cv2.IMWRITE_JPEG_QUALITY, 95]),
        "PNG": (".png", "image/png", [cv2.IMWRITE_PNG_COMPRESSION, 3]),
        "WEBP": (".webp", "image/webp", [cv2.IMWRITE_WEBP_QUALITY, 95]),
    }
    extension, media_type, parameters = formats.get(decoded.format, formats["PNG"])
    ok, encoded = cv2.imencode(extension, pixels_bgr, parameters)
    if not ok:
        raise RuntimeError("Protected image encoding failed.")
    return encoded.tobytes(), media_type, extension


def _protect(decoded: DecodedImage, analysis: AnalysisResponse, settings: ProtectionSettings):
    if analysis.image.width != decoded.width or analysis.image.height != decoded.height:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="The analysis does not match the uploaded image dimensions. Analyze this image again.",
        )
    result = ImageAnonymizer().anonymize(decoded.pixels_bgr, analysis, settings)
    details = analysis.analysis
    statuses = {
        "face_detection": details.face_detection.status,
        "license_plate_detection": details.license_plate_detection.status,
        "card_detection": details.card_detection.status,
        "document_detection": details.document_detection.status,
        "qr_detection": details.qr_detection.status,
        "barcode_detection": details.barcode_detection.status,
        "ocr": details.ocr.status,
    }
    engine = PrivacyRiskEngine()
    before = engine.calculate(
        analysis.image, details.face_detection.faces, details.main_subject,
        details.license_plate_detection.plates, details.card_detection.cards,
        details.sensitive_text.items, statuses, documents=details.document_detection.documents,
        qr_codes=details.qr_detection.items, barcodes=details.barcode_detection.items,
    )
    after = engine.calculate(
        analysis.image, details.face_detection.faces, details.main_subject,
        details.license_plate_detection.plates, details.card_detection.cards,
        details.sensitive_text.items, statuses, settings, result.protection,
        documents=details.document_detection.documents,
        qr_codes=details.qr_detection.items, barcodes=details.barcode_detection.items,
    )
    result.protection.risk = engine.compare(before, after)
    return result, _encode_image(decoded, result.pixels_bgr)


@router.post("/protect")
async def protect_image(
    image: UploadFile = File(...),
    analysis: str = Form(...),
    settings: str = Form(...),
) -> Response:
    decoded = await read_and_validate_image(image)
    analysis_payload = _parse_payload(AnalysisResponse, analysis, "analysis")
    privacy_settings = _parse_payload(ProtectionSettings, settings, "privacy settings")
    try:
        result, (content, media_type, extension) = await asyncio.to_thread(
            _protect, decoded, analysis_payload, privacy_settings
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="The image could not be protected locally. Please try again.",
        ) from exc
    metadata = result.protection.model_dump_json()
    return Response(
        content=content,
        media_type=media_type,
        headers={
            "X-Protection-Metadata": quote(metadata, safe=""),
            "Content-Disposition": f'attachment; filename="privacy-protected{extension}"',
            "Cache-Control": "no-store",
        },
    )
