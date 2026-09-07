"""Secure, in-memory image upload validation and decoding."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import HTTPException, UploadFile, status

from app.config import ALLOWED_EXTENSIONS, ALLOWED_MIME_TYPES, settings


@dataclass(frozen=True)
class DecodedImage:
    pixels_bgr: np.ndarray
    width: int
    height: int
    format: str
    decode_ms: int = 0
    content_sha256: str = ""


def _detected_format(content: bytes) -> tuple[str, str] | None:
    if content.startswith(b"\xff\xd8\xff"):
        return "JPEG", "image/jpeg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "PNG", "image/png"
    if len(content) >= 12 and content[:4] == b"RIFF" and content[8:12] == b"WEBP":
        return "WEBP", "image/webp"
    return None


async def read_and_validate_image(upload: UploadFile) -> DecodedImage:
    declared_mime = (upload.content_type or "").lower()
    if declared_mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Choose a JPG, PNG or WEBP image.",
        )

    extension = Path(upload.filename or "").suffix.lower()
    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="The file extension must be JPG, JPEG, PNG or WEBP.",
        )

    content = await upload.read(settings.max_image_bytes + 1)
    await upload.close()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The uploaded image is empty.")
    if len(content) > settings.max_image_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"Image must be {settings.max_image_bytes // (1024 * 1024)} MB or smaller.",
        )

    detected = _detected_format(content)
    if detected is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The upload is not a valid JPG, PNG or WEBP image.",
        )
    image_format, actual_mime = detected
    if declared_mime != actual_mime:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The image content does not match its declared file type.",
        )

    encoded = np.frombuffer(content, dtype=np.uint8)
    decode_started = time.perf_counter()
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    decode_ms = max(0, round((time.perf_counter() - decode_started) * 1000))
    if image is None or image.ndim != 3 or image.shape[2] != 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The image is corrupted or could not be decoded.",
        )

    height, width = image.shape[:2]
    if width <= 0 or height <= 0:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="The image has invalid dimensions.")
    if width * height > 50_000_000:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="Image resolution is too large. Use an image under 50 megapixels.",
        )
    return DecodedImage(
        image, width, height, image_format, decode_ms=decode_ms,
        content_sha256=hashlib.sha256(content).hexdigest(),
    )
