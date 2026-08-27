"""Typed API and internal detection models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, Field


Category = Literal["face", "person", "background_person"]


@dataclass(frozen=True)
class RawDetection:
    category: Literal["face", "person"]
    confidence: float
    x: int
    y: int
    width: int
    height: int
    source: str
    is_main_subject: bool = False
    subject_score: float | None = None
    explanation: str = ""

    @property
    def x2(self) -> int:
        return self.x + self.width

    @property
    def y2(self) -> int:
        return self.y + self.height

    def clipped(self, image_width: int, image_height: int) -> RawDetection:
        x1 = max(0, min(self.x, image_width))
        y1 = max(0, min(self.y, image_height))
        x2 = max(x1, min(self.x2, image_width))
        y2 = max(y1, min(self.y2, image_height))
        return replace(self, x=x1, y=y1, width=x2 - x1, height=y2 - y1)


class BoundingBox(BaseModel):
    x: int = Field(ge=0)
    y: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)


class NormalizedBoundingBox(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(ge=0, le=1)
    height: float = Field(ge=0, le=1)


class CenterPoint(BaseModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class DetectionResult(BaseModel):
    id: str
    category: Category
    label: str
    confidence: float = Field(ge=0, le=1)
    boundingBox: BoundingBox
    normalizedBoundingBox: NormalizedBoundingBox
    center: CenterPoint
    relativeArea: float = Field(ge=0, le=1)
    isMainSubject: bool
    subjectScore: float | None = Field(default=None, ge=0, le=1)
    recommendedAnonymization: bool
    explanation: str
    source: str


class ImageDetails(BaseModel):
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str


class DetectionSummary(BaseModel):
    totalObjects: int = Field(ge=0)
    faces: int = Field(ge=0)
    people: int = Field(ge=0)
    backgroundFaces: int = Field(ge=0)
    mainSubjectDetected: bool


class AnalysisResponse(BaseModel):
    success: Literal[True] = True
    image: ImageDetails
    detections: list[DetectionResult]
    summary: DetectionSummary
    processingTimeMs: int = Field(ge=0)
