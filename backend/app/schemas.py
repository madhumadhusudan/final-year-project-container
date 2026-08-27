"""Typed API and internal detection models."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from pydantic import BaseModel, Field


@dataclass(frozen=True)
class RawDetection:
    class_id: int
    class_name: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    def clipped(self, image_width: int, image_height: int) -> RawDetection:
        x1 = max(0, min(self.x1, image_width))
        y1 = max(0, min(self.y1, image_height))
        x2 = max(x1, min(self.x2, image_width))
        y2 = max(y1, min(self.y2, image_height))
        return replace(self, x1=x1, y1=y1, x2=x2, y2=y2)


class BoundingBox(BaseModel):
    x1: int = Field(ge=0)
    y1: int = Field(ge=0)
    x2: int = Field(ge=0)
    y2: int = Field(ge=0)


class DetectionResult(BaseModel):
    id: int = Field(gt=0)
    class_id: int = Field(ge=0)
    class_name: str
    confidence: float = Field(ge=0, le=1)
    bounding_box: BoundingBox


class ImageDetails(BaseModel):
    filename: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    format: str


class AnalysisDetails(BaseModel):
    status: Literal["completed"] = "completed"
    model: str
    detection_count: int = Field(ge=0)
    detections: list[DetectionResult]


class PerformanceDetails(BaseModel):
    inference_time_ms: int = Field(ge=0)


class AnalysisResponse(BaseModel):
    status: Literal["success"] = "success"
    image: ImageDetails
    analysis: AnalysisDetails
    performance: PerformanceDetails
