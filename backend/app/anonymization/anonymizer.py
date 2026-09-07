"""Local, region-specific blur, pixelation, and blackout operations."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from app.schemas import AnalysisResponse, ProtectionBreakdown, ProtectionMetadata, ProtectionSettings

from .region_utils import Region, contains_region, merge_regions, pad_region, sanitize_region


FACE_PADDING = 0.12
PLATE_PADDING = 0.06
CARD_PADDING = 0.04
DOCUMENT_PADDING = 0.05
QR_PADDING = 0.09
BARCODE_PADDING = 0.06
TEXT_PADDING = 0.08


@dataclass(frozen=True)
class AnonymizationResult:
    pixels_bgr: np.ndarray
    protection: ProtectionMetadata


class ImageAnonymizer:
    """Apply a selected privacy method only to validated sensitive regions."""

    blur_ratios = {"low": 0.10, "medium": 0.20, "high": 0.34}
    pixel_divisors = {"low": 16, "medium": 9, "high": 5}

    @staticmethod
    def _odd_kernel(limit: int, desired: int) -> int:
        largest = limit if limit % 2 == 1 else limit - 1
        candidate = desired if desired % 2 == 1 else desired + 1
        return max(1, min(largest, candidate))

    def blur_region(self, image: np.ndarray, box: object, strength: str = "medium") -> bool:
        region = sanitize_region(box, image.shape[1], image.shape[0])
        if region is None:
            return False
        roi = image[region.y1:region.y2, region.x1:region.x2]
        ratio = self.blur_ratios.get(strength, self.blur_ratios["medium"])
        kernel_x = self._odd_kernel(region.width, max(3, round(region.width * ratio)))
        kernel_y = self._odd_kernel(region.height, max(3, round(region.height * ratio)))
        if kernel_x == 1 and kernel_y == 1:
            return False
        image[region.y1:region.y2, region.x1:region.x2] = cv2.GaussianBlur(
            roi, (kernel_x, kernel_y), sigmaX=0, sigmaY=0
        )
        return True

    def pixelate_region(self, image: np.ndarray, box: object, strength: str = "medium") -> bool:
        region = sanitize_region(box, image.shape[1], image.shape[0])
        if region is None:
            return False
        roi = image[region.y1:region.y2, region.x1:region.x2]
        divisor = self.pixel_divisors.get(strength, self.pixel_divisors["medium"])
        target_width = max(1, min(region.width, divisor))
        target_height = max(1, min(region.height, divisor))
        if target_width == region.width and target_height == region.height:
            return False
        reduced = cv2.resize(roi, (target_width, target_height), interpolation=cv2.INTER_AREA)
        image[region.y1:region.y2, region.x1:region.x2] = cv2.resize(
            reduced, (region.width, region.height), interpolation=cv2.INTER_NEAREST
        )
        return True

    @staticmethod
    def blackout_region(image: np.ndarray, box: object) -> bool:
        region = sanitize_region(box, image.shape[1], image.shape[0])
        if region is None:
            return False
        image[region.y1:region.y2, region.x1:region.x2] = 0
        return True

    def _apply(
        self, image: np.ndarray, region: Region, settings: ProtectionSettings, category: str = "",
    ) -> bool:
        # Low-strength visual degradation can remain machine-decodable. QR and
        # barcode protection therefore has a medium-strength privacy floor.
        strength = (
            "medium"
            if category in {"qr_codes", "barcodes"} and settings.strength == "low"
            else settings.strength
        )
        if settings.anonymization_method == "blur":
            return self.blur_region(image, region, strength)
        if settings.anonymization_method == "pixelate":
            return self.pixelate_region(image, region, strength)
        return self.blackout_region(image, region)

    @staticmethod
    def _expanded(box: object, ratio: float, width: int, height: int) -> Region | None:
        region = sanitize_region(box, width, height)
        return pad_region(region, ratio, width, height) if region else None

    def anonymize(
        self, image_bgr: np.ndarray, analysis: AnalysisResponse, settings: ProtectionSettings
    ) -> AnonymizationResult:
        image = image_bgr.copy()
        height, width = image.shape[:2]
        details = analysis.analysis
        warnings: list[str] = []
        category_regions: dict[str, list[Region]] = {
            "background_faces": [], "license_plates": [], "cards": [],
            "identity_documents": [], "qr_codes": [], "barcodes": [], "sensitive_text": [],
        }

        subject_identified = details.main_subject.status == "identified"
        preserved_face_id = details.main_subject.face_id if subject_identified else None
        preserved_region = next((
            self._expanded(face.bounding_box, FACE_PADDING, width, height)
            for face in details.face_detection.faces
            if face.face_id == preserved_face_id
        ), None)
        if settings.protect_background_faces:
            external_faces = [
                face for face in details.face_detection.faces if face.role != "document_face"
            ]
            for face in external_faces:
                if preserved_face_id is not None and face.face_id == preserved_face_id:
                    continue
                region = self._expanded(face.bounding_box, FACE_PADDING, width, height)
                if region:
                    category_regions["background_faces"].append(region)
            if external_faces and details.main_subject.status == "uncertain":
                warnings.append("Main subject could not be identified confidently. All detected faces were protected.")
            elif external_faces and details.main_subject.status == "not_found":
                warnings.append("No main subject was identified. All detected faces were protected.")

        if settings.protect_license_plates:
            category_regions["license_plates"] = [
                region for item in details.license_plate_detection.plates
                if (region := self._expanded(item.bounding_box, PLATE_PADDING, width, height)) is not None
            ]
        if settings.protect_cards:
            category_regions["cards"] = [
                region for item in details.card_detection.cards
                if (region := self._expanded(item.bounding_box, CARD_PADDING, width, height)) is not None
            ]

        if settings.protect_identity_documents:
            category_regions["identity_documents"] = [
                region for item in details.document_detection.documents
                if (region := self._expanded(item.bounding_box, DOCUMENT_PADDING, width, height)) is not None
            ]

        parent_regions = category_regions["identity_documents"] + category_regions["cards"]
        covered_code_counts = {"qr_codes": 0, "barcodes": 0}
        code_specs = (
            (
                "qr_codes", details.qr_detection.items, QR_PADDING,
                settings.protect_qr_codes,
            ),
            (
                "barcodes", details.barcode_detection.items, BARCODE_PADDING,
                settings.protect_barcodes,
            ),
        )
        for category, items, padding, enabled in code_specs:
            for item in items:
                region = self._expanded(item.bounding_box, padding, width, height)
                if region is None:
                    continue
                if any(contains_region(parent, region) for parent in parent_regions):
                    covered_code_counts[category] += 1
                elif enabled:
                    category_regions[category].append(region)

        if settings.protect_sensitive_text:
            text_regions = [
                region for item in details.sensitive_text.items
                if (region := self._expanded(item.bounding_box, TEXT_PADDING, width, height)) is not None
            ]
            covering_regions = (
                category_regions["identity_documents"]
                + category_regions["cards"] + category_regions["license_plates"]
                + category_regions["qr_codes"] + category_regions["barcodes"]
            )
            text_regions = [
                region for region in text_regions
                if not any(contains_region(container, region) for container in covering_regions)
            ]
            category_regions["sensitive_text"] = merge_regions(text_regions)

        counts: dict[str, int] = {}
        applied_counts: dict[str, int] = {}
        for category in (
            "identity_documents", "cards", "license_plates", "qr_codes", "barcodes",
            "sensitive_text", "background_faces",
        ):
            applied = sum(
                self._apply(image, region, settings, category) for region in category_regions[category]
            )
            applied_counts[category] = applied
            counts[category] = applied + covered_code_counts.get(category, 0)

        # Face preservation has final precedence over intersecting privacy boxes.
        if preserved_region is not None:
            image[preserved_region.y1:preserved_region.y2, preserved_region.x1:preserved_region.x2] = image_bgr[
                preserved_region.y1:preserved_region.y2, preserved_region.x1:preserved_region.x2
            ]

        breakdown = ProtectionBreakdown(**counts)
        metadata = ProtectionMetadata(
            method=settings.anonymization_method,
            strength=settings.strength,
            regions_protected=sum(applied_counts.values()),
            breakdown=breakdown,
            main_subject_preserved=preserved_region is not None,
            warnings=warnings,
        )
        return AnonymizationResult(image, metadata)
