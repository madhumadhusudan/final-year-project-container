"""Explainable OCR-assisted identity-document classification."""

from __future__ import annotations

import re

from app.schemas import (
    BoundingBox, DocumentResult, OCRTextResult, Point, SensitiveTextResult,
)


EXPLICIT_TYPES = {
    "aadhaar": "aadhaar_card", "aadhar": "aadhaar_card",
    "aadhaar_card": "aadhaar_card", "aadhar_card": "aadhaar_card",
    "pan": "pan_card", "pan_card": "pan_card",
    "permanent_account_number_card": "pan_card",
    "passport": "passport", "indian_passport": "passport",
    "driving_license": "driving_license", "driving_licence": "driving_license",
    "driver_license": "driving_license", "driver_licence": "driving_license",
    "drivers_license": "driving_license", "drivers_licence": "driving_license",
}

CONTEXT_TERMS = {
    "aadhaar_card": (
        re.compile(r"\baadhaa?r\b", re.I),
        re.compile(r"\buidai\b|unique identification authority", re.I),
    ),
    "pan_card": (
        re.compile(r"permanent account number|income tax", re.I),
        re.compile(r"\bpan\b", re.I),
    ),
    "passport": (
        re.compile(r"\bpassport\b", re.I),
        re.compile(r"republic of india|nationality|passport\s*(?:no|number)", re.I),
    ),
    "driving_license": (
        re.compile(r"driving\s+licen[cs]e", re.I),
        re.compile(r"\bdl\s*(?:no|number)\b|date of birth|\bdob\b", re.I),
    ),
}

SENSITIVE_SUPPORT = {
    "aadhaar_card": "aadhaar_like_number",
    "pan_card": "pan_like_number",
}


def _overlap_ratio(box: BoundingBox, other: BoundingBox) -> float:
    intersection = max(0, min(box.x2, other.x2) - max(box.x1, other.x1)) * max(
        0, min(box.y2, other.y2) - max(box.y1, other.y1)
    )
    area = max(1, (box.x2 - box.x1) * (box.y2 - box.y1))
    return intersection / area


class DocumentClassifier:
    """Combine true model labels with only OCR evidence inside each document box."""

    @staticmethod
    def _context(
        document_type: str, texts: list[OCRTextResult], sensitive: list[SensitiveTextResult],
    ) -> tuple[int, list[str]]:
        joined = " ".join(text.normalized_text for text in texts)
        reasons: list[str] = []
        score = 0
        for pattern in CONTEXT_TERMS[document_type]:
            if pattern.search(joined):
                score += 1
                reasons.append(f"OCR contains supporting {document_type.replace('_', ' ')} terminology.")
        expected_sensitive = SENSITIVE_SUPPORT.get(document_type)
        if expected_sensitive and any(item.type == expected_sensitive for item in sensitive):
            score += 2
            reasons.append(f"A masked {expected_sensitive.replace('_', ' ')} was detected inside the document region.")
        return score, reasons

    def classify(
        self, document_id: int, model_class: str, model_confidence: float,
        bounding_box: BoundingBox, image_width: int, image_height: int,
        texts: list[OCRTextResult], sensitive_items: list[SensitiveTextResult],
    ) -> DocumentResult:
        region_texts = [text for text in texts if _overlap_ratio(text.bounding_box, bounding_box) >= 0.5]
        region_sensitive = [
            item for item in sensitive_items if _overlap_ratio(item.bounding_box, bounding_box) >= 0.5
        ]
        reasons = [f"Dedicated document detector predicted the model-native class '{model_class}'."]
        explicit_type = EXPLICIT_TYPES.get(model_class)
        final_type = explicit_type or "identity_document"
        classification_status = "model_confirmed" if explicit_type else "uncertain"
        classification_confidence = model_confidence if explicit_type else min(0.69, model_confidence * 0.75)

        if explicit_type:
            context_score, context_reasons = self._context(explicit_type, region_texts, region_sensitive)
            if context_score:
                classification_status = "context_supported"
                classification_confidence = min(1.0, model_confidence + min(0.12, 0.04 * context_score))
                reasons.extend(context_reasons)
        else:
            context_results = {
                kind: self._context(kind, region_texts, region_sensitive)
                for kind in CONTEXT_TERMS
            }
            ranked = sorted(context_results.items(), key=lambda item: (-item[1][0], item[0]))
            best_type, (best_score, best_reasons) = ranked[0]
            second_score = ranked[1][1][0]
            # Generic model boxes require multiple independent context clues and a clear winner.
            if best_score >= 2 and best_score > second_score:
                final_type = best_type
                classification_status = "context_supported"
                classification_confidence = min(0.92, 0.55 * model_confidence + 0.12 * best_score)
                reasons.extend(best_reasons)
            else:
                reasons.append("OCR context was insufficient for a more specific identity-document type.")

        area = (bounding_box.x2 - bounding_box.x1) * (bounding_box.y2 - bounding_box.y1)
        center_x = (bounding_box.x1 + bounding_box.x2) / 2
        center_y = (bounding_box.y1 + bounding_box.y2) / 2
        return DocumentResult(
            document_id=document_id, class_name=model_class, confidence=round(model_confidence, 6),
            bounding_box=bounding_box, area_ratio=round(area / max(1, image_width * image_height), 8),
            center=Point(x=center_x, y=center_y), final_document_type=final_type,
            classification_confidence=round(classification_confidence, 6),
            classification_status=classification_status, classification_reasons=reasons,
            ocr_text_ids=[text.text_id for text in region_texts],
            sensitive_text_ids=[item.id for item in region_sensitive],
        )
