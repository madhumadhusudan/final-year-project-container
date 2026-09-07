"""Deterministic and explainable privacy-risk scoring from detected evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Mapping

from app.schemas import (
    BarcodeResult, CardResult, DocumentResult, FaceResult, ImageDetails, MainSubjectDetails,
    PrivacyObjectResult, QRCodeResult,
    PrivacyRiskDetails, ProtectionMetadata, ProtectionSettings, RiskAssessment,
    RiskBreakdown, RiskFactor, RiskReductionDetails, RiskScoreSummary,
    SensitiveTextResult,
)


@dataclass(frozen=True)
class RiskConfiguration:
    """Calibrated Day 9 weights. All values are local and intentionally explicit."""

    levels: tuple[tuple[int, str], ...] = (
        (20, "LOW"), (40, "MODERATE"), (60, "ELEVATED"),
        (80, "HIGH"), (101, "CRITICAL"),
    )
    category_caps: tuple[tuple[str, float], ...] = (
        ("background_faces", 35.0), ("license_plates", 30.0),
        ("payment_cards", 35.0), ("identity_documents", 40.0),
        ("qr_codes", 30.0), ("barcodes", 15.0),
        ("sensitive_text", 60.0),
        ("context_uncertainty", 10.0),
    )


@dataclass(frozen=True)
class _Candidate:
    type: str
    category: str
    contribution: float
    reason: str
    top_risk: str
    recommendation: str


class PrivacyRiskEngine:
    """Rule-based scoring with confidence, visibility, grouping, and category caps."""

    TEXT_WEIGHTS = {
        "phone_number": 10.0,
        "email": 8.0,
        "pan_like_number": 18.0,
        "aadhaar_like_number": 22.0,
        "payment_card_number": 25.0,
        "possible_address": 10.0,
        "pincode": 3.0,
        "url": 3.0,
        "possible_expiry_date": 4.0,
    }

    def __init__(self, configuration: RiskConfiguration | None = None) -> None:
        self.configuration = configuration or RiskConfiguration()
        self._caps = dict(self.configuration.category_caps)

    @staticmethod
    def _confidence_multiplier(confidence: float) -> float:
        confidence = max(0.0, min(float(confidence), 1.0))
        return 0.7 + 0.3 * confidence

    @staticmethod
    def _visibility_multiplier(area_ratio: float, kind: str) -> float:
        ratio = max(0.0, min(float(area_ratio), 1.0))
        thresholds = {
            "face": ((0.002, 0.65), (0.01, 0.80), (0.04, 1.0), (1.01, 1.15)),
            "plate": ((0.0005, 0.65), (0.002, 0.80), (0.01, 0.95), (1.01, 1.10)),
            "card": ((0.01, 0.75), (0.05, 0.90), (0.15, 1.0), (1.01, 1.08)),
            "document": ((0.01, 0.65), (0.05, 0.80), (0.15, 0.95), (1.01, 1.08)),
            "code": ((0.0005, 0.70), (0.003, 0.85), (0.015, 1.0), (1.01, 1.08)),
            "text": ((0.0001, 0.75), (0.001, 0.90), (0.005, 1.0), (1.01, 1.08)),
        }[kind]
        return next(multiplier for upper_bound, multiplier in thresholds if ratio <= upper_bound)

    @staticmethod
    def _box_area_ratio(item: object, image_area: int) -> float:
        box = getattr(item, "bounding_box")
        area = max(0, box.x2 - box.x1) * max(0, box.y2 - box.y1)
        return area / max(1, image_area)

    @staticmethod
    def _overlap_ratio(first: object, second: object) -> float:
        a, b = getattr(first, "bounding_box"), getattr(second, "bounding_box")
        intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
            0, min(a.y2, b.y2) - max(a.y1, b.y1)
        )
        first_area = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
        return intersection / first_area

    @staticmethod
    def _iou(first: object, second: object) -> float:
        a, b = getattr(first, "bounding_box"), getattr(second, "bounding_box")
        intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(
            0, min(a.y2, b.y2) - max(a.y1, b.y1)
        )
        first_area = max(0, (a.x2 - a.x1) * (a.y2 - a.y1))
        second_area = max(0, (b.x2 - b.x1) * (b.y2 - b.y1))
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    @classmethod
    def _deduplicate_text(cls, items: Iterable[SensitiveTextResult]) -> list[SensitiveTextResult]:
        accepted: list[SensitiveTextResult] = []
        for candidate in sorted(items, key=lambda item: (-item.confidence, item.id)):
            duplicate = any(
                candidate.type == existing.type
                and (
                    candidate.masked_value.casefold() == existing.masked_value.casefold()
                    or cls._iou(candidate, existing) >= 0.5
                )
                for existing in accepted
            )
            if not duplicate:
                accepted.append(candidate)
        return sorted(accepted, key=lambda item: item.id)

    @staticmethod
    def _severity(contribution: float) -> str:
        if contribution < 5:
            return "low"
        if contribution < 12:
            return "medium"
        if contribution < 20:
            return "high"
        return "critical"

    def level_for_score(self, score: int) -> str:
        bounded = max(0, min(int(score), 100))
        return next(label for upper_bound, label in self.configuration.levels if bounded < upper_bound)

    @staticmethod
    def _allocate_breakdown(values: Mapping[str, float], target: int) -> dict[str, int]:
        """Largest-remainder allocation keeps displayed categories equal to the score."""
        floors = {key: int(value) for key, value in values.items()}
        remaining = target - sum(floors.values())
        order = sorted(values, key=lambda key: (-(values[key] - floors[key]), key))
        for key in order[:max(0, remaining)]:
            floors[key] += 1
        return floors

    def _assessment(self, statuses: Mapping[str, str]) -> RiskAssessment:
        unavailable = sorted(module for module, status in statuses.items() if status not in {"completed", "skipped"})
        return RiskAssessment(
            status="partial" if unavailable else "complete",
            unavailable_modules=unavailable,
        )

    def _face_candidates(
        self, faces: list[FaceResult], main_subject: MainSubjectDetails,
    ) -> tuple[list[_Candidate], int]:
        if main_subject.status == "identified":
            exposed = [
                face for face in faces
                if face.face_id != main_subject.face_id and face.role != "document_face"
            ]
        else:
            # With no reliable preservation target, every visible face is an exposure.
            exposed = [face for face in faces if face.role != "document_face"]
        candidates = []
        for face in exposed:
            contribution = (
                12.0 * self._confidence_multiplier(face.confidence)
                * self._visibility_multiplier(face.area_ratio, "face")
            )
            candidates.append(_Candidate(
                "background_face", "background_faces", contribution,
                "A visible non-primary face may identify another person.",
                "Background face visible", "Protect background faces.",
            ))
        return candidates, len(exposed)

    def _object_candidates(
        self, objects: list[PrivacyObjectResult], texts: list[SensitiveTextResult],
        image_area: int, kind: str,
    ) -> tuple[list[_Candidate], set[int]]:
        candidates, grouped_text_ids = [], set()
        for item in objects:
            related_types = (
                {"license_plate_text"} if kind == "plate"
                else {"payment_card_number", "possible_expiry_date"}
            )
            related = [
                text for text in texts
                if text.type in related_types and self._overlap_ratio(text, item) >= 0.5
            ]
            grouped_text_ids.update(text.id for text in related)
            confidence = self._confidence_multiplier(item.confidence)
            visibility = self._visibility_multiplier(self._box_area_ratio(item, image_area), kind)
            if kind == "plate":
                readable_bonus = min(4.0, sum(3.0 * self._confidence_multiplier(text.confidence) for text in related))
                contribution = 18.0 * confidence * visibility + readable_bonus
                reason = "A confirmed license plate is visible."
                if related:
                    reason = "A confirmed license plate is visible and OCR indicates it is readable."
                candidates.append(_Candidate(
                    "license_plate", "license_plates", contribution, reason,
                    "License plate exposed", "Anonymize visible license plates.",
                ))
            else:
                number_items = [text for text in related if text.type == "payment_card_number"]
                expiry_items = [text for text in related if text.type == "possible_expiry_date"]
                readable_bonus = min(
                    7.0,
                    sum((5.0 if text.luhn_valid else 3.5) * self._confidence_multiplier(text.confidence)
                        for text in number_items)
                    + sum(1.5 * self._confidence_multiplier(text.confidence) for text in expiry_items),
                )
                contribution = 27.0 * confidence * visibility + readable_bonus
                reason = "A confirmed payment-card region is visible."
                if related:
                    reason = "A confirmed payment card is visible with readable sensitive card content."
                candidates.append(_Candidate(
                    "payment_card", "payment_cards", contribution, reason,
                    "Payment card exposed", "Hide the complete payment-card region.",
                ))
        return candidates, grouped_text_ids

    def _document_candidates(
        self, documents: list[DocumentResult], texts: list[SensitiveTextResult], image_area: int,
    ) -> tuple[list[_Candidate], set[int]]:
        weights = {
            "aadhaar_card": 28.0, "pan_card": 23.0, "passport": 28.0,
            "driving_license": 23.0, "identity_document": 20.0,
        }
        display_names = {
            "aadhaar_card": "Aadhaar-like identity document",
            "pan_card": "PAN-like identity document", "passport": "passport",
            "driving_license": "driving licence", "identity_document": "identity document",
        }
        candidates: list[_Candidate] = []
        grouped_ids: set[int] = set()
        by_id = {item.id: item for item in texts}
        for document in documents:
            related = [by_id[item_id] for item_id in document.sensitive_text_ids if item_id in by_id]
            grouped_ids.update(item.id for item in related)
            confidence = self._confidence_multiplier(
                max(document.confidence, document.classification_confidence)
            )
            visibility = self._visibility_multiplier(document.area_ratio, "document")
            sensitive_bonus = min(
                4.0, sum(2.5 * self._confidence_multiplier(item.confidence) for item in related)
            )
            readable_bonus = min(2.0, 0.5 * len(document.ocr_text_ids))
            contribution = weights[document.final_document_type] * confidence * visibility
            contribution += sensitive_bonus + readable_bonus
            name = display_names[document.final_document_type]
            reason = f"A detected {name} exposes a complete high-sensitivity document region."
            if related or document.ocr_text_ids:
                reason = f"A detected {name} is visible with readable OCR context inside its region."
            candidates.append(_Candidate(
                document.final_document_type, "identity_documents", contribution, reason,
                f"{name.capitalize()} exposed", "Protect the complete identity-document region.",
            ))
        return candidates, grouped_ids

    def _text_candidates(
        self, texts: list[SensitiveTextResult], grouped_ids: set[int], image_area: int,
    ) -> list[_Candidate]:
        available = [text for text in texts if text.id not in grouped_ids]
        addresses = [text for text in available if text.type == "possible_address"]
        pincodes = [text for text in available if text.type == "pincode"]
        address_bonus = min(3.0, sum(2.0 * self._confidence_multiplier(item.confidence) for item in pincodes))
        candidates = []
        for text in available:
            if text.type == "pincode" and addresses:
                continue
            weight = self.TEXT_WEIGHTS.get(text.type)
            if weight is None:
                continue
            contribution = (
                weight * self._confidence_multiplier(text.confidence)
                * self._visibility_multiplier(self._box_area_ratio(text, image_area), "text")
            )
            reason_by_type = {
                "phone_number": "A visible phone number can expose direct contact information.",
                "email": "A visible email address can expose contact information.",
                "pan_like_number": "A high-sensitivity PAN-like pattern is visible; it is not government-verified.",
                "aadhaar_like_number": "A high-sensitivity Aadhaar-like pattern is visible; identity is not verified.",
                "payment_card_number": "A payment-card-number-like sequence is visible.",
                "possible_address": "Possible address information is visible.",
                "pincode": "A PIN code is visible without stronger address context.",
                "url": "A web address classified as sensitive text is visible.",
                "possible_expiry_date": "A possible payment-card expiry date is visible.",
            }
            if text.type == "possible_address" and address_bonus:
                contribution += address_bonus
                reason_by_type["possible_address"] = "Possible address information and an associated PIN code are visible."
            label_by_type = {
                "phone_number": "Phone number visible", "email": "Email address visible",
                "pan_like_number": "PAN-like data visible", "aadhaar_like_number": "Aadhaar-like data visible",
                "payment_card_number": "Payment-card-number-like data visible",
                "possible_address": "Possible address visible", "pincode": "PIN code visible",
                "url": "Web address visible", "possible_expiry_date": "Possible card expiry visible",
            }
            candidates.append(_Candidate(
                text.type, "sensitive_text", contribution, reason_by_type[text.type],
                label_by_type[text.type], "Protect sensitive text before sharing.",
            ))
        return candidates

    def _code_candidates(
        self, qr_codes: list[QRCodeResult], barcodes: list[BarcodeResult], image_area: int,
    ) -> list[_Candidate]:
        qr_weights = {
            "payment": 18.0, "contact": 16.0, "wifi": 16.0, "url": 7.0,
            "identifier": 9.0, "text": 7.0, "unknown": 8.0,
        }
        candidates: list[_Candidate] = []
        for item in qr_codes:
            visibility = self._visibility_multiplier(self._box_area_ratio(item, image_area), "code")
            if item.parent_type == "identity_document":
                contribution = 6.0 * visibility
                factor_type = "document_qr"
                reason = "A QR code is visible inside an identity document; only a limited readability bonus is added."
            elif item.parent_type == "payment_card":
                contribution = 4.0 * visibility
                factor_type = "card_qr"
                reason = "A QR code is visible inside a confirmed card region."
            else:
                contribution = qr_weights[item.content_type] * visibility
                factor_type = f"qr_{item.content_type}"
                descriptions = {
                    "payment": "A locally decoded payment QR is visible.",
                    "contact": "A QR encoding contact information is visible.",
                    "wifi": "A QR encoding Wi-Fi credentials is visible.",
                    "url": "A QR encoding a web address is visible.",
                    "identifier": "A QR encoding an identifier is visible.",
                    "text": "A QR encoding text is visible.",
                    "unknown": "A QR code with unknown or undecodable content is visible.",
                }
                reason = descriptions[item.content_type]
            candidates.append(_Candidate(
                factor_type, "qr_codes", contribution, reason,
                "QR code exposed", "Protect visible QR codes before sharing.",
            ))

        for item in barcodes:
            visibility = self._visibility_multiplier(self._box_area_ratio(item, image_area), "code")
            if item.parent_type == "identity_document":
                contribution, factor_type = 4.0 * visibility, "document_barcode"
                reason = "A barcode is visible inside an identity document; only a limited context bonus is added."
            elif item.parent_type == "payment_card":
                contribution, factor_type = 3.0 * visibility, "card_barcode"
                reason = "A barcode is visible inside a confirmed card region."
            elif item.decoded and item.format in {"EAN-8", "EAN-13", "UPC-A", "UPC-E"}:
                contribution, factor_type = 2.0 * visibility, "product_barcode"
                reason = "A decoded retail-format barcode is visible without sensitive parent context."
            elif not item.decoded:
                contribution, factor_type = 5.0 * visibility, "unknown_barcode"
                reason = "A barcode with undecodable content is visible."
            else:
                contribution, factor_type = 6.0 * visibility, "identifier_barcode"
                reason = "A barcode encoding an identifier is visible."
            candidates.append(_Candidate(
                factor_type, "barcodes", contribution, reason,
                "Barcode exposed", "Protect sensitive or contextual barcodes before sharing.",
            ))
        return candidates

    @staticmethod
    def _protection_effectiveness(settings: ProtectionSettings) -> float:
        if settings.anonymization_method == "blackout":
            return 1.0
        table = {
            "blur": {"low": 0.80, "medium": 0.92, "high": 0.97},
            "pixelate": {"low": 0.85, "medium": 0.92, "high": 0.96},
        }
        return table[settings.anonymization_method][settings.strength]

    def _apply_protection(
        self, candidates: list[_Candidate], settings: ProtectionSettings,
        protection: ProtectionMetadata,
    ) -> list[_Candidate]:
        effectiveness = self._protection_effectiveness(settings)
        toggle = {
            "background_faces": settings.protect_background_faces,
            "license_plates": settings.protect_license_plates,
            "payment_cards": settings.protect_cards,
            "identity_documents": settings.protect_identity_documents,
            "qr_codes": settings.protect_qr_codes,
            "barcodes": settings.protect_barcodes,
            "sensitive_text": settings.protect_sensitive_text,
            "context_uncertainty": settings.protect_background_faces,
        }
        actual_counts = {
            "background_faces": protection.breakdown.background_faces,
            "license_plates": protection.breakdown.license_plates,
            "payment_cards": protection.breakdown.cards,
            "identity_documents": protection.breakdown.identity_documents,
            "qr_codes": protection.breakdown.qr_codes,
            "barcodes": protection.breakdown.barcodes,
            "sensitive_text": protection.breakdown.sensitive_text,
        }
        eligible_counts = {
            category: sum(candidate.category == category for candidate in candidates)
            for category in actual_counts
        }
        residual = []
        for candidate in candidates:
            fraction = 0.0
            if candidate.type.startswith("document_") and settings.protect_identity_documents:
                fraction = min(1.0, actual_counts["identity_documents"])
            elif candidate.type.startswith("card_") and settings.protect_cards:
                fraction = min(1.0, actual_counts["payment_cards"])
            elif toggle[candidate.category]:
                if candidate.category == "context_uncertainty":
                    face_count = eligible_counts["background_faces"]
                    fraction = min(1.0, actual_counts["background_faces"] / max(1, face_count))
                else:
                    fraction = min(
                        1.0,
                        actual_counts[candidate.category] / max(1, eligible_counts[candidate.category]),
                    )
            remaining = candidate.contribution * (1.0 - effectiveness * fraction)
            if remaining >= 0.005:
                residual.append(replace(candidate, contribution=remaining))
        return residual

    def _build_result(
        self, candidates: list[_Candidate], assessment: RiskAssessment,
        counts: Mapping[str, int],
    ) -> PrivacyRiskDetails:
        category_raw = {category: 0.0 for category in self._caps}
        for candidate in candidates:
            category_raw[candidate.category] += candidate.contribution
        category_capped = {
            category: min(total, self._caps[category]) for category, total in category_raw.items()
        }
        capped_total = sum(category_capped.values())
        total_scale = min(1.0, 100.0 / capped_total) if capped_total else 1.0
        normalized = {key: value * total_scale for key, value in category_capped.items()}
        score = max(0, min(100, round(sum(normalized.values()))))
        breakdown_values = self._allocate_breakdown(normalized, score)

        category_scales = {
            category: (
                normalized[category] / category_raw[category] if category_raw[category] else 0.0
            ) for category in category_raw
        }
        factors = [
            RiskFactor(
                type=candidate.type,
                category=candidate.category,
                severity=self._severity(candidate.contribution * category_scales[candidate.category]),
                contribution=round(candidate.contribution * category_scales[candidate.category], 1),
                reason=candidate.reason,
            )
            for candidate in candidates
            if candidate.contribution * category_scales[candidate.category] >= 0.05
        ]
        factors.sort(key=lambda factor: (-factor.contribution, factor.type))

        category_messages = {
            "payment_cards": f"{counts.get('payment_cards', 0)} payment card(s) exposed",
            "identity_documents": f"{counts.get('identity_documents', 0)} identity document(s) exposed",
            "qr_codes": f"{counts.get('qr_codes', 0)} QR code(s) exposed",
            "barcodes": f"{counts.get('barcodes', 0)} barcode(s) exposed",
            "background_faces": f"{counts.get('background_faces', 0)} background face(s) visible",
            "license_plates": f"{counts.get('license_plates', 0)} license plate(s) exposed",
            "context_uncertainty": "Main subject is uncertain",
        }
        ranked_messages = [
            (normalized[category], message) for category, message in category_messages.items()
            if normalized[category] > 0
        ]
        sensitive = [candidate for candidate in candidates if candidate.category == "sensitive_text"]
        if sensitive:
            highest = max(sensitive, key=lambda item: item.contribution)
            ranked_messages.append((normalized["sensitive_text"], highest.top_risk))
        top_risks = [message for _, message in sorted(ranked_messages, key=lambda item: (-item[0], item[1]))[:3]]

        recommendations = []
        for candidate in sorted(candidates, key=lambda item: -item.contribution):
            if candidate.recommendation not in recommendations:
                recommendations.append(candidate.recommendation)

        summaries = {
            "LOW": "No or limited detected privacy-sensitive exposure is visible.",
            "MODERATE": "Some privacy-sensitive elements are visible.",
            "ELEVATED": "Several or clearly visible privacy-sensitive elements were detected.",
            "HIGH": "Multiple high-impact privacy-sensitive elements are visible.",
            "CRITICAL": "Critical privacy exposure was detected across the image.",
        }
        level = self.level_for_score(score)
        return PrivacyRiskDetails(
            score=score, level=level, summary=summaries[level],
            breakdown=RiskBreakdown(**breakdown_values), factors=factors,
            top_risks=top_risks, recommendations=recommendations, assessment=assessment,
        )

    def calculate(
        self, image_metadata: ImageDetails, faces: list[FaceResult], main_subject: MainSubjectDetails,
        plates: list[PrivacyObjectResult], cards: list[CardResult],
        sensitive_text: list[SensitiveTextResult], detector_statuses: Mapping[str, str],
        protection_settings: ProtectionSettings | None = None,
        protection_metadata: ProtectionMetadata | None = None,
        documents: list[DocumentResult] | None = None,
        qr_codes: list[QRCodeResult] | None = None,
        barcodes: list[BarcodeResult] | None = None,
    ) -> PrivacyRiskDetails:
        """Calculate original or metadata-derived residual risk without rerunning detectors."""
        image_area = image_metadata.width * image_metadata.height
        texts = self._deduplicate_text(sensitive_text)
        face_candidates, face_count = self._face_candidates(faces, main_subject)
        plate_candidates, plate_text_ids = self._object_candidates(plates, texts, image_area, "plate")
        card_candidates, card_text_ids = self._object_candidates(cards, texts, image_area, "card")
        document_candidates, document_text_ids = self._document_candidates(
            documents or [], texts, image_area,
        )
        text_candidates = self._text_candidates(
            texts, plate_text_ids | card_text_ids | document_text_ids, image_area,
        )
        code_candidates = self._code_candidates(qr_codes or [], barcodes or [], image_area)
        candidates = (
            face_candidates + plate_candidates + card_candidates
            + document_candidates + code_candidates + text_candidates
        )
        if main_subject.status == "uncertain" and face_count:
            candidates.append(_Candidate(
                "main_subject_uncertainty", "context_uncertainty", 8.0,
                "The system cannot confidently determine which person should remain visible.",
                "Main subject is uncertain", "Review which person should remain visible.",
            ))
        counts = {
            "background_faces": face_count,
            "license_plates": len(plates),
            "payment_cards": len(cards),
            "identity_documents": len(documents or []),
            "qr_codes": len(qr_codes or []),
            "barcodes": len(barcodes or []),
        }
        if protection_settings is not None and protection_metadata is not None:
            candidates = self._apply_protection(candidates, protection_settings, protection_metadata)
        return self._build_result(candidates, self._assessment(detector_statuses), counts)

    @staticmethod
    def compare(before: PrivacyRiskDetails, after: PrivacyRiskDetails) -> RiskReductionDetails:
        reduction = max(0, before.score - after.score)
        percent = round((reduction / before.score) * 100, 1) if before.score else 0.0
        return RiskReductionDetails(
            before=RiskScoreSummary(score=before.score, level=before.level),
            after=RiskScoreSummary(score=after.score, level=after.level),
            reduction=reduction, reduction_percent=percent,
        )
