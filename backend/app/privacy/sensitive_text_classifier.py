"""Deterministic, explainable sensitive-text classification and masking."""

from __future__ import annotations

import re

from app.schemas import OCRTextResult, PrivacyObjectResult, SensitiveTextResult

EMAIL = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
URL = re.compile(r"(?i)\b(?:https?://|www\.)[^\s]+|\b[A-Z0-9.-]+\.(?:com|org|net|in|io)(?:/[^\s]*)?\b")
PAN = re.compile(r"(?i)\b[A-Z]{5}[0-9]{4}[A-Z]\b")
PHONE = re.compile(r"(?<!\d)(?:(?:\+91|0)[\s-]?)?[6-9]\d{4}[\s-]?\d{5}(?!\d)")
AADHAAR = re.compile(r"(?<!\d)(?:\d{4}[ -]?){2}\d{4}(?!\d)")
CARD = re.compile(r"(?<!\d)(?:\d[ -]?){12,18}\d(?!\d)")
EXPIRY = re.compile(r"(?<!\d)(?:0[1-9]|1[0-2])/(?:\d{2}|\d{4})(?!\d)")
PINCODE = re.compile(r"(?<!\d)[1-9]\d{5}(?!\d)")
ADDRESS_WORDS = re.compile(r"(?i)\b(?:road|rd|street|st|nagar|layout|main|cross|district|taluk|village|pin|pincode)\b")
NUMERIC_CONTEXT = re.compile(r"(?i)\b(?:phone|mobile|telephone|tel|aadhaar|aadhar|card|pin|pincode)\b")


def _contextual_numeric_value(value: str) -> str:
    """Repair a narrow OCR O/0 confusion only when an explicit numeric label exists.

    Applying this globally would turn ordinary words into identifiers and increase
    false positives. The candidate must retain enough digits for the downstream
    format validators to make the final decision.
    """
    if not NUMERIC_CONTEXT.search(value) or sum(character.isdigit() for character in value) < 5:
        return value
    return value.replace("O", "0").replace("o", "0")


def luhn_valid(value: str) -> bool:
    digits = [int(character) for character in re.sub(r"\D", "", value)]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    parity = len(digits) % 2
    for index, digit in enumerate(digits):
        if index % 2 == parity:
            digit *= 2
            if digit > 9:
                digit -= 9
        total += digit
    return total % 10 == 0


def _overlaps(text: OCRTextResult, region: PrivacyObjectResult) -> bool:
    a, b = text.bounding_box, region.bounding_box
    intersection = max(0, min(a.x2, b.x2) - max(a.x1, b.x1)) * max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    text_area = max(1, (a.x2 - a.x1) * (a.y2 - a.y1))
    return intersection / text_area >= 0.5


def _mask(value: str, kind: str) -> str:
    compact = value.strip()
    if kind == "email":
        local, domain = compact.split("@", 1)
        return f"{local[:1]}***@{domain}"
    if kind == "pan_like_number":
        return f"{compact[:5]}****{compact[-1:]}"
    if kind == "url":
        return compact
    digits = re.sub(r"\D", "", compact)
    if kind == "aadhaar_like_number":
        return f"XXXX XXXX {digits[-4:]}"
    if kind == "payment_card_number":
        return f"**** **** **** {digits[-4:]}"
    if kind in {"phone_number", "pincode"}:
        return f"******{digits[-4:]}"
    if kind == "possible_expiry_date":
        return "**/**"
    if kind == "license_plate_text":
        return f"***{compact[-3:]}" if len(compact) > 3 else "***"
    return "Sensitive location text"


class SensitiveTextClassifier:
    def classify(self, texts: list[OCRTextResult], plates: list[PrivacyObjectResult],
                 cards: list[PrivacyObjectResult]) -> list[SensitiveTextResult]:
        results = []
        address_context = any(ADDRESS_WORDS.search(text.normalized_text) for text in texts)
        for text in texts:
            value = text.normalized_text
            numeric_value = _contextual_numeric_value(value)
            display_value = value
            kind = reason = None
            pattern_confidence = 0.0
            luhn = None
            in_card = any(_overlaps(text, card) for card in cards)
            in_plate = any(_overlaps(text, plate) for plate in plates)
            if in_plate and re.search(r"[A-Z0-9]", value, re.I):
                kind, reason, pattern_confidence = "license_plate_text", "OCR text overlaps a confirmed license plate region.", 0.98
            elif (match := EMAIL.search(value)):
                kind, reason, pattern_confidence = "email", "Matches an email address pattern.", 0.98
                display_value = match.group()
            elif (match := URL.search(value)):
                kind, reason, pattern_confidence = "url", "Matches a web address pattern.", 0.95
                display_value = match.group()
            elif (match := PAN.search(value)):
                kind, reason, pattern_confidence = "pan_like_number", "Matches the Indian PAN-like alphanumeric structure.", 0.94
                display_value = match.group()
            elif (match := PHONE.search(numeric_value)):
                kind, reason, pattern_confidence = "phone_number", "Matches a likely Indian phone-number format.", 0.94
                display_value = match.group()
            elif (card_match := CARD.search(numeric_value)):
                luhn = luhn_valid(card_match.group())
                if luhn or in_card:
                    kind, reason = "payment_card_number", "Matches a payment-card-length sequence with Luhn or confirmed card-region support."
                    pattern_confidence = 0.98 if luhn else 0.82
                    display_value = card_match.group()
            elif (match := AADHAAR.search(numeric_value)):
                kind, reason, pattern_confidence = "aadhaar_like_number", "Matches a 12-digit Aadhaar-like structure; identity is not verified.", 0.88
                display_value = match.group()
            elif (match := EXPIRY.search(value)) and in_card:
                kind, reason, pattern_confidence = "possible_expiry_date", "Matches an expiry-date pattern inside a confirmed card region.", 0.88
                display_value = match.group()
            elif (match := PINCODE.search(numeric_value)) and address_context:
                kind, reason, pattern_confidence = "pincode", "Matches an Indian PIN code near address-like text.", 0.76
                display_value = match.group()
            elif ADDRESS_WORDS.search(value):
                kind, reason, pattern_confidence = "possible_address", "Contains address-specific terms; classified conservatively.", 0.72
            if kind is None:
                continue
            confidence = round(min(1.0, 0.65 * text.confidence + 0.35 * pattern_confidence), 6)
            results.append(SensitiveTextResult(
                id=len(results) + 1, text_id=text.text_id, type=kind,
                masked_value=_mask(display_value, kind), confidence=confidence, reason=reason,
                bounding_box=text.bounding_box, luhn_valid=luhn,
            ))
        return results
