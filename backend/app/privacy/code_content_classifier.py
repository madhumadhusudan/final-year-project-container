"""Privacy-safe classification of transient, locally decoded QR/barcode payloads."""

from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import urlsplit


@dataclass(frozen=True)
class SafeCodeContent:
    content_type: str
    masked_preview: str


class CodeContentClassifier:
    """Never returns the original payload and never performs payload-driven I/O."""

    _payment_prefixes = (
        "upi://pay", "tez://upi/pay", "phonepe://pay", "paytmmp://pay", "bharatpe://pay",
    )
    _identifier = re.compile(r"^[A-Za-z0-9._:/+\-=]{6,128}$")

    @staticmethod
    def _identifier_preview(value: str) -> str:
        compact = "".join(character for character in value if character.isalnum())
        if len(compact) < 4:
            return "Identifier encoded"
        return f"{'*' * min(8, max(4, len(compact) - 4))}{compact[-4:]}"

    @staticmethod
    def _url_preview(value: str) -> str:
        try:
            hostname = (urlsplit(value).hostname or "").strip(".").casefold()
        except ValueError:
            hostname = ""
        if hostname and len(hostname) <= 80 and all(
            character.isalnum() or character in ".-" for character in hostname
        ):
            return f"URL encoded ({hostname})"
        return "URL encoded"

    def classify(
        self, payload: str | None, *, code_kind: str, barcode_format: str | None = None,
    ) -> SafeCodeContent:
        if not payload:
            label = "QR code content could not be decoded" if code_kind == "qr" else "Barcode content could not be decoded"
            return SafeCodeContent("unknown", label)

        value = payload.strip()
        lowered = value.casefold()
        if lowered.startswith(self._payment_prefixes):
            return SafeCodeContent("payment", "Payment information encoded")
        if lowered.startswith("begin:vcard") or lowered.startswith("mecard:"):
            return SafeCodeContent("contact", "Contact information encoded")
        if lowered.startswith("wifi:"):
            return SafeCodeContent("wifi", "Wi-Fi credentials encoded")
        if lowered.startswith(("http://", "https://")):
            return SafeCodeContent("url", self._url_preview(value))

        if code_kind == "barcode":
            return SafeCodeContent("identifier", self._identifier_preview(value))
        if self._identifier.fullmatch(value):
            return SafeCodeContent("identifier", self._identifier_preview(value))
        if value.isprintable() and any(character.isspace() for character in value):
            return SafeCodeContent("text", "Text content encoded")
        return SafeCodeContent("unknown", "Decoded content hidden")

    @staticmethod
    def privacy_level(
        *, code_kind: str, content_type: str, decoded: bool,
        barcode_format: str | None = None, parent_type: str | None = None,
    ) -> str:
        if parent_type == "identity_document":
            return "critical"
        if parent_type == "payment_card":
            return "high"
        if code_kind == "qr":
            if content_type in {"payment", "contact", "wifi"}:
                return "high"
            return "moderate"
        if decoded and barcode_format in {"EAN-8", "EAN-13", "UPC-A", "UPC-E"}:
            return "low"
        if not decoded:
            return "moderate"
        return "moderate"
