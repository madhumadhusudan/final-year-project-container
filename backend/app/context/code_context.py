"""Spatial association between codes and already-confirmed sensitive regions."""

from __future__ import annotations

from collections.abc import Iterable

from app.utils.geometry import center_inside, overlap_ratio


def associate_code(
    code_box: object, documents: Iterable[object], cards: Iterable[object],
) -> tuple[str | None, int | None]:
    """Prefer document context, then card context, only with strong spatial evidence."""
    document_matches = [
        (overlap_ratio(code_box, item), item) for item in documents
        if center_inside(code_box, item) and overlap_ratio(code_box, item) >= 0.6
    ]
    if document_matches:
        _, document = max(document_matches, key=lambda match: match[0])
        return "identity_document", getattr(document, "document_id")

    card_matches = [
        (overlap_ratio(code_box, item), item) for item in cards
        if center_inside(code_box, item) and overlap_ratio(code_box, item) >= 0.6
    ]
    if card_matches:
        _, card = max(card_matches, key=lambda match: match[0])
        return "payment_card", getattr(card, "card_id", None) or getattr(card, "id")
    return None, None
