"""Bounded, expiring analysis metadata cache; image pixels are never retained."""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass

from app.config import settings
from app.schemas import AnalysisResponse


@dataclass(frozen=True)
class AnalysisSession:
    analysis: AnalysisResponse
    content_sha256: str
    expires_at: float


class AnalysisSessionCache:
    def __init__(self, max_entries: int, ttl_seconds: int) -> None:
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds
        self._entries: OrderedDict[str, AnalysisSession] = OrderedDict()
        self._lock = threading.RLock()

    def _purge(self) -> None:
        now = time.time()
        for key in [key for key, value in self._entries.items() if value.expires_at <= now]:
            self._entries.pop(key, None)

    def put(self, analysis: AnalysisResponse, content_sha256: str) -> str:
        with self._lock:
            self._purge()
            analysis_id = uuid.uuid4().hex
            self._entries[analysis_id] = AnalysisSession(
                analysis.model_copy(deep=True), content_sha256, time.time() + self.ttl_seconds,
            )
            while len(self._entries) > self.max_entries:
                self._entries.popitem(last=False)
            return analysis_id

    def get(self, analysis_id: str) -> AnalysisSession | None:
        with self._lock:
            self._purge()
            value = self._entries.get(analysis_id)
            if value is not None:
                self._entries.move_to_end(analysis_id)
            return value


analysis_session_cache = AnalysisSessionCache(
    settings.analysis_session_max_entries, settings.analysis_session_ttl_seconds,
)
