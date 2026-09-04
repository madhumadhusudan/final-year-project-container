"""Low-latency, detector-selective analysis for transient camera frames."""

from __future__ import annotations

import time
from collections.abc import Iterable

from app.schemas import (
    BarcodeResult, BoundingBox, CardResult, FaceResult, ImageDetails, LiveFramePerformance,
    LiveFrameResponse, LiveModuleResult, LiveRegion, LiveRiskSummary, MainSubjectDetails,
    OCRTextResult, Point, PrivacyObjectResult, QRCodeResult,
)
from app.utils.image_validation import DecodedImage


LIVE_MODULES = frozenset({"faces", "plates", "cards", "documents", "codes", "ocr"})


class LiveDetectionService:
    """Adapter around the application's already initialized local detectors.

    The normal image pipeline deliberately stays unchanged. Live requests choose
    detector groups, allowing the browser to schedule fast face checks separately
    from expensive OCR and optional models. Frames never leave memory here.
    """

    def __init__(self, service) -> None:
        self._service = service

    def capabilities(self) -> dict[str, bool]:
        face = self._service._face_detector
        code = self._service._code_detector
        return {
            "background_faces": not face.name.endswith("unavailable"),
            "license_plates": self._service._plate_detector is not None,
            "payment_cards": self._service._card_detector is not None,
            "identity_documents": self._service._document_detector is not None,
            "qr_codes": bool(code and code.qr_available),
            "barcodes": bool(code and code.barcode_available),
            "sensitive_text": not self._service._ocr_service.name.endswith("unavailable"),
        }

    @staticmethod
    def _faces(raw_faces, decoded: DecodedImage) -> list[FaceResult]:
        image_area = decoded.width * decoded.height
        diagonal = (decoded.width ** 2 + decoded.height ** 2) ** 0.5
        faces: list[FaceResult] = []
        for raw in raw_faces:
            x1, y1 = max(0, min(raw.x1, decoded.width)), max(0, min(raw.y1, decoded.height))
            x2, y2 = max(x1, min(raw.x2, decoded.width)), max(y1, min(raw.y2, decoded.height))
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            distance = (((center_x - decoded.width / 2) ** 2 + (center_y - decoded.height / 2) ** 2) ** 0.5) / diagonal
            faces.append(FaceResult(
                face_id=len(faces) + 1,
                confidence=round(raw.confidence, 6),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                width=width, height=height, area=width * height,
                area_ratio=round((width * height) / image_area, 8),
                center=Point(x=center_x, y=center_y),
                normalized_center=Point(x=center_x / decoded.width, y=center_y / decoded.height),
                distance_from_image_center=round(distance, 8),
            ))
        return faces

    @staticmethod
    def _module(status: str, started: float, count: int = 0, message: str | None = None) -> LiveModuleResult:
        return LiveModuleResult(
            status=status, duration_ms=max(0, round((time.perf_counter() - started) * 1000)),
            detection_count=count, message=message,
        )

    @staticmethod
    def _regions(category: str, items: Iterable, id_attribute: str | None = None) -> list[LiveRegion]:
        regions = []
        for index, item in enumerate(items, start=1):
            item_id = getattr(item, id_attribute, index) if id_attribute else index
            regions.append(LiveRegion(
                detection_id=f"{category}_{item_id}", category=category,
                bounding_box=item.bounding_box, confidence=getattr(item, "confidence", None),
            ))
        return regions

    def analyze(
        self, decoded: DecodedImage, frame_id: int, captured_at_ms: int,
        requested_modules: set[str], preserve_main_subject: bool,
    ) -> LiveFrameResponse:
        started = time.perf_counter()
        received_at_ms = int(time.time() * 1000)
        modules = {
            name: LiveModuleResult(status="skipped") for name in sorted(LIVE_MODULES)
        }
        regions: list[LiveRegion] = []
        faces: list[FaceResult] = []
        plates: list[PrivacyObjectResult] = []
        cards: list[CardResult] = []
        documents = []
        qr_codes: list[QRCodeResult] = []
        barcodes: list[BarcodeResult] = []
        sensitive_items = []
        main_subject = MainSubjectDetails(status="not_found", reason="Face detection was not requested for this frame.")

        if "faces" in requested_modules:
            module_started = time.perf_counter()
            try:
                raw_faces = self._service._face_detector.detect(decoded.pixels_bgr)
                faces = self._faces(raw_faces, decoded)
                main_subject = self._service._subject_analyzer.analyze(faces, [])
                modules["faces"] = self._module("completed", module_started, len(faces))
                for face in faces:
                    regions.append(LiveRegion(
                        detection_id=f"face_{face.face_id}", category="face",
                        bounding_box=face.bounding_box, confidence=face.confidence,
                        role=face.role if preserve_main_subject else "background_face",
                    ))
            except RuntimeError:
                status = "unavailable" if self._service._face_detector.name.endswith("unavailable") else "error"
                modules["faces"] = self._module(status, module_started, message="Local face detection is unavailable.")
                main_subject = MainSubjectDetails(status="not_found", reason="Local face detection is unavailable.")

        if "plates" in requested_modules:
            module_started = time.perf_counter()
            detector = self._service._plate_detector
            if detector is None:
                modules["plates"] = self._module("unavailable", module_started, message="Dedicated plate model is unavailable.")
            else:
                try:
                    plates = self._service._privacy_results(detector.detect(decoded.pixels_bgr), decoded.width, decoded.height)
                    modules["plates"] = self._module("completed", module_started, len(plates))
                    regions.extend(self._regions("license_plate", plates))
                except RuntimeError:
                    modules["plates"] = self._module("error", module_started, message="Local plate detection failed.")

        if "cards" in requested_modules:
            module_started = time.perf_counter()
            detector = self._service._card_detector
            if detector is None:
                modules["cards"] = self._module("unavailable", module_started, message="Dedicated payment-card model is unavailable.")
            else:
                try:
                    cards = self._service._privacy_results(
                        detector.detect(decoded.pixels_bgr), decoded.width, decoded.height, cards=True,
                    )
                    modules["cards"] = self._module("completed", module_started, len(cards))
                    regions.extend(self._regions("payment_card", cards, "card_id"))
                except RuntimeError:
                    modules["cards"] = self._module("error", module_started, message="Local payment-card detection failed.")

        if "documents" in requested_modules:
            module_started = time.perf_counter()
            detector = self._service._document_detector
            if detector is None:
                modules["documents"] = self._module("unavailable", module_started, message="Dedicated document model is unavailable.")
            else:
                try:
                    raw = self._service._privacy_results(detector.detect(decoded.pixels_bgr), decoded.width, decoded.height)
                    documents = [
                        self._service._document_classifier.classify(
                            document_id=index, model_class=item.class_name, model_confidence=item.confidence,
                            bounding_box=item.bounding_box, image_width=decoded.width,
                            image_height=decoded.height, texts=[], sensitive_items=[],
                        ) for index, item in enumerate(raw, start=1)
                    ]
                    modules["documents"] = self._module("completed", module_started, len(documents))
                    regions.extend(self._regions("identity_document", documents, "document_id"))
                except RuntimeError:
                    modules["documents"] = self._module("error", module_started, message="Local document detection failed.")

        if "codes" in requested_modules:
            module_started = time.perf_counter()
            detector = self._service._code_detector
            code_error = False
            if detector is None:
                modules["codes"] = self._module("unavailable", module_started, message="Local code detectors are unavailable.")
            else:
                try:
                    raw_qr = detector.detect_qr(decoded.pixels_bgr) if detector.qr_available else []
                    raw_barcodes = detector.detect_barcodes(decoded.pixels_bgr) if detector.barcode_available else []
                    qr_codes = self._service._code_results(raw_qr, decoded.width, decoded.height, documents, cards, kind="qr")
                    barcodes = self._service._code_results(
                        raw_barcodes, decoded.width, decoded.height, documents, cards, kind="barcode",
                    )
                except RuntimeError:
                    code_error = True
                if code_error:
                    modules["codes"] = self._module("error", module_started, message="Local QR/barcode detection failed.")
                elif not detector.qr_available and not detector.barcode_available:
                    modules["codes"] = self._module("unavailable", module_started, message="Local code detectors are unavailable.")
                else:
                    count = len(qr_codes) + len(barcodes)
                    modules["codes"] = self._module("completed", module_started, count)
                    regions.extend(self._regions("qr_code", qr_codes, "qr_id"))
                    regions.extend(self._regions("barcode", barcodes, "barcode_id"))

        if "ocr" in requested_modules:
            module_started = time.perf_counter()
            try:
                raw_texts = self._service._ocr_service.extract_text(decoded.pixels_bgr)
                texts = []
                for raw in raw_texts:
                    x1, y1 = max(0, min(raw.x1, decoded.width)), max(0, min(raw.y1, decoded.height))
                    x2, y2 = max(x1, min(raw.x2, decoded.width)), max(y1, min(raw.y2, decoded.height))
                    if x2 > x1 and y2 > y1 and 0 <= raw.confidence <= 1:
                        texts.append(OCRTextResult(
                            text_id=len(texts) + 1, raw_text=raw.raw_text,
                            normalized_text=raw.normalized_text, confidence=round(raw.confidence, 6),
                            bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                        ))
                sensitive_items = self._service._sensitive_text_classifier.classify(texts, plates, cards)
                modules["ocr"] = self._module("completed", module_started, len(sensitive_items))
                regions.extend(self._regions("sensitive_text", sensitive_items))
            except RuntimeError:
                unavailable = self._service._ocr_service.name.endswith("unavailable")
                modules["ocr"] = self._module(
                    "unavailable" if unavailable else "error", module_started,
                    message="Local OCR is unavailable." if unavailable else "Local OCR failed.",
                )

        detector_statuses = {
            name: result.status for name, result in modules.items()
            if name in requested_modules
        }
        risk = self._service._risk_engine.calculate(
            ImageDetails(filename="live-frame", width=decoded.width, height=decoded.height, format=decoded.format),
            faces, main_subject, plates, cards, sensitive_items, detector_statuses,
            documents=documents, qr_codes=qr_codes, barcodes=barcodes,
        )
        completed_at_ms = int(time.time() * 1000)
        return LiveFrameResponse(
            frame_id=frame_id, captured_at_ms=captured_at_ms,
            image=ImageDetails(filename="live-frame", width=decoded.width, height=decoded.height, format=decoded.format),
            regions=regions, main_subject=main_subject,
            modules=modules, risk=LiveRiskSummary(score=risk.score, level=risk.level),
            performance=LiveFramePerformance(
                total_analysis_ms=max(0, round((time.perf_counter() - started) * 1000)),
                server_received_at_ms=received_at_ms, server_completed_at_ms=completed_at_ms,
            ),
        )
