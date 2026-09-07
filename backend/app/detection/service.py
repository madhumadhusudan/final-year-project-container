"""One-time model lifecycle, inference, and response assembly."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

from app.config import Settings, settings
from app.schemas import (
    AnalysisDetails, AnalysisOptions, AnalysisResponse, BarcodeDetectionDetails, BarcodeResult, BoundingBox,
    CardDetectionDetails, CardResult, DetectionResult, DetectorDiagnostics, DocumentDetectionDetails,
    FaceDetectionDetails, FaceResult, ImageDetails, LicensePlateDetectionDetails,
    ObjectDetectionDetails, OCRDetails, OCRTextResult, PerformanceDetails, Point,
    PrivacyObjectResult, PrivacySensitiveElements, QRCodeResult, QRDetectionDetails, SensitiveTextDetails,
)
from app.context.code_context import associate_code
from app.context.main_subject_analyzer import MainSubjectAnalyzer
from app.ocr.ocr_service import OCRService, UnavailableOCRService
from app.privacy.sensitive_text_classifier import SensitiveTextClassifier
from app.privacy.document_classifier import DocumentClassifier
from app.privacy.code_content_classifier import CodeContentClassifier
from app.privacy.risk_score import PrivacyRiskEngine
from app.utils.image_validation import DecodedImage
from app.utils.analysis_coordinates import AnalysisImageCache
from app.analysis.image_optimizer import (
    FACE_MAX_SIDE, PRIVACY_MAX_SIDE, YOLO_IMAGE_SIZE, extract_text_optimized, scale_values, scaled_detect,
)

from .yolo_detector import YoloDetector
from .face_detector import FaceDetector, UnavailableFaceDetector
from .card_detector import CardDetector
from .license_plate_detector import LicensePlateDetector
from .document_detector import DocumentDetector
from .code_detector import CodeDetector
from .registry import DetectorRegistry


_DETECTOR_EXECUTOR = ThreadPoolExecutor(max_workers=max(2, settings.image_detector_workers), thread_name_prefix="image-detector")


class DetectionService:
    def __init__(self, detector: YoloDetector, face_detector: FaceDetector, app_settings: Settings = settings,
                 plate_detector: LicensePlateDetector | None = None, card_detector: CardDetector | None = None,
                 ocr_service: OCRService | UnavailableOCRService | None = None,
                 document_detector: DocumentDetector | None = None,
                 code_detector: CodeDetector | None = None, device: str = "cpu") -> None:
        self._detector = detector
        self._face_detector = face_detector
        self._settings = app_settings
        self._plate_detector = plate_detector
        self._card_detector = card_detector
        self._document_detector = document_detector
        self._code_detector = code_detector
        self._subject_analyzer = MainSubjectAnalyzer(app_settings)
        self._ocr_service = ocr_service or UnavailableOCRService()
        self._sensitive_text_classifier = SensitiveTextClassifier()
        self._document_classifier = DocumentClassifier()
        self._code_classifier = CodeContentClassifier()
        self._risk_engine = PrivacyRiskEngine()
        self._device = device

    @staticmethod
    def _privacy_results(
        raw_results, image_width: int, image_height: int, *, cards: bool = False
    ) -> list[PrivacyObjectResult]:
        results = []
        for raw in raw_results:
            x1, y1 = max(0, min(raw.x1, image_width)), max(0, min(raw.y1, image_height))
            x2, y2 = max(x1, min(raw.x2, image_width)), max(y1, min(raw.y2, image_height))
            if x2 <= x1 or y2 <= y1:
                continue
            result_id = len(results) + 1
            result_type = CardResult if cards else PrivacyObjectResult
            extra = {"card_id": result_id} if cards else {}
            results.append(result_type(
                id=result_id, class_name=raw.class_name, confidence=round(raw.confidence, 6),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2), **extra,
            ))
        return results

    @staticmethod
    def _match_plates_to_vehicles(plates: list[PrivacyObjectResult], detections: list[DetectionResult]) -> None:
        vehicles = [item for item in detections if item.class_name in {"car", "truck", "bus", "motorcycle"}]
        for plate in plates:
            center_x = (plate.bounding_box.x1 + plate.bounding_box.x2) / 2
            center_y = (plate.bounding_box.y1 + plate.bounding_box.y2) / 2
            containing = [vehicle for vehicle in vehicles if
                          vehicle.bounding_box.x1 <= center_x <= vehicle.bounding_box.x2 and
                          vehicle.bounding_box.y1 <= center_y <= vehicle.bounding_box.y2]
            if containing:
                plate.matched_vehicle_id = min(
                    containing,
                    key=lambda vehicle: (vehicle.bounding_box.x2 - vehicle.bounding_box.x1)
                    * (vehicle.bounding_box.y2 - vehicle.bounding_box.y1),
                ).id

    @staticmethod
    def _exclude_document_faces(
        faces: list[FaceResult], documents: list[PrivacyObjectResult],
    ) -> list[FaceResult]:
        """Mark portrait photos substantially contained by detected documents."""
        external_faces = []
        for face in faces:
            face_box = face.bounding_box
            contained = False
            for document in documents:
                doc_box = document.bounding_box
                intersection = max(0, min(face_box.x2, doc_box.x2) - max(face_box.x1, doc_box.x1)) * max(
                    0, min(face_box.y2, doc_box.y2) - max(face_box.y1, doc_box.y1)
                )
                if intersection / max(1, face.area) >= 0.8:
                    contained = True
                    break
            if contained:
                face.role = "document_face"
            else:
                external_faces.append(face)
        return external_faces

    def _code_results(
        self, raw_items, image_width: int, image_height: int, documents, cards, *, kind: str,
    ) -> list[QRCodeResult] | list[BarcodeResult]:
        results = []
        for raw in raw_items:
            box = BoundingBox(x1=raw.x1, y1=raw.y1, x2=raw.x2, y2=raw.y2)
            parent_type, parent_id = associate_code(box, documents, cards)
            safe_content = self._code_classifier.classify(
                raw.payload, code_kind=kind, barcode_format=raw.format,
            )
            common = {
                "confidence": None,
                "bounding_box": box,
                "polygon": [Point(x=point[0], y=point[1]) for point in raw.polygon],
                "decoded": raw.payload is not None,
                "content_type": safe_content.content_type,
                "masked_preview": safe_content.masked_preview,
                "privacy_level": self._code_classifier.privacy_level(
                    code_kind=kind, content_type=safe_content.content_type,
                    decoded=raw.payload is not None, barcode_format=raw.format,
                    parent_type=parent_type,
                ),
                "parent_type": parent_type,
                "parent_id": parent_id,
            }
            if kind == "qr":
                results.append(QRCodeResult(qr_id=len(results) + 1, **common))
            else:
                results.append(BarcodeResult(
                    barcode_id=len(results) + 1, format=raw.format, **common,
                ))
        return results

    def analyze(self, decoded: DecodedImage, filename: str, options: AnalysisOptions | None = None) -> AnalysisResponse:
        total_started = time.perf_counter()
        options = options or AnalysisOptions()
        image_cache = AnalysisImageCache(decoded.pixels_bgr)
        yolo_size = YOLO_IMAGE_SIZE[options.performance_profile]
        face_size = FACE_MAX_SIDE[options.performance_profile]
        privacy_size = PRIVACY_MAX_SIDE[options.performance_profile]
        yolo_args = (yolo_size,) if hasattr(self._detector, "inference_image_size") else ()
        qr_args = (
            (True,)
            if getattr(self._code_detector, "supports_exhaustive_qr", False)
            else ()
        )
        # YuNet and YOLO have independent, locked model instances. Starting both
        # together reduces wall time without concurrent access to either model.
        object_future = face_future = qr_future = None
        parallel = bool(self._settings.image_parallel_detectors and options.detect_objects and options.detect_faces)
        if parallel:
            image_cache.get(yolo_size)
            image_cache.get(face_size)
            if options.detect_qr and self._code_detector is not None and self._code_detector.qr_available:
                image_cache.get(privacy_size)
                qr_future = _DETECTOR_EXECUTOR.submit(
                    scaled_detect, self._code_detector, image_cache, privacy_size, "detect_qr",
                    qr_args,
                )
            object_future = _DETECTOR_EXECUTOR.submit(scaled_detect, self._detector, image_cache, yolo_size, "detect", yolo_args)
            face_future = _DETECTOR_EXECUTOR.submit(scaled_detect, self._face_detector, image_cache, face_size)
        if options.detect_objects:
            predictions, object_ms = object_future.result() if object_future else scaled_detect(self._detector, image_cache, yolo_size, method_args=yolo_args)
        else:
            predictions, object_ms = [], 0
        valid_predictions = []
        for prediction in predictions:
            clipped = prediction.clipped(decoded.width, decoded.height)
            if clipped.width > 0 and clipped.height > 0:
                valid_predictions.append(clipped)

        detections = [
            DetectionResult(
                id=index,
                class_id=detection.class_id,
                class_name=detection.class_name,
                confidence=round(detection.confidence, 6),
                bounding_box=BoundingBox(x1=detection.x1, y1=detection.y1, x2=detection.x2, y2=detection.y2),
            )
            for index, detection in enumerate(valid_predictions, start=1)
        ]
        face_error = None
        face_status = "skipped" if not options.detect_faces else "completed"
        if options.detect_faces:
            try:
                raw_faces, face_ms = face_future.result() if face_future else scaled_detect(self._face_detector, image_cache, face_size)
            except RuntimeError:
                raw_faces, face_ms, face_status = [], 0, "error"
                face_error = "Face detection was unavailable for this image."
        else:
            raw_faces, face_ms = [], 0
        image_area = decoded.width * decoded.height
        image_diagonal = (decoded.width ** 2 + decoded.height ** 2) ** 0.5
        faces = []
        for face in raw_faces:
            x1 = max(0, min(face.x1, decoded.width))
            y1 = max(0, min(face.y1, decoded.height))
            x2 = max(x1, min(face.x2, decoded.width))
            y2 = max(y1, min(face.y2, decoded.height))
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue
            center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
            distance = (((center_x - decoded.width / 2) ** 2 + (center_y - decoded.height / 2) ** 2) ** 0.5) / image_diagonal
            faces.append(FaceResult(
                face_id=len(faces) + 1,
                confidence=round(face.confidence, 6),
                bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
                width=width, height=height, area=width * height,
                area_ratio=round((width * height) / image_area, 8),
                center=Point(x=center_x, y=center_y),
                normalized_center=Point(x=center_x / decoded.width, y=center_y / decoded.height),
                distance_from_image_center=round(distance, 8),
            ))
        plate_started = time.perf_counter()
        plate_error = None
        if not options.detect_plates:
            raw_plates, plate_status = [], "skipped"
        elif self._plate_detector is None:
            raw_plates = []
            plate_status = "unavailable"
        else:
            try:
                raw_plates, _ = scaled_detect(self._plate_detector, image_cache, privacy_size)
                plate_status = "completed"
            except RuntimeError:
                raw_plates, plate_status, plate_error = [], "error", "License plate detection failed locally."
        plate_ms = max(0, round((time.perf_counter() - plate_started) * 1000))
        plates = self._privacy_results(raw_plates, decoded.width, decoded.height)
        self._match_plates_to_vehicles(plates, detections)

        card_started = time.perf_counter()
        card_error = None
        card_run = None
        if not options.detect_cards:
            raw_cards, card_status = [], "skipped"
        elif self._card_detector is None:
            raw_cards = []
            card_status = "unavailable"
        else:
            try:
                if hasattr(self._card_detector, "detect_with_diagnostics"):
                    view = image_cache.get(privacy_size)
                    card_run = self._card_detector.detect_with_diagnostics(view.pixels_bgr)
                    raw_cards = scale_values(card_run.detections, view.scale_x, view.scale_y)
                else:
                    raw_cards, _ = scaled_detect(self._card_detector, image_cache, privacy_size)
                card_status = "completed"
            except RuntimeError:
                raw_cards, card_status, card_error = [], "error", "Payment card detection failed locally."
        card_ms = max(0, round((time.perf_counter() - card_started) * 1000))
        cards = self._privacy_results(raw_cards, decoded.width, decoded.height, cards=True)
        card_diagnostics = None
        if self._settings.app_environment == "development" and card_status == "completed":
            card_diagnostics = DetectorDiagnostics(
                loaded=self._card_detector is not None,
                model_name=getattr(self._card_detector, "model_name", self._settings.card_model_path.name),
                model_class_names=getattr(self._card_detector, "model_class_names", []),
                inference_image_size=getattr(
                    self._card_detector, "inference_image_size", self._settings.card_inference_image_size
                ),
                confidence_threshold=getattr(
                    self._card_detector, "confidence_threshold", self._settings.card_confidence_threshold
                ),
                raw_detection_count=getattr(card_run, "raw_detection_count", len(raw_cards)),
                accepted_detection_count=len(cards),
                inference_time_ms=getattr(card_run, "inference_time_ms", card_ms),
                tile_size=getattr(self._card_detector, "tile_size", self._settings.card_tile_size),
                tile_inference_image_size=getattr(
                    self._card_detector, "tile_inference_image_size",
                    self._settings.card_tile_inference_image_size,
                ),
                tiles_processed=getattr(card_run, "tiles_processed", 1),
            )

        document_started = time.perf_counter()
        document_error = None
        if not options.detect_documents:
            raw_documents, document_status = [], "skipped"
        elif self._document_detector is None:
            raw_documents = []
            document_status = "unavailable"
        else:
            try:
                raw_documents, _ = scaled_detect(self._document_detector, image_cache, privacy_size)
                document_status = "completed"
            except RuntimeError:
                raw_documents, document_status = [], "error"
                document_error = "Identity document detection failed locally."
        document_ms = max(0, round((time.perf_counter() - document_started) * 1000))
        document_regions = self._privacy_results(raw_documents, decoded.width, decoded.height)

        qr_started = time.perf_counter()
        qr_error = None
        qr_inference_ms = None
        if not options.detect_qr:
            raw_qr_codes, qr_status = [], "skipped"
        elif self._code_detector is None or not self._code_detector.qr_available:
            raw_qr_codes, qr_status = [], "unavailable"
        else:
            try:
                raw_qr_codes, qr_inference_ms = qr_future.result() if qr_future else scaled_detect(
                    self._code_detector, image_cache, privacy_size, "detect_qr",
                    qr_args,
                )
                qr_status = "completed"
            except RuntimeError:
                raw_qr_codes, qr_status = [], "error"
                qr_error = "Local QR detection failed for this image."
        qr_ms = qr_inference_ms if qr_inference_ms is not None else max(
            0, round((time.perf_counter() - qr_started) * 1000),
        )

        barcode_started = time.perf_counter()
        barcode_error = None
        if not options.detect_barcodes:
            raw_barcodes, barcode_status = [], "skipped"
        elif self._code_detector is None or not self._code_detector.barcode_available:
            raw_barcodes, barcode_status = [], "unavailable"
        else:
            try:
                raw_barcodes, _ = scaled_detect(self._code_detector, image_cache, privacy_size, "detect_barcodes")
                barcode_status = "completed"
            except RuntimeError:
                raw_barcodes, barcode_status = [], "error"
                barcode_error = "Local barcode detection failed for this image."
        barcode_ms = max(0, round((time.perf_counter() - barcode_started) * 1000))

        context_started = time.perf_counter()
        context_faces = self._exclude_document_faces(faces, document_regions)
        main_subject = self._subject_analyzer.analyze(context_faces, detections)
        context_ms = max(0, round((time.perf_counter() - context_started) * 1000))

        if not options.detect_sensitive_text:
            raw_texts, ocr_ms, ocr_status, ocr_error = [], 0, "skipped", "Disabled for this analysis."
        else:
            try:
                raw_texts, ocr_ms = extract_text_optimized(
                    self._ocr_service, image_cache, options.performance_profile,
                    [item.bounding_box for item in [*plates, *cards, *document_regions]],
                )
                ocr_status, ocr_error = "completed", None
            except RuntimeError:
                raw_texts, ocr_ms = [], 0
                ocr_status = "unavailable" if self._ocr_service.name.endswith("unavailable") else "error"
                ocr_error = "The local OCR engine is unavailable." if ocr_status == "unavailable" else "Local OCR failed for this image."
        texts = []
        for raw in raw_texts:
            x1, y1 = max(0, min(raw.x1, decoded.width)), max(0, min(raw.y1, decoded.height))
            x2, y2 = max(x1, min(raw.x2, decoded.width)), max(y1, min(raw.y2, decoded.height))
            if x2 <= x1 or y2 <= y1 or not 0 <= raw.confidence <= 1:
                continue
            texts.append(OCRTextResult(
                text_id=len(texts) + 1, raw_text=raw.raw_text, normalized_text=raw.normalized_text,
                confidence=round(raw.confidence, 6), bounding_box=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
            ))
        sensitive_started = time.perf_counter()
        sensitive_items = self._sensitive_text_classifier.classify(texts, plates, cards) if ocr_status == "completed" else []
        sensitive_ms = max(0, round((time.perf_counter() - sensitive_started) * 1000))
        classification_started = time.perf_counter()
        documents = [
            self._document_classifier.classify(
                document_id=index, model_class=region.class_name, model_confidence=region.confidence,
                bounding_box=region.bounding_box, image_width=decoded.width, image_height=decoded.height,
                texts=texts, sensitive_items=sensitive_items,
            )
            for index, region in enumerate(document_regions, start=1)
        ]
        document_classification_ms = max(0, round((time.perf_counter() - classification_started) * 1000))
        code_classification_started = time.perf_counter()
        qr_codes = self._code_results(
            raw_qr_codes, decoded.width, decoded.height, documents, cards, kind="qr",
        )
        barcodes = self._code_results(
            raw_barcodes, decoded.width, decoded.height, documents, cards, kind="barcode",
        )
        code_classification_ms = max(0, round((time.perf_counter() - code_classification_started) * 1000))
        objects = ObjectDetectionDetails(model=self._settings.yolo_model_name, detection_count=len(detections), detections=detections)
        image_details = ImageDetails(
            filename=filename, width=decoded.width, height=decoded.height, format=decoded.format,
        )
        risk_started = time.perf_counter()
        privacy_risk = self._risk_engine.calculate(
            image_details, faces, main_subject, plates, cards, sensitive_items,
            {
                "face_detection": face_status,
                "license_plate_detection": plate_status,
                "card_detection": card_status,
                "document_detection": document_status,
                "qr_detection": qr_status,
                "barcode_detection": barcode_status,
                "ocr": ocr_status,
            },
            documents=documents,
            qr_codes=qr_codes,
            barcodes=barcodes,
        )
        risk_ms = max(0, round((time.perf_counter() - risk_started) * 1000))
        background_face_count = sum(
            face.role in {"background_face", "unclassified"} for face in faces
        )
        total_ms = max(0, round((time.perf_counter() - total_started) * 1000))
        return AnalysisResponse(
            image=image_details,
            analysis=AnalysisDetails(
                model=self._settings.yolo_model_name,
                detection_count=len(detections),
                detections=detections,
                object_detection=objects,
                face_detection=FaceDetectionDetails(
                    status=face_status, detector=self._face_detector.name,
                    face_count=len(faces), faces=faces, error=face_error,
                ),
                main_subject=main_subject,
                license_plate_detection=LicensePlateDetectionDetails(
                    status=plate_status, detector=getattr(self._plate_detector, "name", "model_required"),
                    plate_count=len(plates), plates=plates,
                    message=plate_error or ("Dedicated license plate model required." if plate_status == "unavailable" else None),
                    model_source=self._settings.license_plate_model_source or None,
                    supported_classes=sorted(LicensePlateDetector.supported_classes),
                ),
                card_detection=CardDetectionDetails(
                    status=card_status, detector=getattr(self._card_detector, "name", "model_required"),
                    card_count=len(cards), cards=cards,
                    message=card_error or ("Dedicated payment card model required." if card_status == "unavailable" else None),
                    model_source=self._settings.card_model_source or None,
                    supported_classes=sorted(CardDetector.supported_classes),
                    diagnostics=card_diagnostics,
                ),
                document_detection=DocumentDetectionDetails(
                    status=document_status,
                    detector=getattr(self._document_detector, "name", "model_required"),
                    document_count=len(documents), documents=documents,
                    message=document_error or (
                        "Dedicated document model required." if document_status == "unavailable" else None
                    ),
                    model_source=self._settings.document_model_source or None,
                    model_license=self._settings.document_model_license or None,
                    model_class_names=(getattr(self._document_detector, "model_class_names", []) if document_status == "completed" else []),
                    confidence_threshold=(getattr(
                        self._document_detector, "confidence_threshold",
                        self._settings.document_confidence_threshold,
                    ) if document_status == "completed" else self._settings.document_confidence_threshold),
                    inference_image_size=(getattr(
                        self._document_detector, "inference_image_size",
                        self._settings.document_inference_image_size,
                    ) if document_status == "completed" else self._settings.document_inference_image_size),
                ),
                qr_detection=QRDetectionDetails(
                    status=qr_status,
                    detector=getattr(self._code_detector, "qr_name", "opencv_qrcode_detector"),
                    qr_count=len(qr_codes),
                    items=qr_codes,
                    message=qr_error or (
                        "OpenCV QRCodeDetector is unavailable." if qr_status == "unavailable" else None
                    ),
                ),
                barcode_detection=BarcodeDetectionDetails(
                    status=barcode_status,
                    detector=getattr(self._code_detector, "barcode_name", "opencv_barcode_detector"),
                    barcode_count=len(barcodes),
                    items=barcodes,
                    supported_formats=list(CodeDetector.supported_barcode_formats),
                    message=barcode_error or (
                        "OpenCV BarcodeDetector is unavailable." if barcode_status == "unavailable" else None
                    ),
                ),
                ocr=OCRDetails(
                    status=ocr_status, engine=self._ocr_service.name, languages=list(self._settings.ocr_languages),
                    text_count=len(texts), texts=texts, message=ocr_error,
                ),
                sensitive_text=SensitiveTextDetails(
                    status=ocr_status, count=len(sensitive_items), items=sensitive_items, message=ocr_error,
                ),
                privacy_risk=privacy_risk,
                privacy_sensitive_elements=PrivacySensitiveElements(
                    background_faces=background_face_count, license_plates=len(plates),
                    payment_cards=len(cards), identity_documents=len(documents),
                    qr_codes=len(qr_codes), barcodes=len(barcodes),
                    sensitive_text=len(sensitive_items),
                ),
            ),
            performance=PerformanceDetails(
                inference_time_ms=object_ms, object_detection_ms=object_ms,
                face_detection_ms=face_ms, total_analysis_ms=total_ms,
                license_plate_detection_ms=plate_ms, card_detection_ms=card_ms,
                document_detection_ms=document_ms,
                document_classification_ms=document_classification_ms,
                qr_detection_ms=qr_ms, barcode_detection_ms=barcode_ms,
                code_classification_ms=code_classification_ms,
                context_analysis_ms=context_ms,
                ocr_detection_ms=ocr_ms, sensitive_text_analysis_ms=sensitive_ms,
                image_decode_ms=decoded.decode_ms, image_preprocess_ms=image_cache.preprocess_ms,
                yolo_ms=object_ms, plate_detection_ms=plate_ms, ocr_ms=ocr_ms,
                risk_score_ms=risk_ms, device=self._device,
                performance_profile=options.performance_profile, parallel_execution=parallel,
                enabled_modules=[name for name, enabled in {
                    "objects": options.detect_objects, "faces": options.detect_faces,
                    "plates": options.detect_plates, "cards": options.detect_cards,
                    "documents": options.detect_documents, "qr": options.detect_qr,
                    "barcodes": options.detect_barcodes, "sensitive_text": options.detect_sensitive_text,
                }.items() if enabled],
            ),
        )


@lru_cache(maxsize=1)
def get_detection_service() -> DetectionService:
    registry = DetectorRegistry(settings)
    return DetectionService(
        registry.yolo, registry.face, settings, registry.plate, registry.card, registry.ocr,
        registry.document, registry.code, registry.device,
    )
