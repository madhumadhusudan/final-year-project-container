"""Measured Day 16 acceptance runner using only local, safe synthetic media.

The JSON output contains counts, boxes, timings, and masked classifications only.
It deliberately never serializes OCR text or QR/barcode payloads.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from app.anonymization.anonymizer import ImageAnonymizer
from app.config import settings
from app.detection.card_detector import CardDetector
from app.detection.code_detector import CodeDetector
from app.detection.face_detector import FaceDetector
from app.detection.service import get_detection_service
from app.ocr.ocr_service import OCRService
from app.privacy.code_content_classifier import CodeContentClassifier
from app.privacy.risk_score import PrivacyRiskEngine
from app.privacy.sensitive_text_classifier import SensitiveTextClassifier
from app.routes.protection import _protect
from app.schemas import AnalysisOptions, BoundingBox, OCRTextResult, ProtectionSettings
from app.utils.image_validation import DecodedImage

from .fixture_factory import (
    difficult_privacy_scene, ean13, load_fixture, negative_shape_scene, normal_text_image,
    qr_image, rotate_bound, sensitive_text_image, synthetic_card, synthetic_document, synthetic_plate,
)
from .metrics import DetectionMetrics, combine, match_boxes


def metric_dict(value: DetectionMetrics) -> dict:
    def rounded(number):
        return round(number, 4) if number is not None else None
    return {
        "tp": value.true_positives, "fp": value.false_positives, "fn": value.false_negatives,
        "precision": rounded(value.precision), "recall": rounded(value.recall),
        "f1": rounded(value.f1), "average_iou": rounded(value.average_iou),
    }


def face_calibration() -> dict:
    fixtures = [
        (
            load_fixture("synthetic_three_people.png"),
            [(190, 195, 310, 375), (620, 155, 890, 530), (1235, 290, 1365, 470)],
        ),
        (
            load_fixture("synthetic_profile_occlusion_lowlight.png"),
            [(265, 210, 455, 535), (1040, 335, 1280, 655)],
        ),
    ]
    negative = negative_shape_scene()
    output = {}
    for threshold in (0.20, 0.25, 0.30, 0.40, 0.50, 0.75):
        detector = FaceDetector(settings.face_model_path, threshold)
        samples, latencies = [], []
        for pixels, truth in fixtures:
            started = time.perf_counter()
            predictions = detector.detect(pixels)
            latencies.append((time.perf_counter() - started) * 1000)
            samples.append(match_boxes(
                truth, [(item.x1, item.y1, item.x2, item.y2) for item in predictions], 0.45,
            ))
        started = time.perf_counter()
        negative_predictions = detector.detect(negative)
        latencies.append((time.perf_counter() - started) * 1000)
        aggregate = combine(samples + [DetectionMetrics(0, len(negative_predictions), 0)])
        output[f"{threshold:.2f}"] = {
            **metric_dict(aggregate), "average_latency_ms": round(statistics.mean(latencies), 2),
        }
    return {"chosen_threshold": settings.face_confidence_threshold, "thresholds": output}


def _place_card(card: np.ndarray, size: tuple[int, int], angle: float = 0, occlude: bool = False) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    resized = cv2.resize(card, size, interpolation=cv2.INTER_AREA)
    if angle:
        resized = rotate_bound(resized, angle)
    canvas = np.full((900, 1400, 3), (185, 170, 145), np.uint8)
    x1, y1 = 500, 350
    x2, y2 = x1 + resized.shape[1], y1 + resized.shape[0]
    canvas[y1:y2, x1:x2] = resized
    if occlude:
        cv2.rectangle(canvas, (x2 - resized.shape[1] // 4, y1), (x2, y2), (185, 170, 145), -1)
    return canvas, (x1, y1, x2, y2)


def card_validation() -> dict:
    if not settings.card_model_path.is_file():
        return {"status": "unavailable", "reason": "Configured local model file is absent."}
    detector = CardDetector(
        settings.card_model_path, settings.card_confidence_threshold,
        settings.card_inference_image_size, settings.card_tile_size,
        settings.card_tile_overlap, settings.card_tile_inference_image_size,
    )
    labeled = synthetic_card()
    x1, y1, x2, y2 = labeled.boxes[0]
    crop = labeled.pixels[y1:y2, x1:x2]
    scenes = {
        "close_up": (labeled.pixels, labeled.boxes[0]),
        "small": _place_card(crop, (230, 142)),
        "rotated": _place_card(crop, (360, 222), 22),
        "partial": _place_card(crop, (360, 222), 0, True),
        "in_hand_simulation": _place_card(crop, (330, 204), -8, True),
        "on_table": _place_card(crop, (360, 222), -5),
        "in_clutter": _place_card(crop, (300, 185), 12),
    }
    metrics, details, latencies = [], {}, []
    for name, (pixels, truth) in scenes.items():
        if name == "in_clutter":
            for offset in range(0, 1400, 120):
                cv2.rectangle(pixels, (offset, 50), (min(1399, offset + 65), 130), (80, 120, 160), -1)
        started = time.perf_counter()
        predictions = detector.detect(pixels)
        latency = (time.perf_counter() - started) * 1000
        latencies.append(latency)
        measured = match_boxes([truth], [(item.x1, item.y1, item.x2, item.y2) for item in predictions], 0.35)
        metrics.append(measured)
        details[name] = {
            "detected": len(predictions), "missed": measured.false_negatives,
            "false_positives": measured.false_positives,
            "confidences": [round(item.confidence, 4) for item in predictions],
            "latency_ms": round(latency, 2),
        }
    negative_started = time.perf_counter()
    negatives = detector.detect(negative_shape_scene())
    negative_latency = (time.perf_counter() - negative_started) * 1000
    metrics.append(DetectionMetrics(0, len(negatives), 0))
    return {
        "status": "completed", "chosen_threshold": settings.card_confidence_threshold,
        "metrics": metric_dict(combine(metrics)), "average_latency_ms": round(statistics.mean(latencies), 2),
        "cases": details, "negative_scene_false_positives": len(negatives),
        "negative_latency_ms": round(negative_latency, 2),
    }


def code_validation() -> dict:
    detector, classifier = CodeDetector(), CodeContentClassifier()
    qr_cases = {
        "url": "https://example.com/safe", "payment": "upi://pay?pa=dummy@example.invalid&am=1",
        "contact": "BEGIN:VCARD\nFN:Dummy Person\nTEL:0000000000\nEND:VCARD",
        "wifi": "WIFI:T:WPA;S:Dummy;P:not-real;;", "text": "Privacy Protection Demo",
        "unknown": "?",
    }
    classifications, qr_latencies = {}, []
    for name, payload in qr_cases.items():
        started = time.perf_counter()
        detected = detector.detect_qr(qr_image(payload), exhaustive=True)
        qr_latencies.append((time.perf_counter() - started) * 1000)
        classifications[name] = {
            "detected": len(detected), "decoded": bool(detected and detected[0].payload is not None),
            "classification": classifier.classify(
                detected[0].payload if detected else None, code_kind="qr",
            ).content_type,
            "confidence": None,
        }
    rotations = {}
    rotation_source = qr_image("https://example.com/rotation")
    for angle in (0, 15, 30, 45, 90):
        detected = detector.detect_qr(rotate_bound(rotation_source, angle), exhaustive=True)
        rotations[str(angle)] = {"detected": len(detected), "decoded": bool(detected and detected[0].payload)}

    barcode = ean13(module=3)
    barcode_scene = np.full((700, 1200, 3), 255, np.uint8)
    barcode_scene[250:250 + barcode.shape[0], 300:300 + barcode.shape[1]] = barcode
    started = time.perf_counter()
    barcodes = detector.detect_barcodes(barcode_scene)
    barcode_latency = (time.perf_counter() - started) * 1000

    redecodes = {"qr": {}, "barcode": {}}
    qr = qr_image("https://example.com/protection")
    qr_item = detector.detect_qr(qr)[0]
    barcode_item = next(item for item in barcodes if item.payload)
    for method in ("blur", "pixelate", "blackout"):
        for name, pixels, item, detect in (
            ("qr", qr, qr_item, lambda value: detector.detect_qr(value, exhaustive=True)),
            ("barcode", barcode_scene, barcode_item, detector.detect_barcodes),
        ):
            protected = pixels.copy()
            box = (item.x1, item.y1, item.x2, item.y2)
            if method == "blur":
                ImageAnonymizer().blur_region(protected, box, "medium")
            elif method == "pixelate":
                ImageAnonymizer().pixelate_region(protected, box, "medium")
            else:
                ImageAnonymizer().blackout_region(protected, box)
            redecodes[name][method] = sum(
                detected.payload is not None for detected in detect(protected)
            )
    return {
        "qr": {"cases": classifications, "rotations": rotations,
               "average_latency_ms": round(statistics.mean(qr_latencies), 2)},
        "barcode": {
            "detected": len(barcodes), "decoded": sum(item.payload is not None for item in barcodes),
            "formats": sorted({item.format for item in barcodes if item.format}),
            "confidence": None, "latency_ms": round(barcode_latency, 2),
        },
        "post_protection_successful_decode_counts": redecodes,
    }


def _safe_ocr_results(ocr: OCRService, pixels: np.ndarray) -> tuple[list[OCRTextResult], float]:
    started = time.perf_counter()
    raw = ocr.extract_text(pixels)
    latency = (time.perf_counter() - started) * 1000
    return ([
        OCRTextResult(
            text_id=index, raw_text=item.raw_text, normalized_text=item.normalized_text,
            confidence=item.confidence,
            bounding_box=BoundingBox(x1=item.x1, y1=item.y1, x2=item.x2, y2=item.y2),
        ) for index, item in enumerate(raw, 1)
    ], latency)


def ocr_validation() -> dict:
    ocr = OCRService(list(settings.ocr_languages), settings.ocr_model_directory)
    classifier = SensitiveTextClassifier()
    positive_pixels, _ = sensitive_text_image()
    positives, positive_ms = _safe_ocr_results(ocr, positive_pixels)
    positive_types = sorted({item.type for item in classifier.classify(positives, [], [])})
    negatives, negative_ms = _safe_ocr_results(ocr, normal_text_image())
    negative_sensitive = classifier.classify(negatives, [], [])

    protected = positive_pixels.copy()
    ImageAnonymizer().blackout_region(protected, (0, 0, protected.shape[1], protected.shape[0]))
    reread, reread_ms = _safe_ocr_results(ocr, protected)
    reread_sensitive = classifier.classify(reread, [], [])

    # One combined reread keeps runtime bounded while genuinely checking all
    # three methods against dummy plate/card/document characters.
    sources = {
        "license_plate": synthetic_plate().pixels,
        "payment_card": synthetic_card().pixels,
        "identity_document": synthetic_document().pixels,
    }
    protected_mosaic = np.full((960, 1560, 3), 255, np.uint8)
    for row, pixels in enumerate(sources.values()):
        cell = cv2.resize(pixels, (500, 300), interpolation=cv2.INTER_AREA)
        for column, method in enumerate(("blur", "pixelate", "blackout")):
            altered = cell.copy()
            box = (0, 0, altered.shape[1], altered.shape[0])
            if method == "blur":
                ImageAnonymizer().blur_region(altered, box, "high")
            elif method == "pixelate":
                ImageAnonymizer().pixelate_region(altered, box, "high")
            else:
                ImageAnonymizer().blackout_region(altered, box)
            protected_mosaic[row * 320:row * 320 + 300, column * 520:column * 520 + 500] = altered
    protected_objects, object_reread_ms = _safe_ocr_results(ocr, protected_mosaic)
    compact_reread = "".join(
        character for item in protected_objects for character in item.normalized_text.upper()
        if character.isalnum()
    )
    object_readability = {
        "license_plate": "ZZ00TEST" in compact_reread,
        "payment_card": "4000000000000002" in compact_reread,
        "identity_document": "ABCDE1234F" in compact_reread,
    }
    return {
        "positive_lines_expected": 7, "ocr_lines_detected": len(positives),
        "sensitive_categories_detected": positive_types,
        "sensitive_category_count": len(positive_types),
        "normal_lines_expected": 3, "normal_lines_detected": len(negatives),
        "normal_text_false_positives": len(negative_sensitive),
        "post_blackout_ocr_lines": len(reread),
        "post_blackout_sensitive_items": len(reread_sensitive),
        "protected_object_original_still_readable": object_readability,
        "latency_ms": {
            "positive": round(positive_ms, 2), "negative": round(negative_ms, 2),
            "reread": round(reread_ms, 2), "protected_object_reread": round(object_reread_ms, 2),
        },
    }


def profile_and_critical_scene_validation() -> dict:
    scene = difficult_privacy_scene()
    decoded = DecodedImage(scene.pixels, scene.pixels.shape[1], scene.pixels.shape[0], "PNG")
    service = get_detection_service()
    # Exclude one-time lazy model initialization from the profile comparison.
    service.analyze(decoded, "synthetic-day16-warmup.png", AnalysisOptions(performance_profile="balanced"))
    output, responses = {}, {}
    for profile in ("fast", "balanced", "accuracy"):
        response = service.analyze(decoded, "synthetic-day16-critical.png", AnalysisOptions(performance_profile=profile))
        responses[profile] = response
        details = response.analysis
        counts = {
            "faces": details.face_detection.face_count,
            "license_plates": details.license_plate_detection.plate_count,
            "payment_cards": details.card_detection.card_count,
            "identity_documents": details.document_detection.document_count,
            "qr_codes": details.qr_detection.qr_count,
            "barcodes": details.barcode_detection.barcode_count,
            "sensitive_text": details.sensitive_text.count,
        }
        predicted_boxes = {
            "faces": [tuple(getattr(item.bounding_box, key) for key in ("x1", "y1", "x2", "y2")) for item in details.face_detection.faces],
            "license_plates": [tuple(getattr(item.bounding_box, key) for key in ("x1", "y1", "x2", "y2")) for item in details.license_plate_detection.plates],
            "payment_cards": [tuple(getattr(item.bounding_box, key) for key in ("x1", "y1", "x2", "y2")) for item in details.card_detection.cards],
            "identity_documents": [tuple(getattr(item.bounding_box, key) for key in ("x1", "y1", "x2", "y2")) for item in details.document_detection.documents],
            "qr_codes": [tuple(getattr(item.bounding_box, key) for key in ("x1", "y1", "x2", "y2")) for item in details.qr_detection.items],
            "barcodes": [tuple(getattr(item.bounding_box, key) for key in ("x1", "y1", "x2", "y2")) for item in details.barcode_detection.items],
            "sensitive_text": [tuple(getattr(item.bounding_box, key) for key in ("x1", "y1", "x2", "y2")) for item in details.sensitive_text.items],
        }
        box_metrics = {
            name: metric_dict(match_boxes(list(scene.boxes.get(name, ())), boxes, 0.25))
            for name, boxes in predicted_boxes.items()
        }
        output[profile] = {
            "latency_ms": response.performance.total_analysis_ms, "detections": counts,
            "metrics": box_metrics,
            "misses": {name: values["fn"] for name, values in box_metrics.items()},
            "assessment_status": details.privacy_risk.assessment.status,
            "risk_score": details.privacy_risk.score,
            "main_subject_status": details.main_subject.status,
        }
    accuracy = responses["accuracy"]
    result, _encoded = _protect(
        decoded, accuracy, ProtectionSettings(anonymization_method="blackout", strength="high"),
    )
    risk = result.protection.risk
    output["critical_protection"] = {
        "regions_protected": result.protection.regions_protected,
        "breakdown": result.protection.breakdown.model_dump(),
        "original_risk": risk.before.score if risk else None,
        "residual_risk": risk.after.score if risk else None,
        "residual_not_greater": bool(risk and risk.after.score <= risk.before.score),
        "download_encoding_succeeded": bool(_encoded[0]),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-profiles", action="store_true")
    parser.add_argument("--profiles-only", action="store_true")
    parser.add_argument("--ocr-only", action="store_true")
    args = parser.parse_args()
    report = {"fixture_policy": "synthetic/dummy/generated only"}
    if args.ocr_only:
        report["ocr"] = ocr_validation()
    elif not args.profiles_only:
        report.update({
            "face": face_calibration(),
            "payment_card": card_validation(),
            "license_plate": {
                "status": "completed" if settings.license_plate_model_path.is_file() else "unavailable",
                "metrics": None,
            },
            "identity_document": {
                "status": "completed" if settings.document_model_path.is_file() else "unavailable",
                "metrics": None,
            },
            "ocr": ocr_validation(),
            "codes": code_validation(),
        })
    if not args.skip_profiles:
        report["profiles_and_critical_scene"] = profile_and_critical_scene_validation()
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
