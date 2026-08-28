"""Local acceptance harness for an installed card model and safe synthetic scenes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from app.config import settings
from app.detection.card_detector import CardDetector


def _table_background() -> np.ndarray:
    height, width = 1024, 1536
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :, None]
    base = np.empty((height, width, 3), dtype=np.float32)
    base[:] = (72, 105, 135)
    base += x * np.array([18, 12, 7], dtype=np.float32)
    noise = np.random.default_rng(8).normal(0, 3, base.shape).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def _place_perspective(canvas: np.ndarray, crop: np.ndarray, corners: np.ndarray) -> np.ndarray:
    height, width = crop.shape[:2]
    source = np.float32([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]])
    transform = cv2.getPerspectiveTransform(source, corners.astype(np.float32))
    warped = cv2.warpPerspective(crop, transform, (canvas.shape[1], canvas.shape[0]))
    mask = cv2.warpPerspective(np.full((height, width), 255, np.uint8), transform, (canvas.shape[1], canvas.shape[0]))
    output = canvas.copy()
    output[mask > 0] = warped[mask > 0]
    return output


def _negative_scene(kind: str) -> np.ndarray:
    image = _table_background()
    if kind == "phone":
        cv2.rectangle(image, (630, 610), (880, 970), (20, 20, 25), -1)
        cv2.rectangle(image, (650, 650), (860, 920), (80, 50, 25), -1)
        cv2.circle(image, (755, 945), 10, (120, 120, 120), -1)
    elif kind == "wallet":
        cv2.rectangle(image, (570, 700), (960, 930), (35, 65, 105), -1)
        cv2.line(image, (580, 810), (950, 810), (20, 40, 70), 8)
        cv2.circle(image, (890, 810), 12, (160, 170, 175), -1)
    elif kind == "book":
        cv2.rectangle(image, (510, 610), (1010, 930), (45, 45, 180), -1)
        cv2.rectangle(image, (530, 630), (990, 910), (235, 225, 205), -1)
        cv2.line(image, (760, 630), (760, 910), (90, 80, 70), 5)
    else:
        cv2.rectangle(image, (500, 610), (1030, 930), (245, 245, 245), -1)
        for y in range(660, 880, 42):
            cv2.line(image, (550, y), (960, y), (90, 90, 90), 4)
    return image


def build_scenes(source: np.ndarray, detector: CardDetector) -> dict[str, np.ndarray]:
    close_run = detector.detect_with_diagnostics(source)
    if not close_run.detections:
        raise RuntimeError("The installed model did not detect the held-out source card.")
    largest = max(close_run.detections, key=lambda item: (item.x2 - item.x1) * (item.y2 - item.y1))
    crop = source[largest.y1:largest.y2, largest.x1:largest.x2]
    scenes = {"close_up_card": source}
    scenes["small_card_on_table"] = _place_perspective(
        _table_background(), crop, np.array([[640, 730], [900, 700], [925, 860], [625, 875]])
    )
    scenes["rotated_card"] = _place_perspective(
        _table_background(), crop, np.array([[614, 714], [884, 694], [909, 854], [599, 874]])
    )
    partial = scenes["small_card_on_table"].copy()
    cv2.rectangle(partial, (865, 695), (955, 890), (78, 108, 140), -1)
    scenes["partially_visible_card"] = partial
    for kind in ("phone", "wallet", "book", "rectangular_paper"):
        scenes[kind] = _negative_scene(kind)
    return scenes


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-card", type=Path, required=True, help="Safe synthetic/dummy card image")
    args = parser.parse_args()
    source = cv2.imread(str(args.source_card), cv2.IMREAD_COLOR)
    if source is None:
        raise SystemExit("The source card image could not be decoded.")
    detector = CardDetector(
        settings.card_model_path, settings.card_confidence_threshold,
        settings.card_inference_image_size, settings.card_tile_size, settings.card_tile_overlap,
    )
    report = {
        "model": detector.model_name,
        "class_names": detector.model_class_names,
        "confidence_threshold": detector.confidence_threshold,
        "inference_image_size": detector.inference_image_size,
        "tile_size": detector.tile_size,
        "tile_inference_image_size": detector.tile_inference_image_size,
        "scenes": {},
    }
    for name, scene in build_scenes(source, detector).items():
        run = detector.detect_with_diagnostics(scene)
        report["scenes"][name] = {
            "count": len(run.detections),
            "inference_time_ms": run.inference_time_ms,
            "detections": [
                {
                    "class_name": item.class_name, "confidence": round(item.confidence, 6),
                    "bounding_box": {"x1": item.x1, "y1": item.y1, "x2": item.x2, "y2": item.y2},
                }
                for item in run.detections
            ],
        }
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
