"""Benchmark real /analyze latency without retaining uploaded images."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import httpx


SIZES = ((640, 480), (1280, 720), (1920, 1080))
METRICS = (
    "total_analysis_ms", "object_detection_ms", "face_detection_ms",
    "license_plate_detection_ms", "card_detection_ms", "document_detection_ms",
    "ocr_detection_ms", "qr_detection_ms", "barcode_detection_ms",
    "context_analysis_ms",
)


def benchmark(base_url: str, source: Path, runs: int, profile: str) -> dict:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise SystemExit(f"Could not decode benchmark image: {source}")
    report = {"source": source.name, "runs_per_size": runs, "warmup_ignored": True, "profile": profile, "sizes": {}}
    with httpx.Client(timeout=300) as client:
        for width, height in SIZES:
            resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
            ok, encoded = cv2.imencode(".jpg", resized, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if not ok:
                raise SystemExit("Could not encode benchmark input")
            samples = []
            for iteration in range(runs + 1):
                started = time.perf_counter()
                response = client.post(
                    f"{base_url.rstrip('/')}/analyze",
                    files={"image": (f"benchmark-{width}x{height}.jpg", encoded.tobytes(), "image/jpeg")},
                    data={"performance_profile": profile},
                )
                http_ms = (time.perf_counter() - started) * 1000
                response.raise_for_status()
                if iteration:
                    performance = response.json()["performance"]
                    samples.append({"http_total_ms": round(http_ms, 2), **performance})
            report["sizes"][f"{width}x{height}"] = {
                "mean": {
                    metric: round(statistics.mean(sample[metric] for sample in samples), 2)
                    for metric in ("http_total_ms", *METRICS)
                },
                "samples": samples,
            }
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Safe local benchmark image")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--runs", type=int, default=2, choices=range(2, 11), metavar="2-10")
    parser.add_argument("--profile", choices=("fast", "balanced", "accuracy"), default="balanced")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = benchmark(args.url, args.image, args.runs, args.profile)
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
