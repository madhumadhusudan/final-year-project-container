"""Manual safe-media acceptance run for the Day 14 two-person proof."""

from __future__ import annotations

import argparse
import json
import math
import threading
import time
import sys
import uuid
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import settings
from app.detection.service import get_detection_service
from app.video.models import VideoProcessSettings
from app.video.video_processor import OUTPUT_DIRECTORY, StoredVideo, inspect_video, process_video
from app.video.video_writer import detect_encoding_capabilities


def make_motion_video(image_path: Path, destination: Path, seconds: int = 3, fps: int = 12) -> None:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("The synthetic acceptance image could not be decoded.")
    height, width = image.shape[:2]
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the acceptance-video writer.")
    try:
        for index in range(seconds * fps):
            offset = round(math.sin(index / max(1, fps - 1) * math.pi * 2) * 10)
            transform = np.float32([[1, 0, offset], [0, 1, 0]])
            frame = cv2.warpAffine(image, transform, (width, height), borderMode=cv2.BORDER_REFLECT)
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="Safe fictional two-person image")
    parser.add_argument("--keep-output", action="store_true", help="Keep the protected MP4 for manual preview")
    args = parser.parse_args()
    source_path = Path("uploads") / f"acceptance_{uuid.uuid4().hex}.mp4"
    output_path = OUTPUT_DIRECTORY / f"privacy_acceptance_{uuid.uuid4().hex}.mp4"
    make_motion_video(args.image, source_path)
    metadata = inspect_video(source_path, source_path.name, source_path.stat().st_size, ".mp4", settings)
    source = StoredVideo(uuid.uuid4().hex, source_path, metadata, time.time())
    updates = []
    summary = process_video(
        source, output_path, get_detection_service(),
        VideoProcessSettings(
            preserve_main_subject=True, protect_background_faces=True,
            protect_license_plates=False, protect_cards=False,
            protect_identity_documents=False, protect_qr_codes=False,
            protect_barcodes=False, protect_sensitive_text=False,
            anonymization_method="blur", strength="high", quality_profile="balanced",
        ),
        settings, detect_encoding_capabilities(), threading.Event(),
        lambda state, progress, stage: updates.append({"state": state, "progress": progress, "stage": stage}),
    )
    capture = cv2.VideoCapture(str(output_path))
    validation = {
        "opened": capture.isOpened(),
        "frame_count": round(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": round(capture.get(cv2.CAP_PROP_FPS), 3),
        "width": round(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": round(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "progress_updates": len(updates),
        "progress_monotonic": all(
            before["progress"] is None or after["progress"] is None or before["progress"] <= after["progress"]
            for before, after in zip(updates, updates[1:])
        ),
    }
    capture.release()
    print(json.dumps({
        "summary": summary.model_dump(), "output_validation": validation,
        "output_path": str(output_path), "output_retained": args.keep_output,
    }, indent=2))
    if not args.keep_output:
        output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
