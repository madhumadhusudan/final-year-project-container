"""Video encoding and optional safe FFmpeg audio muxing."""

from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(frozen=True)
class EncodingCapabilities:
    ffmpeg_path: str | None
    ffprobe_path: str | None
    h264_available: bool

    @property
    def ffmpeg_available(self) -> bool:
        return self.ffmpeg_path is not None


@dataclass(frozen=True)
class FinalizeResult:
    codec: str
    audio_preserved: bool
    audio_message: str
    encoding_seconds: float


def detect_encoding_capabilities() -> EncodingCapabilities:
    ffmpeg = shutil.which("ffmpeg")
    ffprobe = shutil.which("ffprobe")
    h264 = False
    if ffmpeg:
        try:
            result = subprocess.run(
                [ffmpeg, "-hide_banner", "-encoders"], capture_output=True, text=True,
                timeout=10, check=False, shell=False,
            )
            h264 = "libx264" in result.stdout
        except (OSError, subprocess.SubprocessError):
            pass
    return EncodingCapabilities(ffmpeg, ffprobe, h264)


def source_has_audio(path: Path, capabilities: EncodingCapabilities) -> bool:
    try:
        if capabilities.ffprobe_path:
            result = subprocess.run(
                [capabilities.ffprobe_path, "-v", "error", "-select_streams", "a:0", "-show_entries", "stream=codec_type", "-of", "json", str(path)],
                capture_output=True, text=True, timeout=15, check=False, shell=False,
            )
            return bool(json.loads(result.stdout or "{}").get("streams"))
        if capabilities.ffmpeg_path:
            result = subprocess.run(
                [capabilities.ffmpeg_path, "-hide_banner", "-i", str(path)],
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                timeout=15, check=False, shell=False,
            )
            return "Audio:" in (result.stderr or "")
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    return False


class VideoFrameWriter:
    def __init__(self, path: Path, fps: float, width: int, height: int) -> None:
        self.path = path
        self.codec = "mp4v"
        self._writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
        if not self._writer.isOpened():
            self._writer.release()
            raise RuntimeError("The local video writer could not initialize MP4 output.")

    def write(self, frame) -> None:
        self._writer.write(frame)

    def close(self) -> None:
        self._writer.release()


def finalize_video(intermediate: Path, source: Path, output: Path, capabilities: EncodingCapabilities, cancel_event) -> FinalizeResult:
    started = time.perf_counter()
    has_audio = source_has_audio(source, capabilities)
    if not capabilities.ffmpeg_path:
        intermediate.replace(output)
        return FinalizeResult("MPEG-4 Part 2 (mp4v)", False, "Audio preservation unavailable in current environment.", time.perf_counter() - started)

    command = [capabilities.ffmpeg_path, "-y", "-i", str(intermediate), "-i", str(source), "-map", "0:v:0", "-map", "1:a:0?", "-shortest"]
    codec = "H.264 (libx264)" if capabilities.h264_available else "MPEG-4 Part 2 (mp4v)"
    command += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20"] if capabilities.h264_available else ["-c:v", "copy"]
    command += ["-c:a", "aac", "-movflags", "+faststart", str(output)]
    process = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, shell=False)
    while process.poll() is None:
        if cancel_event.wait(0.1):
            process.terminate()
            try:
                process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                process.kill()
            output.unlink(missing_ok=True)
            raise InterruptedError("Video processing was cancelled.")
    if process.returncode != 0 or not output.exists():
        output.unlink(missing_ok=True)
        intermediate.replace(output)
        return FinalizeResult("MPEG-4 Part 2 (mp4v)", False, "FFmpeg audio muxing failed; the protected video has no audio.", time.perf_counter() - started)
    intermediate.unlink(missing_ok=True)
    message = "Original audio was preserved." if has_audio else "The source video did not contain an audio stream."
    return FinalizeResult(codec, has_audio, message, time.perf_counter() - started)
