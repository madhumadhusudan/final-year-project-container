# Day 14: Video Privacy Protection

The `/video` workspace adds local video-file anonymization without changing the
existing image or live-camera paths.

## Safety and storage

- MP4, MOV, AVI, and WEBM are accepted subject to actual OpenCV codec decoding.
- Extension, MIME, signature, byte size, dimensions, frame count, FPS, duration,
  and first-frame decoding are validated.
- Defaults are 150 MB and 180 seconds; both are configurable.
- UUID paths are used. User filenames never enter a subprocess command.
- Frames stream from reader to writer in memory; there is no frame dump.
- Sources and intermediates are removed after processing. Finished outputs use a
  one-hour default TTL and opportunistic cleanup.
- FFmpeg uses argument arrays with `shell=False`. No upload leaves the machine.

## Sampling profiles

| Profile | Width | General | Faces | Plates | Cards | Documents | QR/barcode | OCR |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Performance | 640 | 10 | 5 | 15 | 15 | 15 | 15 | 30 |
| Balanced | 960 | 7 | 3 | 10 | 10 | 10 | 10 | 20 |
| Accuracy | 1280 | 4 | 2 | 5 | 6 | 6 | 6 | 12 |

Intervals are frame counts. Unavailable optional detectors are disabled in the UI
and listed in a partial assessment.

## Tracking and main subject

Same-category matching combines intersection over union, normalized center
distance, and box-area similarity. Matched motion updates a smoothed velocity,
which predicts boxes between sampled checks. IDs such as `face_track_1` and
`plate_track_1` exist only within one job and are not biometric identities.
Tracks expire 18 frames after their last successful match by default.

The first confident main-subject face track remains selected. If a scheduled face
check misses it, no face is preserved on that frame; this deliberately prefers
temporary over-protection. A replacement is selected only after expiry.

## Encoding, audio, and risk

OpenCV first writes source-resolution, source-FPS MP4V. If FFmpeg is present, it
maps optional original audio and uses H.264 only when `libx264` is actually listed.
Failures retain MP4V and explicitly report missing audio.

Video risk combines peak active-region risk (65%) and category persistence (35%).
Frequency is labelled as sampled-frame data. Residual risk is conservative and is
not forced to zero. All performance and output measurements come from the job.

## Known limitations

- No face recognition or cross-session identity tracking.
- No representative-frame manual main-subject picker yet.
- Severe motion or long occlusion can temporarily over-protect or lose a track.
- Unsupported codec variants are rejected during real decode validation.
- A real two-person clip should be manually acceptance-tested on the final demo
  machine because model quality and codec availability are machine-dependent.
