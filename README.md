# Social Media Privacy Guard

Context-aware, local image anonymization for safer social-media sharing. This B.E. final-year project detects privacy risks, identifies a likely main subject, and selectively protects only sensitive regions.

> **Current status: Day 16 — Accuracy validation and reliability hardening (partial: optional plate/document models remain unavailable)**

The Day 14 feature set is now performance-tuned: image analysis uses reusable detector-sized views,
selective modules, safe YOLO/face overlap, ROI OCR, cached model instances, and expiring analysis IDs.

Day 16 adds a synthetic-only formal accuracy suite, detector threshold/box metrics,
FAST/BALANCED/ACCURACY comparison, real QR/barcode/OCR protection rereads, and
regression hardening. See `docs/day16_accuracy_report.md` for measured results and
explicitly unverified criteria; unavailable detectors are never reported as successful.

## What Works

- In-memory JPG/JPEG, PNG, and WEBP validation and processing up to 10 MB.
- YOLOv8 object detection and dedicated OpenCV YuNet face detection.
- Explainable main-subject classification with background-face roles and a safe uncertain-subject state.
- Optional local license-plate and payment-card YOLO detectors with honest unavailable/error states.
- Local EasyOCR extraction and conservative sensitive-text classification for email, phone, PAN-like, Aadhaar-like, card-like, address, PIN-code, URL, expiry, and detector-supported plate text.
- Original-coordinate overlays with independent layer controls.
- Selective Gaussian blur, pixelation, and blackout with Low/Medium/High strength where applicable.
- Independent protection toggles for background faces, plates, cards, and sensitive text.
- Main-subject face preservation, including protection from overlapping padded regions.
- Safe fallback that protects all detected faces when the main subject is uncertain or absent.
- Responsive Before/After comparison, real category counts, warnings, and local download.
- Deterministic privacy risk score from 0–100 with confidence/visibility weighting, category caps, factor explanations, and detector-completeness status.
- Original-versus-residual risk calculation from successful protection metadata, including point and percentage reduction.
- Dedicated local identity-document detector contract with strict model-native label validation and honest unavailable/error states.
- Explainable OCR-assisted Aadhaar/PAN/passport/driving-licence classification, printed-document-face filtering, document risk, responsive overlays, and whole-document protection.
- Real local OpenCV QR detection/decoding with multiple-code support, polygon geometry, safe URL/payment/contact/Wi-Fi classification, and masked previews.
- Real local OpenCV barcode detection/decoding for EAN-8, EAN-13, UPC-A, and UPC-E, with honest format/decode status and low-risk handling for standalone retail codes.
- Document/card spatial association, grouped QR/barcode risk, independent overlay/protection controls, padded code anonymization, and explicit module failure states.
- No permanent image storage and no external image-processing APIs.
- `/live` protected camera preview with explicit video-only permission, deterministic cleanup, front/back camera switching, and a privacy-first full-frame fallback blur.
- In-memory downscaled frame analysis with single-request backpressure, frame-ID stale-result rejection, multi-rate detector scheduling, and adaptive face cadence.
- Browser-side blur, pixelation, and blackout over tracked regions, with coordinate-space conversion, temporal smoothing, detection-miss grace periods, and stable/manual main-subject selection.
- Real render/analysis FPS and request-latency measurements, live risk/warnings, tab visibility throttling, and honest detector availability.
- `/video` upload with strict container/size/duration validation and real OpenCV metadata.
- Bounded asynchronous video jobs with frame-based progress, cancellation, and transient-file cleanup.
- Configurable detector sampling, category-aware tracking, stable session-only IDs, motion prediction, and track expiry.
- Temporally stable main-subject preservation with safe protection on uncertain or missed face checks.
- Streaming blur, pixelation, or blackout at source resolution/FPS without per-frame image dumps.
- MP4 output with optional safe FFmpeg H.264/audio muxing, honest codec/audio reporting, preview, and download.
- Peak/persistence video risk, sampled-frame category frequency, residual risk, partial-coverage status, and measured timings.

## Day 14 Video Pipeline

```text
Validated temporary video upload
  → real duration / resolution / FPS / frame-count metadata
  → bounded in-memory job queue (one heavy job by default)
  → streaming OpenCV decode (no frame dump)
  → profile-based detector sampling on a bounded analysis image
  → category-aware temporal association and short track grace
  → stable main-subject track or safe all-face protection fallback
  → padded blur / pixelate / blackout on every output frame
  → MP4 encode at source resolution and FPS
  → optional FFmpeg H.264 encode and original-audio mux
  → transient protected preview/download and sampled-frame risk report
```

Balanced analyzes at up to 960 px width: general objects every 7 frames,
faces every 3, plates/cards/documents/codes every 10, and OCR every 20.
Performance uses 640 px with wider intervals; Accuracy uses 1280 px with
narrower intervals. Every decoded frame is still protected and encoded.

## Day 13 Live Pipeline

```text
Webcam video (never displayed raw)
  → protected canvas rendered at browser refresh cadence
  → downscaled 480 / 640 / 768 px JPEG sample
  → one in-flight POST /analyze-frame request at most
  → selected local detector groups (faces fast; heavy modules slower)
  → analysis-frame boxes mapped to native video and canvas coordinates
  → category-aware spatial track association + smoothing + miss grace
  → preserve stable/manual main-subject track when requested
  → browser canvas blur / pixelate / blackout
```

Balanced mode targets face analysis every 180 ms, QR/barcodes every 1000 ms,
plates every 1200 ms, cards and OCR every 1800 ms, and documents every 2000 ms.
Performance and Accuracy modes change both analysis width and intervals. These are
scheduling targets, not promised frame rates; the face interval increases within
bounds when measured request latency is high. Only one heavy group is added to a
face request at a time, OCR never runs on every rendered frame, and hidden tabs
pause new analysis requests.

The current local installation reports plate and identity-document detection as
unavailable because their dedicated model weights are absent. Their live controls
are disabled rather than implying coverage. Face, payment-card, QR, barcode, and
OCR availability is discovered from `GET /live/capabilities`.

## Optimized Image Pipeline

```text
Instant object-URL preview
  -> one validated image decode
  -> cached detector-sized BGR views
  -> selected YOLO + face inference in parallel on CPU
  -> conditional plate/card/document/code detectors
  -> ROI OCR on document/card/plate/text-like regions
  -> context and risk calculation
  -> bounded 10-minute analysis session
  -> /protect anonymization using analysis_id (no detector rerun)
```

Image profiles use these maximum sides: Fast `512` YOLO / `640` face / `768` privacy
objects / `480` OCR; Balanced `640` / `800` / `960` / `640`; Accuracy `960` /
`960` / `1280` / `1280`. Accuracy uses bounded full-frame OCR; Fast and Balanced
use primary/text-like crops. All coordinates map back to the untouched original used
by protection and download.

## Day 11 Pipeline

```text
Upload and validate image
  → local object, face, plate, card, and OCR analysis
  → optional dedicated identity-document detection
  → OCR-assisted document classification and document-photo face filtering
  → local QR and barcode region detection with optional local decode
  → safe payload classification and document/card spatial association
  → main-subject/context classification
  → privacy-risk feature extraction, grouping, and deterministic scoring
  → user privacy settings
  → sanitize, clamp, pad, merge, and prioritize regions
  → selective blur / pixelate / blackout
  → restore confidently identified main-subject face
  → in-memory image encoding
  → residual-risk calculation from successfully protected regions
  → Before/After preview, risk reduction, and local download
```

`POST /analyze` returns the analysis once with an expiring `analysis_id`. `POST /protect` receives that ID with the same original image, verifies its SHA-256 content hash, and reuses the cached regions, so expensive detectors and OCR are not rerun. The earlier serialized-analysis request remains compatible. The protected image is returned directly as binary data; compact protection metadata is exposed in `X-Protection-Metadata`. No result database or permanent output file is used.

The risk score uses only detected privacy evidence. A confidently selected main subject adds no face risk; background/unclassified faces, plates, cards, identity documents, QR codes, barcodes, classified sensitive text, and main-subject uncertainty do. QR risk depends on safe content category and context. Standalone decoded retail barcodes remain low risk, while undecodable or context-associated codes remain protectable. Identity-document codes add only a limited readability/context bonus instead of duplicating the full document risk. Final totals are clamped to 100.

Related detections are grouped: readable plate OCR is a limited plate bonus, card number/expiry OCR is a limited card bonus, address and PIN code form one address exposure, and overlapping or matching masked OCR classifications are deduplicated. Missing or failed face, plate, card, or OCR modules produce a visible `partial` assessment instead of implying a comprehensive zero-risk result.

Region precedence is:

1. Full identity-document, payment-card, and plate regions take precedence.
2. QR and barcode regions use safety padding and are skipped when a protected parent already covers them.
3. Full parent/code regions cover sensitive OCR regions contained inside them.
4. Overlapping/adjacent remaining sensitive-text regions are merged.
5. Face regions remain independently protected.
6. The confidently identified main-subject face is restored last, preventing accidental anonymization from overlaps.

## Repository Structure

```text
backend/
  app/anonymization/           Safe region utilities and ImageAnonymizer
  app/context/                 Main-subject and code-parent association
  app/detection/               YOLO, YuNet, plate, card, document, QR, and barcode detectors
  app/ocr/                     Local OCR and text normalization
  app/privacy/                 Sensitive-text/document classification and privacy-risk scoring
  app/routes/                  Image, live-camera, and video job APIs
  app/video/                   Video validation, analysis, tracking, jobs, and encoding
  app/utils/                   Secure image validation
  models/                      Local model weights; most are Git-ignored
  tests/                       Detection, OCR, API, and anonymization tests
frontend/
  src/components/              Upload, analysis, controls, results, comparison
  src/services/api.js          Health, analysis, and protection clients
  tests/                       Unit and browser-flow tests
docs/                          Responsive project screenshots
```

## Setup

Prerequisites: Node.js 20+, npm, and Python 3.10+.

Backend:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

Frontend:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

The API runs at <http://127.0.0.1:8000> and Vite at <http://127.0.0.1:5173>. Copy the `.env.example` files only when defaults need to change.

## Local Models

The checked-in YuNet model supports face detection. `yolov8n.pt` and EasyOCR assets may be present locally but are ignored by Git. Dedicated plate and card protection activates only when trusted compatible weights are placed at the configured paths.

| Variable | Default | Purpose |
|---|---|---|
| `MAX_IMAGE_BYTES` | `10485760` | Maximum compressed upload size |
| `YOLO_WEIGHTS_PATH` | `backend/models/yolov8n.pt` | Local object detector weights |
| `YOLO_CONFIDENCE_THRESHOLD` | `0.35` | Object threshold |
| `FACE_MODEL_PATH` | `backend/models/face_detection_yunet_2023mar.onnx` | YuNet face model |
| `FACE_CONFIDENCE_THRESHOLD` | `0.75` | Face threshold |
| `LICENSE_PLATE_MODEL_PATH` | `backend/models/license_plate_detector.pt` | Optional local plate model |
| `CARD_MODEL_PATH` | `backend/models/card_detector.pt` | Optional local card model |
| `DOCUMENT_MODEL_PATH` | `backend/models/document_detector.pt` | Optional dedicated identity-document model |
| `DOCUMENT_MODEL_SOURCE` | empty | Local model provenance record |
| `DOCUMENT_MODEL_LICENSE` | empty | Local model usage/redistribution terms |
| `DOCUMENT_CONFIDENCE_THRESHOLD` | `0.35` | Document detector threshold |
| `DOCUMENT_INFERENCE_IMAGE_SIZE` | `960` | Document detector input size |
| `PRIVACY_OBJECT_CONFIDENCE_THRESHOLD` | `0.35` | Plate/card threshold |
| `OCR_LANGUAGES` | `en` | EasyOCR languages |
| `OCR_MODEL_DIRECTORY` | `backend/models/easyocr` | Local OCR assets |
| `MAX_VIDEO_BYTES` | `157286400` | Maximum temporary video upload (150 MB) |
| `MAX_VIDEO_DURATION_SECONDS` | `180` | Maximum prototype duration (3 minutes) |
| `MAX_CONCURRENT_VIDEO_JOBS` | `1` | Bounded heavy video workers |
| `VIDEO_OUTPUT_TTL_SECONDS` | `3600` | Protected-output download lifetime |
| `VIDEO_TRACK_EXPIRY_FRAMES` | `18` | Maximum track age since its last detection |
| `VIDEO_DEFAULT_FPS` | `25` | Safe fallback for invalid source FPS metadata |
| `IMAGE_PARALLEL_DETECTORS` | `true` | Safely overlap independent YOLO and face inference |
| `IMAGE_DETECTOR_WORKERS` | `4` | Bounded image detector worker pool |
| `ANALYSIS_SESSION_TTL_SECONDS` | `600` | In-memory analysis reuse lifetime |
| `ANALYSIS_SESSION_MAX_ENTRIES` | `32` | Maximum expiring analysis records |
| `YOLO_MAX_DETECTIONS` | `100` | Upper bound for general object predictions |

Missing plate/card/document models are reported as `unavailable`; a working model with zero detections is reported as `completed` with count `0`. No document model is bundled or inferred from COCO classes.

## API

- `GET /` — service identity
- `GET /health` — connectivity status
- `POST /analyze` — multipart `image` plus optional `analysis_options` or `performance_profile`; returns structured analysis, `analysis_id`, and timings
- `POST /api/v1/analyze/image` — compatibility alias for `/analyze`
- `POST /protect` — multipart `image`, `analysis_id` (or backward-compatible serialized `analysis`), and serialized `settings`; returns protected image bytes without detector inference
- `GET /live/capabilities` — reports genuinely loaded local live detector modules and supported analysis widths
- `POST /analyze-frame` — multipart downscaled `image`, `frame_id`, `captured_at_ms`, detector `modules`, and `preserve_main_subject`; returns transient regions and timings
- `GET /video/capabilities` — limits, profiles, detector coverage, and FFmpeg/H.264 support
- `POST /video/upload` — multipart `video`; returns a temporary ID and real metadata
- `DELETE /video/upload/{upload_id}` — removes an unused temporary upload
- `POST /video/process` — accepts an upload ID and settings; returns a background job ID
- `GET /video/status/{job_id}` — real progress, stage, state, error, and final report
- `DELETE /video/cancel/{job_id}` — signals cancellation and cleanup
- `GET /video/result/{job_id}` — streams the actual protected MP4 while available

Default protection settings:

```json
{
  "protect_background_faces": true,
  "protect_license_plates": true,
  "protect_cards": true,
  "protect_identity_documents": true,
  "protect_qr_codes": true,
  "protect_barcodes": true,
  "protect_sensitive_text": true,
  "anonymization_method": "blur",
  "strength": "medium"
}
```

The protection metadata header decodes to:

```json
{
  "status": "completed",
  "method": "blur",
  "strength": "medium",
  "regions_protected": 5,
  "breakdown": {
    "background_faces": 1,
    "license_plates": 1,
    "cards": 1,
    "identity_documents": 1,
    "qr_codes": 2,
    "barcodes": 1,
    "sensitive_text": 1
  },
  "main_subject_preserved": true,
  "warnings": [],
  "risk": {
    "before": { "score": 72, "level": "HIGH" },
    "after": { "score": 7, "level": "LOW" },
    "reduction": 65,
    "reduction_percent": 90.3
  }
}
```

Counts come from validated regions actually processed; detector counts are never fabricated.

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

cd ..\frontend
npm.cmd test
npm.cmd run build
```

Day 13 adds synthetic live-frame API tests for validation, frame IDs, response
schema, module availability, and no disk persistence. Frontend unit tests cover
three-space coordinate mapping, region association, smoothing, miss grace,
multi-rate scheduling, stale-response rejection, stream-track cleanup, risk, and
protection toggles. All earlier regression tests remain active.

Day 14 adds generated, non-private video tests for real metadata, invalid-container
rejection, asynchronous output/download, explicit upload deletion, cancellation
cleanup, moving-region association/expiry, stable main-subject preservation, and
sampled face/plate/QR/OCR masks between detector frames. Frontend tests cover
video extension/size validation and metadata formatting.

## Privacy and Limitations

- Images and detection crops are not logged or stored by the application.
- Multipart handling may spool larger requests into the operating-system temporary area until the upload is closed.
- JPEG and WEBP results must be re-encoded, so pixels outside protected regions can have small codec-level differences; PNG preserves decoded non-sensitive pixels exactly.
- Image metadata/EXIF and PNG alpha are not retained in the current OpenCV output path.
- Detection quality depends on local models, lighting, pose, scale, occlusion, and OCR quality.
- Dedicated plate and payment-card detection cannot run until compatible local weights are supplied.
- Identity-document detection remains unavailable until a provenance-reviewed compatible local checkpoint is configured; the application does not ship or download one automatically.
- QR and barcode quality depends on code size, quiet zones, rotation, perspective, blur, and occlusion. OpenCV may detect a region without decoding it; this is reported as `decoded: false`, never as safe.
- Decoded code payloads are transient local strings. The API/UI returns only safe categories and masked previews; it never opens URLs, initiates payments, connects to Wi-Fi, executes payload text, or sends it to a cloud service.
- Risk is a calibrated exposure indicator, not a probability of harm or a guarantee that unavailable detectors would find nothing.
- Residual risk is derived from protection settings and successful-region metadata; detectors are intentionally not rerun on the altered image.
- Live camera access still depends on browser secure-context rules (`localhost` is accepted by modern browsers; remote mobile testing normally requires HTTPS).
- Camera permission, physical front/back switching, two-person movement quality, and the hardware privacy indicator require manual validation on the target device; headless tests cannot certify camera hardware behavior.
- Canvas masks follow the latest local detections and reduce flicker but cannot guarantee zero exposure during fast movement, severe occlusion, detector misses, or unsupported categories. If detection becomes unavailable, the UI marks protection unavailable and applies a full-frame safety blur.
- Video detection quality drops with fast motion, long occlusion, tiny regions, motion blur, or wider detector cadence. Prediction and expiry reduce flicker but cannot guarantee perfect masks.
- Manual representative-frame main-subject picking is not included; automatic temporal selection fails safe when its selected track is missed.
- Without FFmpeg, OpenCV produces MP4V without audio and the UI reports that explicitly. With FFmpeg and `libx264`, the app encodes H.264 and maps optional original audio using argument arrays and `shell=False`.
- Sources and intermediates are removed after success, failure, or worker-observed cancellation. Protected outputs are transient and cleaned opportunistically after their configured TTL.
- No identity verification, authenticity checking, face recognition, GAN replacement, generative inpainting, authentication, database, cloud storage, publishing, or camera recording is included in Day 14.
