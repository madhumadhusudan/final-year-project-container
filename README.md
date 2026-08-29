# Social Media Privacy Guard

Context-aware, local image anonymization for safer social-media sharing. This B.E. final-year project detects privacy risks, identifies a likely main subject, and selectively protects only sensitive regions.

> **Current status: Day 10 — Identity-Document Privacy Pipeline**

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
- No permanent image storage and no external image-processing APIs.

## Day 10 Pipeline

```text
Upload and validate image
  → local object, face, plate, card, and OCR analysis
  → optional dedicated identity-document detection
  → OCR-assisted document classification and document-photo face filtering
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

`POST /analyze` returns the analysis once. `POST /protect` receives that analysis with the same original image and validates matching dimensions, so expensive detectors are not rerun. The protected image is returned directly as binary data; compact protection metadata is exposed in `X-Protection-Metadata`. No result database or permanent output file is used.

The risk score uses only detected privacy evidence. A confidently selected main subject adds no face risk; background/unclassified faces, plates, cards, identity documents, classified sensitive text, and main-subject uncertainty do. Confidence uses the bounded multiplier `0.7 + 0.3 × confidence`. Visibility uses bounded size bands tailored to each region type. Document bases are Aadhaar-like 28, PAN-like 23, passport 28, driving licence 23, and generic identity document 20, with a limited six-point OCR/readability bonus. Category caps are faces 35, plates 30, cards 35, identity documents 40, sensitive text 60, and context 10. Final totals are clamped to 100.

Related detections are grouped: readable plate OCR is a limited plate bonus, card number/expiry OCR is a limited card bonus, address and PIN code form one address exposure, and overlapping or matching masked OCR classifications are deduplicated. Missing or failed face, plate, card, or OCR modules produce a visible `partial` assessment instead of implying a comprehensive zero-risk result.

Region precedence is:

1. Full payment-card and plate regions cover sensitive OCR regions contained inside them.
2. Overlapping/adjacent remaining sensitive-text regions are merged.
3. Face regions remain independently protected.
4. The confidently identified main-subject face is restored last, preventing accidental anonymization from overlaps.

## Repository Structure

```text
backend/
  app/anonymization/           Safe region utilities and ImageAnonymizer
  app/context/                 Main-subject analysis
  app/detection/               YOLO, YuNet, plate, and card detectors
  app/ocr/                     Local OCR and text normalization
  app/privacy/                 Sensitive-text/document classification and privacy-risk scoring
  app/routes/                  /analyze and /protect
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

Missing plate/card/document models are reported as `unavailable`; a working model with zero detections is reported as `completed` with count `0`. No document model is bundled or inferred from COCO classes.

## API

- `GET /` — service identity
- `GET /health` — connectivity status
- `POST /analyze` — multipart `image`; returns structured analysis and timings
- `POST /api/v1/analyze/image` — compatibility alias for `/analyze`
- `POST /protect` — multipart `image`, serialized `analysis`, and serialized `settings`; returns protected image bytes

Default protection settings:

```json
{
  "protect_background_faces": true,
  "protect_license_plates": true,
  "protect_cards": true,
  "protect_identity_documents": true,
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

Day 10 adds safe synthetic tests for strict detector labels, Aadhaar/PAN OCR support, passport/driving-licence labels, generic-document uncertainty, false-positive class rejection, original-coordinate boxes, multiple document IDs, printed-document-face filtering, OCR grouping, visibility-weighted risk, partial assessment, whole-region blur/pixelate/blackout, toggles, API metadata, and responsive overlays. All Day 1–9 regression tests remain active.

## Privacy and Limitations

- Images and detection crops are not logged or stored by the application.
- Multipart handling may spool larger requests into the operating-system temporary area until the upload is closed.
- JPEG and WEBP results must be re-encoded, so pixels outside protected regions can have small codec-level differences; PNG preserves decoded non-sensitive pixels exactly.
- Image metadata/EXIF and PNG alpha are not retained in the current OpenCV output path.
- Detection quality depends on local models, lighting, pose, scale, occlusion, and OCR quality.
- Dedicated plate and payment-card detection cannot run until compatible local weights are supplied.
- Identity-document detection remains unavailable until a provenance-reviewed compatible local checkpoint is configured; the application does not ship or download one automatically.
- Risk is a calibrated exposure indicator, not a probability of harm or a guarantee that unavailable detectors would find nothing.
- Residual risk is derived from protection settings and successful-region metadata; detectors are intentionally not rerun on the altered image.
- No identity verification, authenticity checking, face recognition, GAN replacement, generative inpainting, authentication, database, cloud storage, publishing, video, or live-camera processing is included in Day 10.
