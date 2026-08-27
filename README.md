# Social Media Privacy Guard

Context-aware AI anonymization for safer social media sharing. This B.E. final-year project inspects an image before it is shared, identifies the likely main subject, and prepares background privacy-sensitive regions for selective anonymization.

> **Current status: Day 3 — Real AI Detection Engine**

Day 3 provides a working local image-analysis pipeline. It detects faces and people using real pretrained OpenCV models, filters weak predictions, classifies the likely main subject, and renders model-derived bounding boxes and explainable results. It never fabricates detections and does not yet modify or blur the image.

## What Works Today

- In-memory JPG/JPEG, PNG, and WEBP upload up to 10 MB.
- Content-signature, MIME, extension, corruption, and resolution validation.
- Real face detection with OpenCV's bundled pretrained Haar cascade.
- Real person detection with OpenCV's pretrained HOG/SVM pedestrian detector.
- Optional Ultralytics YOLO person-detector adapter behind the same interface.
- Centralized confidence filtering for faces and people.
- Main-subject V1 ranking from relative face area, image-center proximity, and confidence.
- Structured recommendations ready for the Day 4 anonymization engine.
- Responsive model bounding boxes based on normalized original-image coordinates.
- Initial, analyzing, success, no-detection, and safe error states.
- Explainable per-detection result cards and processing time.
- No permanent raw-image storage and no third-party AI API.

## Detection Pipeline

```text
Uploaded image
  → upload/content validation
  → in-memory OpenCV decode
  → aspect-preserving inference resize
  → detector interface
      → OpenCV face detector
      → OpenCV HOG person detector (default)
      → Ultralytics YOLO adapter (optional)
  → coordinate restoration
  → confidence filtering
  → context/main-subject analysis
  → normalized structured response
  → responsive React overlay and results panel
```

Models are initialized once when the detection service is first requested and reused for later requests. Uploaded bytes and decoded pixels exist only for the request; the API explicitly closes each upload after reading it, does not persist input images to `uploads/`, and never exposes server paths. Starlette may temporarily spool a large multipart upload to the operating-system temp area before it is closed.

## Repository Structure

```text
frontend/
  src/components/              Upload, preview, overlay, states, results
  src/services/api.js          Health and image-analysis API client
  src/utils/detection.js       Responsive coordinate mapping and labels
  tests/                       Validation and coordinate unit tests
backend/
  app/config.py                Thresholds, limits, model selection, categories
  app/schemas.py               Internal and API detection models
  app/detection/base.py        Detector interface and composite detector
  app/detection/opencv_detector.py
  app/detection/yolo_detector.py
  app/detection/service.py     Preprocess, infer, filter, postprocess
  app/context/main_subject.py  Explainable subject ranking
  app/routes/analysis.py       Versioned analysis endpoint
  app/utils/image_validation.py
  models/                      Local custom weights (ignored by Git)
  tests/                       Context, filtering, validation, and API tests
docs/                          Project screenshots
```

## Prerequisites

- Node.js 20 or newer and npm
- Python 3.10 or newer
- Windows PowerShell commands are shown below

## Backend Setup

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

The API is available at <http://127.0.0.1:8000>, with interactive documentation at <http://127.0.0.1:8000/docs>.

Configuration is read from process environment variables. `backend/.env.example` documents all available values. If using an `.env` file, load it in the shell or process manager before starting Uvicorn.

| Variable | Default | Purpose |
|---|---:|---|
| `MAX_IMAGE_BYTES` | `10485760` | Maximum compressed upload size |
| `FACE_CONFIDENCE_THRESHOLD` | `0.62` | Minimum displayed face score |
| `PERSON_CONFIDENCE_THRESHOLD` | `0.55` | Minimum displayed person score |
| `DEFAULT_CONFIDENCE_THRESHOLD` | `0.55` | Fallback category threshold |
| `MAIN_SUBJECT_AREA_THRESHOLD` | `0.08` | Minimum face/image area ratio |
| `MAIN_SUBJECT_SCORE_THRESHOLD` | `0.55` | Minimum combined subject score |
| `INFERENCE_MAX_DIMENSION` | `1600` | Longest model-input side |
| `DETECTOR_BACKEND` | `opencv` | `opencv` or optional `yolo` person engine |
| `YOLO_WEIGHTS_PATH` | `models/yolov8n.pt` | Local YOLO weights path |

## Frontend Setup

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Vite serves the UI at <http://localhost:5173>. Copy `frontend/.env.example` to `frontend/.env` only if the backend URL differs.

## API

### `POST /api/v1/analyze/image`

Send `multipart/form-data` with one `image` field. A successful response contains original dimensions, model-derived detections, normalized boxes, context decisions, a summary, and measured inference time.

```json
{
  "success": true,
  "image": { "width": 1920, "height": 1080, "format": "JPEG" },
  "detections": [
    {
      "id": "det_face_1",
      "category": "face",
      "label": "Main subject",
      "confidence": 0.974,
      "boundingBox": { "x": 640, "y": 220, "width": 420, "height": 420, "x2": 1060, "y2": 640 },
      "normalizedBoundingBox": { "x": 0.333, "y": 0.204, "width": 0.219, "height": 0.389 },
      "center": { "x": 0.443, "y": 0.398 },
      "relativeArea": 0.085,
      "isMainSubject": true,
      "subjectScore": 0.91,
      "recommendedAnonymization": false,
      "explanation": "Selected as the main subject because it occupies 8.5% of the image, is positioned near the image center, has high detection confidence.",
      "source": "opencv_haar_face"
    }
  ],
  "summary": { "totalObjects": 1, "faces": 1, "people": 0, "backgroundFaces": 0, "mainSubjectDetected": true },
  "processingTimeMs": 184
}
```

Other endpoints:

- `GET /` — API identity and running status
- `GET /health` — frontend connectivity health check

## Optional YOLO Model Setup

The default installation is fully functional without downloads beyond Python packages. To use the YOLO adapter for person detection:

1. Install the optional `ultralytics` package in the backend virtual environment.
2. Place trusted weights at `backend/models/yolov8n.pt`, or set `YOLO_WEIGHTS_PATH` to another local file.
3. Set `DETECTOR_BACKEND=yolo` before starting Uvicorn.

The base COCO YOLO model supplies the `person` class only. Custom fine-tuned weights can later extend the adapter and category registry without changing the API or frontend. Model binaries are intentionally ignored by Git.

## Supported Detections and Limitations

| Category | Day 3 status | Notes |
|---|---|---|
| Face / background face | Working | Frontal-face cascade; difficult profiles, occlusion, and very small faces can be missed |
| Person / background person | Working | HOG pedestrian model; best for upright, mostly visible people |
| Main subject | Working V1 | Face-based heuristic, not identity recognition |
| License plate | Requires custom model | Architecture placeholder only |
| Aadhaar, PAN, passport, driving licence, voter ID | Requires custom model | Not claimed or shown as working |
| Payment cards, documents, screens, QR/barcodes | Requires custom model | Planned custom classes/detectors |
| Sensitive text | Requires OCR/model work | Planned; no fake OCR results |

OpenCV cascade/HOG scores are model margins transformed to a consistent 0–1 display score; they are useful for filtering and ranking but are not calibrated probabilities. The main-subject rule is classification only and does not blur anything.

## Tests

```powershell
cd backend
.\.venv\Scripts\python.exe -m unittest discover -s tests -v

cd ..\frontend
npm.cmd test
npm.cmd run build
```

Backend tests cover valid, corrupt, mismatched, and oversized uploads; no detections; one/multiple faces; confidence filtering; main-subject calculation; coordinate restoration/normalization; and safe model failures. Frontend tests cover file validation, confidence labels, and normalized-to-responsive box mapping. `frontend/tests/browser-flow.mjs` verifies the live backend flow and layouts at 320, 375, 430, 768, 1024, 1366, 1440, and 1920 px when the documented local browser harness is running.

## Screenshots

- Existing Day 2 desktop reference: `docs/day2-desktop.png`
- Existing Day 2 mobile reference: `docs/day2-mobile-emulated.png`
- Day 3 responsive mobile workspace: `docs/day3-mobile-emulated.png`
- A detection screenshot can be added after analyzing consented project imagery.

## Day 4 Preparation

Every detection already includes a stable request-scoped ID, category, confidence, original and normalized bounding box, relative area, main-subject flag, explainable score, and `recommendedAnonymization`. Day 4 can consume this response to apply selective blur, pixelation, or blackout while protecting the selected main subject.

## Git Checkpoint

After reviewing the changes and tests:

```powershell
git add .
git commit -m "Day 3: implement AI sensitive object detection engine"
```

The existing repository is preserved; do not initialize a nested Git repository.
