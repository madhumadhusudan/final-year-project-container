# Social Media Privacy Guard

Context-aware AI anonymization for safer social media sharing. This B.E. final-year project aims to inspect an image before it is shared, preserve its main subject, and anonymize privacy-sensitive content in the background.

> **Current Status: Day 2 — Image Upload and Privacy Analysis Interface**

The React frontend now provides a responsive, browser-only image selection workspace with drag and drop, validation, preview, metadata, change/remove actions, and honest placeholders for future privacy analysis. The FastAPI health check remains connected to the live backend. Detection, anonymization, OCR, and risk scoring are **Planned / In Development** and are not represented by mock results.

## Problem Being Solved

Images shared online can unintentionally expose faces, number plates, identity documents, or private text. Existing all-or-nothing blur tools provide little awareness of image context. This project will provide user-controlled, context-aware privacy protection while keeping the main subject useful and recognizable.

## Planned Features

- Detect faces, background people, vehicle number plates, identity documents, and sensitive text.
- Identify and preserve the main subject.
- Apply Gaussian blur, pixelation, or blackout to selected regions.
- Extract sensitive text with OCR.
- Generate an explainable privacy risk score from 0–100.
- Let users choose which categories to protect.
- Process images locally wherever practical.

All items above are **Planned / In Development**.

## Technology Stack

- Frontend: React, Vite, Tailwind CSS, JavaScript
- Backend: Python, FastAPI, Uvicorn
- Planned AI: YOLOv8, OpenCV, EasyOCR

## Project Architecture

```text
frontend/                    React client
  src/components/            Reusable UI components
  src/services/              Backend API access
  src/pages/                 Future page-level views
  src/hooks/                 Future React hooks
  src/utils/                 Future client utilities
backend/                     FastAPI service
  app/detection/             Planned object detection
  app/context/               Planned subject/context analysis
  app/anonymization/         Planned privacy transformations
  app/ocr/                   Planned sensitive-text extraction
  app/privacy/               Planned risk evaluation
  app/utils/                 Shared backend utilities
  models/                    Local model weights (ignored by Git)
  uploads/                   Private input files (ignored by Git)
  outputs/                   Generated files (ignored by Git)
  tests/                     Backend tests
docs/                        Project documentation
```

## Prerequisites

- Node.js 20 or newer and npm
- Python 3.10 or newer
- VS Code with a PowerShell terminal (commands below are for Windows)

If PowerShell blocks `npm.ps1` because of the execution policy, use `npm.cmd` as shown below. This avoids changing the machine's security policy.

## Start the Backend

Open a terminal at the repository root:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn main:app --reload
```

These commands call the virtual environment directly and therefore work even when PowerShell blocks activation scripts. If script execution is enabled, you may instead activate it with `.\.venv\Scripts\Activate.ps1` and then use `python` normally.

The API is available at <http://127.0.0.1:8000>. Interactive API docs are at <http://127.0.0.1:8000/docs>.

## Start the Frontend

Open a second terminal at the repository root:

```powershell
cd frontend
npm.cmd install
npm.cmd run dev
```

Vite serves the application at <http://localhost:5173> by default.

Copy `frontend/.env.example` to `frontend/.env` only when the backend uses a different URL:

```env
VITE_API_BASE_URL=http://127.0.0.1:8000
```

## API Endpoints

- `GET /` — API identity and running status
- `GET /health` — service health used by the frontend

## Day 2 Image Workspace

- Accepts JPG/JPEG, PNG, and WEBP images up to 10 MB.
- Keeps selected images in temporary browser memory; images are not uploaded or persisted.
- Shows a responsive preview plus filename, resolution, size, and file type.
- Supports changing, removing, and repeatedly selecting images.
- Shows a clear readiness message when **Analyze Privacy** is selected; it does not simulate AI output.
- Provides disabled future privacy controls and honest empty result states for Day 3 integration.

Run the frontend validation tests with:

```powershell
cd frontend
npm.cmd test
```

## 20-Day Development Approach

Development is incremental: establish and verify the foundation first, then add one testable privacy capability at a time. Later stages will cover detection, context awareness, anonymization, OCR, privacy scoring, user controls, integration, evaluation, and final documentation. Large AI dependencies remain intentionally excluded from Day 2.
