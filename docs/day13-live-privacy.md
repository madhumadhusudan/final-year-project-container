# Day 13 — Live Privacy Validation Notes

## Implemented

- `/live` route with explicit `Start Camera`; the constraint is `video` only and `audio: false`.
- Camera states: idle, requesting permission, active, permission denied, unavailable,
  error, and stopping.
- Stop, switch, and unmount cleanup abort analysis, cancel timers/animation frames,
  call `stop()` on every media track, clear `srcObject`, clear transient tracks, and
  release working canvases.
- Protected canvas is the only visible preview. Before the first successful face
  result, and whenever the backend is unavailable, the whole frame receives a
  safety blur and the UI does not claim selective protection is active.
- Analysis uses memory-only JPEG blobs at 480, 640, or 768 px width. The backend
  returns coordinates and safe classification metadata, never modified frame data.
- One request can be active at a time. Every request carries a monotonically
  increasing frame ID and capture time; non-new response IDs are ignored.
- Faces are requested on the fastest cadence. At most one due heavy group is added
  to a request. Balanced targets: faces 180 ms, codes 1000 ms, plates 1200 ms,
  cards/OCR 1800 ms, documents 2000 ms.
- Temporary tracks use category, IoU, and center distance. Bounding boxes use an
  exponential smoothing alpha of 0.68. A missed refreshed region remains for 550
  ms; slower categories have cadence-aware expiry windows.
- Backend main-subject geometry seeds a stable face track. It does not switch while
  that track remains reliable. The user may select another tracked face manually.
- Blur, pixelation, and blackout are rendered locally with category-specific padding.
- Actual render timestamps, completed analyses, and measured round-trip latency feed
  the performance panel. The face interval adapts between its quality-mode target
  and 600 ms.

## Automated verification

- Backend: 85 tests passed, including 6 Day 13 live API tests.
- Frontend: 16 tests passed, including Day 13 coordinate, tracking, grace,
  scheduling, stale-response, stream cleanup, risk, and toggle tests.
- Vite production build passed.
- `/live` returned through the Vite SPA fallback and loaded live capabilities from
  the local FastAPI server.

## Manual device checks still required

Automated/headless execution has no physical webcam. Validate Allow/Deny, hardware
indicator shutdown, navigation cleanup, front/back switching, 640×480 and 1280×720,
two-person motion, printed synthetic plate/QR/document/text samples, and mobile
orientation on the presentation device. Remote mobile camera access may require
HTTPS. Do not record or commit frames from real people or real identity/payment data.

At development time, installed capabilities were: face, payment card, QR, barcode,
and sensitive text available; license plate and identity document unavailable.
