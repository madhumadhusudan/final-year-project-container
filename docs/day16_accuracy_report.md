# Day 16 Accuracy Validation and Reliability Report

Date: 2026-09-07
Scope: validation and hardening of the existing Day 1–15 local privacy pipeline. No new detector category was added.

## Executive result

Day 16 produced a repeatable synthetic accuracy suite, fixed evidence-backed regressions, and passed the complete automated regression (114 backend tests, 19 frontend tests, and the production build). Detection and protection were both exercised with real local inference.

Day 16 is **not fully complete** under the strict definition of done. Dedicated license-plate and identity-document weights are not installed, the existing image workflow has no manual add/remove-region review feature to regress, and physical-camera FPS/flicker cannot be certified in this headless environment. These limitations are reported as unavailable/not tested, never as zero-risk or fabricated accuracy.

## Data policy and measurement method

- Only deterministic dummy renderings, safe generated QR/EAN-13 codes, and two fictional AI-generated people fixtures were used.
- No real identity number, payment card, document, payment QR, password, address, or personal photograph was used.
- Ground-truth face boxes were independently hand annotated. Matching uses one-to-one greedy IoU (0.45 for faces; 0.25 for mixed-scene privacy items whose boxes may exclude quiet zones).
- Precision, recall, and F1 are emitted only when a real ground-truth denominator exists. `null`/N/A means unavailable or not measurable.
- QR and barcode detectors expose no confidence; their confidence remains `null`.
- Latencies are observations on the current CPU environment, not guarantees.

## Detector summary

| Detector | Ground-truth cases | TP | FP | FN | Precision | Recall | F1 | Average latency | Chosen threshold | Result / limitation |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| Faces (YuNet) | 5 labeled faces + negative scene | 5 | 0 | 0 | 1.000 | 1.000 | 1.000 | 723.52 ms | 0.75 | Mean IoU 0.7245; limited synthetic set |
| Main subject | 1/2/3-person and ambiguous scenarios | N/A | N/A | N/A | N/A | N/A | N/A | <1 ms logic-only | score rules | All formal scenarios passed |
| Background faces | 3-person frame + tracking cases | 2 observed | 0 observed | 0 observed | qualitative | qualitative | qualitative | included above | 0.75 | Stable during small movement |
| License plates | Contract/protection only | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.35 configured | Dedicated weights absent; `unavailable` |
| Payment cards | 7 positive variants + 7 negative types | 4 | 0 | 3 | 1.000 | 0.5714 | 0.7273 | 7,398.09 ms | 0.25 | Rotation/occlusion/clutter misses |
| Identity documents | Classifier/contract/protection only | N/A | N/A | N/A | N/A | N/A | N/A | N/A | 0.35 configured | Dedicated weights absent; `unavailable` |
| Sensitive OCR categories | 7 positive + 3 normal lines | 7 | 0 | 0 | 1.000 | 1.000 | 1.000 | 43,326.45 ms positive image | OCR + patterns | Accurate on this rendered set; slow |
| QR codes | 6 payload classes | 6 | 0 | 0 | 1.000 | 1.000 | 1.000 | 204.66 ms | N/A (`null`) | All five rotations decoded |
| EAN-13 barcode | 1 generated code | 1 | 0 | 0 | 1.000 | 1.000 | 1.000 | 226.90 ms | N/A (`null`) | Real format/decode/box verified |

## Face and main-subject results

Face positives covered single, multiple, small/background, large/foreground, side profile, partial facial occlusion, low light, glasses, near-edge placement, and 15-degree rotation. Negative controls covered poster-like content, cartoon-stylized shapes, grayscale statue-like shapes, and round face-like objects.

YuNet found all five independently labeled faces at 0.75. At 0.20 it produced one extra false positive (precision 0.8333, recall 1.0, F1 0.9091). Thresholds 0.25, 0.30, 0.40, 0.50, and 0.75 all produced 5 TP / 0 FP / 0 FN, so 0.75 was retained instead of lowering it on a small corpus.

Main-subject tests covered one, two, and three people, off-centre prominence, a larger off-centre person, person-box support, partial support, and ambiguity. Clear cases selected the intended prominent subject and marked others `background_face`; similar candidates returned `uncertain` and made every face protectable.

Actual live-frame analysis selected the central subject and two background faces. An 8-pixel movement retained that decision (166 ms then 143 ms). Brief main-face occlusion returned `uncertain` in 148 ms and made remaining faces protectable rather than claiming protection was complete.

## License-plate results

The suite includes safe generated plate material plus existing contract, clipping, vehicle-association, OCR-context, risk, and protection tests. Image-level accuracy cannot be measured because `models/license_plate_detector.pt` is absent. Large/small/angled/low-light/partial/multiple/edge and signboard/sticker/book/random-label detector results are therefore not claimed. The API reports `unavailable`, and assessment remains `partial`.

The dummy plate protection reread covered high-strength blur, pixelate, and blackout in one combined OCR pass. The original dummy characters were not recovered.

## Payment-card results

| Positive case | Detected | Missed | Confidence |
|---|---:|---:|---:|
| Close-up | 1 | 0 | 0.9660 |
| Small | 1 | 0 | 0.9699 |
| Rotated | 0 | 1 | N/A |
| Partial | 1 | 0 | 0.9741 |
| Simulated hand occlusion | 0 | 1 | N/A |
| On table | 1 | 0 | 0.5137 |
| Cluttered/rotated | 0 | 1 | N/A |

Lowering 0.25 to 0.20 returned the same four detections and no recovery. Thresholds through 0.50 retain the four observed positives, including the lowest at 0.5137. The 0.25 default remains because lowering gives no measured recall gain and would accept weaker candidates.

Distinct negatives covered phone, wallet, synthetic ID card, business card, notebook, plain paper, and remote control. The first run found a 0.6038 remote false positive. A general geometry guard now rejects candidates above 2.35:1; all seven negatives pass and the true card still passes. The difficult scene still has one document-as-card false positive (1 TP / 1 FP). It is conservative over-protection but wrong categorization.

The dummy card number was not recovered by OCR from the high-strength blur/pixelate/blackout mosaic.

## Identity-document results

Safe Aadhaar-like, PAN-like, passport-like, driving-licence-like, and generic-ID classifier contracts are covered. Book, certificate/resume-like text, business card, payment card, plain paper, and phone/rectangle labels remain excluded by the detector contract. Image precision/recall is N/A because the checkpoint is absent.

Whole-document precedence, contained OCR/QR grouping, all three protection methods, residual risk, and document-photo face handling passed. The dummy document identifier was not recovered from the protection mosaic.

## OCR results

The positive image contained a dummy phone, email, PAN-like identifier, Aadhaar-like digits, address, PIN code, and Luhn-valid test-card number. EasyOCR returned seven lines and the classifier returned all seven categories after a narrow fix for the observed phone `O`/`0` confusion. That repair activates only with an explicit numeric label and at least five digits; text is not globally rewritten.

`Welcome to VVCE`, `Computer Science Project`, and `Privacy Protection Demo` returned zero sensitive classifications. PIN matching now precedes generic address wording so the dummy PIN is categorized specifically.

- Sensitive image after blackout: 0 OCR lines and 0 sensitive items.
- Protected plate/card/document mosaic: original identifiers readable = false across high-strength blur, pixelate, and blackout.
- Latencies: positive 43,326.45 ms; normal negative 13,871.14 ms; blackout reread 41,742.62 ms; combined protected-object reread 72,772.67 ms.

Small, rotated, blurred, low-contrast, document, and card text are represented by generated transformations and context tests. This small corpus is not a real-world OCR accuracy claim.

## QR and barcode results

Locally generated URL, dummy payment, dummy contact, dummy Wi-Fi, plain text, and unknown QR payloads all detected and decoded. Payloads are classified/masked and never navigated, executed, connected, or transmitted. Confidence is `null`.

Rotations at 0, 15, 30, 45, and 90 degrees each detected/decoded 1/1. A small QR was recovered with the bounded retry. A partial QR may remain detected-but-undecodable and is never marked safe. QR/document grouping and masked previews pass.

OpenCV advertises EAN-8, EAN-13, UPC-A, and UPC-E here, not CODE128. Generated EAN-13 genuinely detected/decoded with its real format and box. Standalone retail EAN-13 remains LOW privacy significance; identity-document-associated code risk is higher.

| Successful decodes after protection | Blur | Pixelate | Blackout |
|---|---:|---:|---:|
| QR | 0 | 0 | 0 |
| EAN-13 | 0 | 0 | 0 |

A blurred QR outline may still be detectable, but its payload is not decodable. Low strength now has a medium-strength floor specifically for machine-readable QR/barcode regions in image and video output.

## Small-object and mode comparison

The same 1920×1280 difficult scene contains two people, small plate, small card, small document, QR, and sensitive text. One-time lazy initialization was excluded.

| Mode | Latency | Detected regions | Misses | Extra detections |
|---|---:|---:|---:|---:|
| FAST | 14,243 ms | 6 | 3: plate, document, document PIN | 1 card-on-document |
| BALANCED | 12,895 ms | 6 | 3: plate, document, document PIN | 1 card-on-document |
| ACCURACY | 88,634 ms | 7 | 2: plate, document | 1 card-on-document |

FAST originally missed the small QR and external sensitive text. The still-image QR retry now runs for every profile, and FAST OCR increased from 480 to a bounded 640 maximum side. Afterward FAST and BALANCED achieved 2/2 faces, 1/1 QR, 1/1 true card, and 1/1 external sensitive text. Absent plate/document models account for their misses. Accuracy additionally recovered document PIN text but was slower due to full-frame OCR. FAST being slightly slower than BALANCED in one sample is timing variance, not a claimed ordering.

## Critical scene protection

Accuracy analysis returned two faces, two card-class regions (one true card and one document false positive), one QR, and two sensitive-text regions; plate/document modules were unavailable. Main subject was uncertain, so both faces were protected.

Blackout applied six de-duplicated regions: two faces, two card-class regions, one QR, and one text region. Protected PNG encoding/download succeeded. Detected-evidence risk fell from 92 to 9. The assessment remains `partial`; residual 9 does not claim unavailable plate/document risks disappeared.

## Risk, grouping, and review regression

- One background face scores below three; small regions score below large/clear equivalents.
- Generic URL QR scores below payment QR; retail barcode scores below document-associated code.
- Same evidence/settings returns the same risk.
- Successful protection does not increase residual risk; zero protected regions receive no fake reduction.
- Failed/unavailable modules produce `partial`, list unavailable modules, and never imply zero risk or fully protected.
- Document + QR + OCR precedence, duplicate OCR, address/PIN grouping, and region de-duplication pass.

Manual add-region correction, Reviewed Risk, and disabling one AI detection do not exist in the current Day 1–15 image workflow. Those requested regression cases cannot be executed without adding a feature, explicitly forbidden for Day 16. No reviewed-risk claim is made.

## Live-camera reliability

- Real generated frames: central main subject plus two background faces; small movement stable; brief occlusion fails safe to uncertain.
- Face-only local analysis samples were 166, 143, and 148 ms (about 6.5 analyses/s).
- Browser tests pass association, smoothing, grace, stale-response rejection, scheduling, cleanup, and selected-main behavior.
- Fetch timeout/backend failure sets `unavailable`; the UI shows a detection-unavailable warning and full-frame safety blur, never false “Protected.”

Physical Render FPS, camera Analysis FPS, extreme flicker, permission, and hardware switching were not measurable here, so hardware reliability is not certified.

## Video reliability

Synthetic API/tracker tests passed motion association, brief-miss protection, sampled masks, all methods, cancellation cleanup, result download, and false-track expiry.

The real generated two-person acceptance video produced 3.0 s, 36 frames at 12 FPS, and reopened at the same 1536×1024 dimensions/frame count/FPS. It analyzed 12 frames, protected 72 face regions across all 36 output frames, took 20.021 s (1.798 processing FPS), and reduced detected risk 50 → 11. Progress had 39 monotonic updates. Audio was unsupported and honestly reported false; codec was MP4V. Temporary input/output were removed.

## Security regression

Source scan/tests found no cloud image upload, raw OCR/QR payload logging, permanent frame/image storage, face recognition, `shell=True`, `eval`, or `os.system`. FFmpeg remains argument-array based with `shell=False`. Malicious-looking QR text is only classified/masked. Acceptance video artifacts were cleaned.

## Issues and severity

| Severity | Issue | Disposition |
|---|---|---|
| CRITICAL | None found in tested supported paths | N/A |
| HIGH | Balanced/Accuracy QR crash from lazy-wrapper argument mismatch | Fixed; regression passes |
| HIGH | FAST small QR and external-text misses | Fixed; difficult scene rerun passes those items |
| HIGH | Card misses rotated, hand-occluded, and cluttered/rotated cases | Open; 0.20 threshold gave no improvement; needs representative training/augmentation |
| HIGH | Plate/document detector coverage unavailable | Open; provenance-reviewed weights required |
| MEDIUM | Card model labels synthetic ID document as card | Open; needs cross-category hard-negative training |
| MEDIUM | Face false positive at 0.20 | Avoided by retaining 0.75 |
| LOW | Remote-control card false positive | Fixed with general 2.35:1 guard |

## Regression results

```text
Backend:  python -m unittest discover -s tests -v  -> 114 passed in 84.139 s
Frontend: npm test                                -> 19 passed
Frontend: npm run build                           -> passed
Video:    day14_acceptance.py synthetic fixture   -> passed; output removed
```

Covered paths include upload validation, preview helpers, analyze endpoints, supported detectors, context, risk, protection toggles, blur/pixelate/blackout, metadata/download, live-frame APIs/tracking, video upload/process/cancel/result, and cleanup.

## Definition-of-done checklist

- [x] Formal synthetic suite and safe fixtures
- [x] Positive/negative supported-detector tests
- [x] Small face/card/QR and small-text checks
- [x] False positives/negatives documented
- [x] Face/card thresholds calibrated without excessive lowering
- [x] FAST/BALANCED/ACCURACY same-image comparison
- [x] Box/padding/overlap/precedence verification
- [x] Blur/pixelate/blackout exercised
- [x] QR/barcode payload re-decode failure
- [x] OCR/plate/card/document dummy identifier reread failure
- [x] Risk ordering, determinism, partial status, grouping, and residual rules
- [x] Live software fail-safe and real synthetic video regression
- [x] Day 15 optimized scaling/session reuse regression
- [x] CRITICAL supported-path issues fixed (none found)
- [x] HIGH QR/FAST regressions fixed
- [ ] License-plate image accuracy (model unavailable)
- [ ] Identity-document image accuracy (model unavailable)
- [ ] Card HIGH recall gap fully fixed (requires model work)
- [ ] Manual false-negative/false-positive review regression (feature absent)
- [ ] Physical camera FPS/flicker and audio-preserving video test (environment unavailable)

**Final criterion:** No. Not every Day 16 criterion passed; the result remains explicitly partial rather than fabricating coverage or violating the no-new-features constraint.
