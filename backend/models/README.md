# Local model requirements

Model weights are intentionally excluded from Git unless their provenance and redistribution terms are verified.

## License plates

Set `LICENSE_PLATE_MODEL_PATH` to a local Ultralytics-compatible model trained for an explicit `license_plate` or `licence_plate` class. Set `LICENSE_PLATE_MODEL_SOURCE` to its source URL or package identifier. The application rejects models without one of those labels; COCO vehicle boxes are never treated as plates.

## Payment cards

The verified local checkpoint belongs at `backend/models/card_detector.pt`. Its source, Apache-2.0 license, SHA-256 digest, and exact trained labels are recorded in `card_detector.manifest.json`. The application accepts explicit payment-card labels (including this model's `creditCardFront` and `creditCardBack`) and does not rename COCO or generic rectangle classes into cards.

The detector uses 960px full-image inference. Large images use overlapping 640px crops at a native 640px inference size so small cards retain enough pixels without increasing false positives. `CARD_CONFIDENCE_THRESHOLD`, `CARD_INFERENCE_IMAGE_SIZE`, `CARD_TILE_SIZE`, `CARD_TILE_INFERENCE_IMAGE_SIZE`, and `CARD_TILE_OVERLAP` are independently configurable.

Review the source model's license before local use or redistribution. User images are processed locally and are never sent to hosted inference services.

## Identity documents

Place an optional Ultralytics-compatible checkpoint at `backend/models/document_detector.pt`, or set `DOCUMENT_MODEL_PATH`. The detector only accepts model-native labels explicitly associated with identity documents, such as Aadhaar, PAN, passport, driving-licence, identity-card, or document labels. COCO classes and arbitrary rectangles are rejected and are never renamed into identity documents.

Record the checkpoint's origin and usage terms with `DOCUMENT_MODEL_SOURCE` and `DOCUMENT_MODEL_LICENSE`. `DOCUMENT_CONFIDENCE_THRESHOLD` defaults to `0.35`, and `DOCUMENT_INFERENCE_IMAGE_SIZE` defaults to `960`. Without a compatible local checkpoint, the API deliberately reports `document_detection.status = "unavailable"` and the privacy-risk assessment remains partial.
