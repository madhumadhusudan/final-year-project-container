# Local model requirements

Model weights are intentionally excluded from Git unless their provenance and redistribution terms are verified.

## License plates

Set `LICENSE_PLATE_MODEL_PATH` to a local Ultralytics-compatible model trained for an explicit `license_plate` or `licence_plate` class. Set `LICENSE_PLATE_MODEL_SOURCE` to its source URL or package identifier. The application rejects models without one of those labels; COCO vehicle boxes are never treated as plates.

## Payment cards

Set `CARD_MODEL_PATH` to a local Ultralytics-compatible model trained for an explicit `card`, `credit_card`, `debit_card`, or `payment_card` class. Set `CARD_MODEL_SOURCE` to its source URL or package identifier. The application rejects other class sets and does not use generic rectangle detection.

Review the source model's license before local use or redistribution. User images are processed locally and are never sent to hosted inference services.
