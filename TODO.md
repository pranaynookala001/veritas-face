# Veritas Face implementation backlog

## Milestone 0 — project foundation

- [x] Define public product contract, architecture, safety boundary, and autonomous-run rules. Verified with root quality gate.
- [x] Add shared API-domain primitives and their unit tests. Verified with `python3 -m unittest discover -s services/api/tests`.
- [x] Add application and C++ service scaffolds plus CI configuration. Verified with root quality gate.

## Milestone 1 — local upload-to-report vertical slice

- [x] Initialize the Next.js application and implement the accessible upload form with client-side file validation. Verified with `npm run check`.
- [x] Add FastAPI health endpoint, request schemas, validated upload intake, and structured error responses. Verified with `python3 -m unittest discover -s services/api/tests` and `npm run check`.
- [x] Add temporary job/artifact storage with retention cleanup and a job-status state machine. Verified with 17 API tests and the root quality gate.
- [x] Implement face detection, primary-face selection, face-quality gates, and deterministic test fixtures. Verified with 24 API tests and the root quality gate.
- [x] Render a completed mock evidence report end-to-end from uploaded image metadata and quality findings. Verified with 30 API tests, 8 web tests, and the root quality gate.

## Milestone 2 — forensic and inference evidence

- [x] Integrate C2PA/Content Credentials verification and EXIF/XMP extraction behind a normalized provenance adapter. Verified with 32 API tests, 8 web tests, and the root quality gate.
- [x] Define the ONNX inference HTTP contract and implement C++ service health/model-info endpoints. Verified with the C++ HTTP contract test and root quality gate.
- [x] Add C++ image preprocessing and CPU ONNX inference with Python/C++ parity tests. Verified with the two C++ HTTP tests, including a generated ONNX fixture compared against Python ONNX Runtime, and the root quality gate.
- [x] Add baseline-detector adapter, score validation, latency capture, and model-version reporting. Verified with 38 API tests, 8 web tests, two C++ inference contract tests, and the root quality gate.
- [x] Implement calibration, detector-disagreement policy, and explainable final-verdict generation. Verified with 47 API tests, 8 web tests, two C++ inference contract tests, and the root quality gate.

## Milestone 3 — training and evaluation

- [x] Curate licensed real and fully synthetic portrait dataset manifests with generator-family-aware train/validation/test splits. Verified with 4 training-manifest tests, `python3 training/validate_manifests.py`, and the root quality gate.
- [x] Add reproducible free-Kaggle fine-tuning notebook for a pretrained lightweight backbone. Verified with 14 training tests, JSON validation, and the root quality gate.
- [x] Export the selected model to ONNX with a model card, license, checksums, and reproducible threshold configuration. Verified with 19 training tests, including release provenance and policy-artifact validation, and the root quality gate.
- [ ] Build benchmark runner for held-out sources and transformations: JPEG, resize, crop, filters, profile pose, blur, and occlusion.
- [ ] Publish metrics, calibration plots, error analysis, model limitations, and latency comparison against the baseline.

## Milestone 4 — hardening and delivery

- [ ] Add Docker Compose for web, API/worker, inference, Redis, Postgres, and temporary object storage.
- [ ] Add rate/size limits, security headers, structured logs, health/readiness checks, and end-to-end tests.
- [ ] Deploy the public CPU demo within free-tier limits; document cold-start and quota behavior.
- [ ] Complete the portfolio README, architecture diagram, setup guide, demo screenshots/video, and pull request to `main`.
