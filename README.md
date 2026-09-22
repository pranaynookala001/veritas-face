# Veritas Face

**Synthetic Portrait Authenticity Analyzer** — an evidence-first system for assessing whether a still portrait is likely camera-origin or fully AI-generated.

> This project reports risk and evidence, not certainty. Only validated content provenance can verify origin; model scores are probabilistic.

## What it will do

1. Accept a JPEG, PNG, or WebP portrait.
2. Detect and quality-check one dominant face.
3. Inspect image metadata and C2PA/Content Credentials.
4. Run a fine-tuned portrait classifier and an independent baseline detector.
5. Calibrate the scores into `likely_synthetic`, `likely_authentic`, or `inconclusive`.
6. Return an explainable report with evidence, quality warnings, and model versions.

## Architecture

```text
Next.js web app
       |
       v
FastAPI job API + worker  --->  provenance / metadata inspection
       |                                   |
       v                                   v
C++ ONNX Runtime service  <---  aligned primary-face crop
       |
       v
calibration + evidence report
```

The browser never receives an authenticity claim without its supporting evidence. Original uploads and derived crops are temporary and are deleted after processing.

## Repository layout

- `apps/web` — public Next.js upload and report experience.
- `services/api` — Python domain model, validation, job orchestration, and report assembly.
- `services/inference` — C++/CMake ONNX Runtime inference service.
- `training` — reproducible Kaggle fine-tuning and evaluation assets.
- `docs` — API, architecture, benchmark, privacy, and operational documentation.

## Development status

See [TODO.md](TODO.md) for the ordered implementation backlog. The first milestone establishes shared contracts and automated quality gates; later milestones add actual upload, inference, and deployment behavior.

## Current local workflow

The local web experience lets a person select one JPEG, PNG, or WebP portrait up to 10 MiB, submit it to the API, and view a completed evidence report. The API independently validates multipart `portrait` content, stores accepted bytes privately for at most 24 hours, then runs local face-quality gates and offline provenance checks in the background. `GET /v1/jobs/{job_id}` exposes a report only after processing completes; it contains safe decoded-image facts (format, dimensions, and EXIF/XMP presence), a normalized C2PA/Content Credentials verification result, quality findings, and—when explicitly configured—a validated baseline-detector probability with its latency and model version. It never exposes image bytes, embedded metadata values, manifest contents, signer identities, a client filename, or a facial embedding. Remote C2PA manifest retrieval is disabled for uploads. Set `VERITAS_FACE_INFERENCE_URL=http://127.0.0.1:8080` to send quality-passing primary-face crops to the local C++ service. A detector score can affect a verdict only when `VERITAS_FACE_CALIBRATION_PATH` names a versioned held-out calibration artifact for that exact detector release; otherwise the report stays `inconclusive`. The policy is probabilistic, treats meaningful disagreement between independent detector families as `inconclusive`, and never treats missing, invalid, or unavailable provenance or metadata as evidence of authenticity or synthetic origin. See [the API contract](docs/api-contract.md) and [face-quality policy](docs/face-quality.md).

The C++ inference service provides local liveness (`/health`, `/healthz`), model-readiness (`/v1/model-info`), and CPU ONNX inference (`POST /v1/infer`) for an explicitly configured model. It accepts a fixed primary-face crop, converts RGB/HWC uint8 bytes to normalized float32 NCHW input, and reports the model score as probabilistic detector evidence. A process without `--model` remains deliberately `unavailable`; a healthy process is not an authenticity result. Its HTTP contract and local launch options are documented in [the API contract](docs/api-contract.md#c-inference-service).

## Training-data governance

The repository contains a reviewed, licence-aware source catalog and a generator-family-disjoint split plan, not portrait data or model weights. Camera-origin records require per-file licence and attribution verification; fully synthetic records must be text-to-image-only and retain a pinned generator revision plus a private output hash. Raw images, face crops, prompts, seeds, materialized manifests, and checkpoints stay out of Git. See [training-data governance](docs/training-data.md) and [the manifest guide](training/manifests/README.md).

The first training path is an unexecuted, reproducible [free-Kaggle MobileNetV3-Small fine-tuning notebook](training/fine_tune_mobilenetv3_kaggle.ipynb). It validates a private, digest-pinned record manifest and every referenced image before training, captures deterministic run metadata, and leaves the held-out test split untouched. The private [model-release procedure](docs/model-release.md) exports a selected checkpoint to a strictly contracted ONNX bundle with an explicit operator-supplied licence, model card, calibration linkage, verdict-policy configuration, and checksums. It produces no checked-in model or performance claim; benchmarking and publication remain later milestones.

## Local prerequisites

- Node.js 20+
- Python 3.11+
- CMake 3.25+
- Docker Desktop (for the complete local stack)

## Safety and scope

V1 only evaluates still images with one dominant human face. It does **not** claim to detect face swaps, identity deepfakes, or every kind of AI edit. Read [the project guardrails](docs/guardrails.md) before using the result for any consequential decision.
