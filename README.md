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

The initial web experience lets a person select one JPEG, PNG, or WebP portrait up to 10 MiB and reports client-side eligibility feedback accessibly. This is a convenience check only: the API will independently validate uploaded content before it is retained or analyzed, and no authenticity verdict is made in the browser.

## Local prerequisites

- Node.js 20+
- Python 3.11+
- CMake 3.25+
- Docker Desktop (for the complete local stack)

## Safety and scope

V1 only evaluates still images with one dominant human face. It does **not** claim to detect face swaps, identity deepfakes, or every kind of AI edit. Read [the project guardrails](docs/guardrails.md) before using the result for any consequential decision.
