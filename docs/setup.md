# Local setup guide

Veritas Face is currently a local-development project. It has no public demo
URL and this guide does not configure a public exposure. Keep the Compose
network private and never place portrait files, face crops, model weights,
calibration artifacts, or real credentials in Git.

## Prerequisites

- Docker Desktop for the full local stack.
- Node.js 20 or newer and npm.
- Python 3.11 or newer.
- CMake 3.25 or newer to run the C++ inference contract checks.

The repository intentionally excludes ONNX weights and calibration artifacts.
You can run the safe upload-to-report path without them; it returns
`inconclusive` rather than an unvalidated model claim.

## Full local stack with Docker Compose

From the repository root:

```sh
cp .env.example .env
docker compose up --build
```

Open <http://localhost:3000>. Confirm the API can accept work at
<http://localhost:8000/readyz>. The initial build can take time because it
installs development dependencies and prepares the CPU inference container.

The default `.env` values are local-only development values. In particular:

- only web (`3000`) and API (`8000`) ports are published;
- Redis, Postgres, MinIO, and inference remain private to the Compose network;
- the inference service is live but model-unavailable until a private model and
  version are mounted; and
- the API accepts 30 attempts per direct client per 60 seconds by default.

Stop the stack and remove local volumes when you are finished:

```sh
docker compose down -v
```

See [docker-compose.md](docker-compose.md) before changing port, retention,
rate-limit, or private model/calibration settings.

## Web and API only, without Docker

This path is useful for frontend and API work. It does not start C++ inference,
so detector evidence stays unavailable by design.

Install dependencies once from the repository root:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e "services/api[test]"
npm ci
```

Start the API in one terminal:

```sh
. .venv/bin/activate
python -m uvicorn app.main:app --app-dir services/api --reload --port 8000
```

Start the web app in another terminal:

```sh
NEXT_PUBLIC_API_BASE_URL=http://localhost:8000 npm run dev --workspace=@veritas-face/web
```

Open <http://localhost:3000>. The API uses an OS-managed temporary directory
when `VERITAS_FACE_ARTIFACT_DIRECTORY` is unset. It still applies the 24-hour
maximum retention policy, but do not treat a development machine as a secure
multi-user service.

## Optional audited detector configuration

Only an operator with an audited private release bundle should configure
detector-backed verdicts. Mount or place an ONNX model outside Git, give it a
release version, and supply a calibration JSON created for that exact detector
release. For Compose, the relevant local-only settings are:

```dotenv
VERITAS_INFERENCE_MODEL=/models/portrait-classifier.onnx
VERITAS_INFERENCE_MODEL_VERSION=release-YYYY.MM.N
VERITAS_INFERENCE_MODEL_ID=synthetic-portrait-classifier
VERITAS_FACE_CALIBRATION_PATH=/calibration/release-YYYY.MM.N.json
```

`VERITAS_FACE_INFERENCE_URL` points the API at the C++ service. A missing,
unreadable, invalid, or mismatched calibration input always keeps the result
`inconclusive`; do not relax that rule to make a demo look decisive. Consult
the [model-release procedure](model-release.md) and [API contract](api-contract.md)
before supplying these values.

## Verify the checkout

Run the same root quality gate used by CI after installing its dependencies:

```sh
python3 -m pip install -e "services/api[test]"
python3 -m pip install -r services/inference/test-requirements.txt
npm ci
npm run check
```

It validates the manifests and Compose file, runs API/training/C++ contract
tests, then tests, lints, and production-builds the web app. The CI workflow is
in [`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

## Troubleshooting

- If `readyz` returns `503`, inspect the API process or Compose health status;
  the artifact directory must be owner-only and writable.
- If the browser cannot reach the API after changing the web port, update
  `VERITAS_FACE_WEB_ORIGINS` to the exact browser origin. If the API base URL
  changes in Compose, rebuild the web image because the public value is bundled
  at build time.
- A report that says the detector is unavailable or calibration is not
  configured is the expected default checkout behavior, not a reason to infer
  anything about the uploaded image.
- `429 rate_limited` means the direct-client local limit was reached. Wait for
  the response's `Retry-After` interval; do not disable limits for a public
  deployment.
