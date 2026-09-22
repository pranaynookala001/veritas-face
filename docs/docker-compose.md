# Local Docker Compose stack

`compose.yaml` starts the complete local development topology: the Next.js web
app, FastAPI process and in-process worker, CPU inference service, Redis,
Postgres, and a private-network MinIO object-store service. It is for local
development only; it does not expose Redis, Postgres, MinIO, or the inference
port on the host.

The current v1 job store intentionally remains a single-process implementation.
The API's background task is the worker, and accepted image bytes are stored in
the owner-only `temporary-artifacts` volume for no more than the existing
24-hour retention window. Redis, Postgres, and MinIO are included to make the
local topology match the next hardening work, but v1 does not yet persist jobs
to Redis or Postgres or send uploaded portraits to MinIO. Do not mistake the
presence of those containers for distributed job durability.

## Start the stack

Install Docker Desktop, then copy the development defaults and start the
services:

```sh
cp .env.example .env
docker compose up --build
```

Open `http://localhost:3000`. The API is available at
`http://localhost:8000/healthz`; it waits for its local Postgres and Redis
dependencies before becoming healthy. Stop and remove all local state with:

```sh
docker compose down -v
```

If `WEB_PORT` changes from `3000`, also set `VERITAS_FACE_WEB_ORIGINS` to the
matching browser origin. `NEXT_PUBLIC_API_BASE_URL` is compiled into the web
bundle, so changing it requires `docker compose build web` before restarting.

## Model and calibration inputs

The repository has no model weights or calibration artifacts. Keep private
release files outside Git: `./models` accepts a read-only ONNX mount and
`./calibration` is ignored for a private calibration JSON. Configure all of
the following only for an audited release:

```dotenv
VERITAS_INFERENCE_MODEL=/models/portrait-classifier.onnx
VERITAS_INFERENCE_MODEL_VERSION=release-YYYY.MM.N
VERITAS_INFERENCE_MODEL_ID=synthetic-portrait-classifier
VERITAS_FACE_CALIBRATION_PATH=/calibration/release-YYYY.MM.N.json
```

Without a model, the inference container remains healthy but deliberately
reports its model as unavailable. Without a valid exact-release calibration
artifact, a detector score cannot determine a verdict. In either case the API
returns an evidence report with an `inconclusive` result rather than treating
missing configuration or provenance as evidence of authenticity.

## Temporary-data boundary

- The API volume stores generated-ID upload artifacts only. It has no client
  filenames, is owner-only inside the API container, and cleanup removes the
  bytes and report at expiry.
- The MinIO `/data` directory is a container `tmpfs`, so its contents disappear
  when the object-store container stops. It is not part of the current v1
  upload path.
- Redis disables snapshots and append-only persistence. Postgres uses a named
  local volume only for local infrastructure testing; remove it with
  `docker compose down -v`.

Never publish the host ports, reuse the example development passwords outside
your machine, or mount raw portraits, face crops, prompts, seeds, or model
checkpoints from a shared directory.
