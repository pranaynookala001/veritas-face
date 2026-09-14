# API contract — draft v1

The API exposes an asynchronous job resource. This document is intentionally implementation-neutral so the Next.js app, Python worker, and C++ inference service agree before the transport code is added.

## `POST /v1/jobs`

Accepts one JPEG, PNG, or WebP image. The response is `202 Accepted` with:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "expires_at": "2026-09-13T23:00:00Z"
}
```

## `GET /v1/jobs/{job_id}`

Returns one of `queued`, `processing`, `completed`, `failed`, or `expired`. A completed job includes a `Report` document.

## Report invariants

- `verdict` is one of `likely_synthetic`, `likely_authentic`, or `inconclusive`.
- `inconclusive` includes at least one quality or evidence reason.
- Detector and provenance evidence retain source/version/status details.
- No raw image, facial embedding, or original filename appears in a report.

## C++ inference service

`POST /v1/infer` accepts a normalized RGB face crop and returns a normalized score, model identifier, model version, and duration. It does not determine the final verdict; Python's calibrated report builder owns that decision.
