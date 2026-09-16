# API contract — v1 intake

The API exposes an asynchronous job resource. Its first implemented boundary validates image content before any artifact storage or analysis begins. A content type from the browser is only a hint: the API verifies the encoded image independently.

## `GET /health`

Returns API liveness and the deployed service version:

```json
{
  "status": "ok",
  "service": "veritas-face-api",
  "version": "0.1.0"
}
```

`GET /healthz` is an equivalent liveness alias for container platforms.

## `POST /v1/jobs`

Accepts one JPEG, PNG, or WebP image. The response is `202 Accepted` with:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "expires_at": "2026-09-13T23:00:00Z"
}
```

Submit the file as `multipart/form-data` with a `portrait` field. Images must be non-empty, at most 10 MiB, and decode as JPEG, PNG, or WebP. When a specific image MIME type is declared, it must match the decoded content; generic `application/octet-stream` is treated as unknown and decoded normally. The receipt deliberately contains neither an original filename nor image bytes. This intake step validates the upload but does not yet expose persistent job status; temporary artifact storage and `GET /v1/jobs/{job_id}` follow in the next milestone.

## Error responses

Every documented client or routing error uses the same envelope:

```json
{
  "error": {
    "code": "invalid_image",
    "message": "The upload is not a valid JPEG, PNG, or WebP image.",
    "issues": [
      {
        "field": "portrait",
        "message": "The upload is not a valid JPEG, PNG, or WebP image.",
        "type": "invalid_image"
      }
    ]
  }
}
```

Error codes include `empty_upload` (400), `upload_too_large` (413), `unsupported_media_type` or `media_type_mismatch` (415), and `invalid_image`, `image_too_large`, `animated_image_not_supported`, or `request_validation_failed` (422). These errors establish file eligibility only; no authenticity claim is made at intake.

## Planned `GET /v1/jobs/{job_id}`

Returns one of `queued`, `processing`, `completed`, `failed`, or `expired`. A completed job includes a `Report` document.

## Report invariants

- `verdict` is one of `likely_synthetic`, `likely_authentic`, or `inconclusive`.
- `inconclusive` includes at least one quality or evidence reason.
- Detector and provenance evidence retain source/version/status details.
- No raw image, facial embedding, or original filename appears in a report.

## C++ inference service

`POST /v1/infer` accepts a normalized RGB face crop and returns a normalized score, model identifier, model version, and duration. It does not determine the final verdict; Python's calibrated report builder owns that decision.
