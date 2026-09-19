# API contract — v1 local evidence report

The API exposes an asynchronous job resource. Its implemented intake boundary validates image content before private, temporary artifact storage or analysis begins. A content type from the browser is only a hint: the API verifies the encoded image independently. The local development worker then assembles an evidence report from safe decoded-image metadata facts, offline C2PA/Content Credentials verification, and face-quality findings.

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

Submit the file as `multipart/form-data` with a `portrait` field. Images must be non-empty, at most 10 MiB, and decode as JPEG, PNG, or WebP. When a specific image MIME type is declared, it must match the decoded content; generic `application/octet-stream` is treated as unknown and decoded normally. The receipt deliberately contains neither an original filename nor image bytes.

The local worker artifact is stored under a generated UUID, not the client filename, with owner-only permissions. It is retained for at most 24 hours. Every job create, status lookup, and worker artifact read performs retention cleanup; expired artifacts and completed reports are deleted at expiry, leaving only non-sensitive `expired` job metadata.

## `GET /v1/jobs/{job_id}`

Returns `200 OK` with the job's non-sensitive state. While a report is being prepared, only these fields are present:

```json
{
  "job_id": "uuid",
  "status": "queued",
  "expires_at": "2026-09-13T23:00:00Z"
}
```

Once the local worker completes, the same resource also includes a report:

```json
{
  "job_id": "uuid",
  "status": "completed",
  "expires_at": "2026-09-13T23:00:00Z",
  "report": {
    "report_version": "local-evidence-v2",
    "calibration_version": "not_calibrated_v1",
    "verdict": "inconclusive",
    "confidence": null,
    "reasons": ["no_face_detected"],
    "evidence": [
      {
        "source": "image_metadata",
        "status": "observed",
        "detail": "Decoded PNG image, 800 × 800 pixels; EXIF metadata is not present; XMP metadata is not present; no embedded metadata is present. Metadata presence or absence is not an authenticity signal.",
        "version": "provenance-adapter-1.0"
      },
      {
        "source": "c2pa_content_credentials",
        "status": "not_present",
        "detail": "No embedded C2PA/Content Credentials manifest was found. Missing provenance is not evidence that this portrait is authentic or synthetic.",
        "version": "c2pa-python-0.37.10"
      },
      {
        "source": "face_quality",
        "status": "inconclusive",
        "detail": "Face-quality gates require an inconclusive result: no_face_detected.",
        "version": "1.0"
      }
    ],
    "model_versions": {
      "c2pa": "c2pa-python-0.37.10",
      "face_quality": "1.0",
      "provenance_adapter": "1.0"
    }
  }
}
```

The state machine permits `queued` → `processing` → `completed` and either active state → `failed`. After retention cleanup, any state becomes `expired`; `expired`, `completed`, and `failed` are terminal. This endpoint never returns an uploaded image, client filename, private artifact location, or embedded metadata values. Unknown identifiers receive the standard `404 job_not_found` error envelope.

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

## Report invariants

- `verdict` is one of `likely_synthetic`, `likely_authentic`, or `inconclusive`.
- `inconclusive` includes at least one quality or evidence reason.
- Detector and provenance evidence retain source/version/status details.
- No raw image, facial embedding, original filename, embedded metadata value, C2PA manifest content, or signer identity appears in a report.
- Every report includes report and calibration version identifiers. The current local evidence report uses `not_calibrated_v1` because it has no detector score to calibrate.

## Local browser connection

The web app submits to `http://localhost:8000` by default and polls its job resource until `report` is available. Set `NEXT_PUBLIC_API_BASE_URL` to use another API URL. The API permits browser requests from `http://localhost:3000` by default; set `VERITAS_FACE_WEB_ORIGINS` to a comma-separated allow-list for another local origin.

The local worker extracts presence-only EXIF/XMP facts and verifies embedded C2PA/Content Credentials with `c2pa-python`. Remote C2PA manifest fetch is disabled, so the worker does not retrieve external manifest stores for uploaded portraits. A C2PA result is normalized to `verified`, `invalid`, `not_present`, or `unavailable`; only `verified` means the embedded credential validated, and it verifies declared provenance rather than portrait authenticity. The report always returns `inconclusive` until a synthetic-portrait detector score is available. In particular, missing, invalid, or unavailable provenance and metadata presence or absence must never be interpreted as origin evidence.

## C++ inference service

`POST /v1/infer` accepts a normalized RGB face crop and returns a normalized score, model identifier, model version, and duration. It does not determine the final verdict; Python's calibrated report builder owns that decision.
