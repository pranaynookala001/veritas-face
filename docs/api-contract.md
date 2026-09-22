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
    "report_version": "local-evidence-v4",
    "calibration_version": "not_available",
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
      },
      {
        "source": "baseline_synthetic_detector",
        "status": "skipped",
        "detail": "The baseline detector was not run because face-quality gates require an inconclusive result.",
        "version": "1.0"
      },
      {
        "source": "detector_calibration",
        "status": "skipped",
        "detail": "Calibration was not evaluated because no usable detector probability was available.",
        "version": "not_available"
      }
    ],
    "model_versions": {
      "baseline_detector_adapter": "1.0",
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
- Every report includes report and calibration version identifiers. `not_available` means no validated artifact was applied; a detector score then remains evidence and cannot determine a verdict.
- A non-null `confidence` is the calibrated detector-family consensus synthetic-portrait probability supporting a `likely_*` verdict. It is probabilistic evidence, not certainty or proof of origin.

## Local browser connection

The web app submits to `http://localhost:8000` by default and polls its job resource until `report` is available. Set `NEXT_PUBLIC_API_BASE_URL` to use another API URL. The API permits browser requests from `http://localhost:3000` by default; set `VERITAS_FACE_WEB_ORIGINS` to a comma-separated allow-list for another local origin.

The local worker extracts presence-only EXIF/XMP facts and verifies embedded C2PA/Content Credentials with `c2pa-python`. Remote C2PA manifest fetch is disabled, so the worker does not retrieve external manifest stores for uploaded portraits. A C2PA result is normalized to `verified`, `invalid`, `not_present`, or `unavailable`; only `verified` means the embedded credential validated, and it verifies declared provenance rather than portrait authenticity. In particular, missing, invalid, or unavailable provenance and metadata presence or absence must never be interpreted as origin evidence.

## Baseline detector adapter

Set `VERITAS_FACE_INFERENCE_URL` to the base URL of the local C++ inference service (for example, `http://127.0.0.1:8080`) to enable the baseline detector. Without that opt-in setting, no portrait bytes leave the API worker and the detector evidence is reported as `unavailable`. The adapter runs only after one primary face has passed every quality gate. It crops that face in memory, converts it to RGB/HWC, resizes it to 224 × 224 pixels, and sends it to `POST /v1/infer`; neither the crop nor the service response body is retained in the completed report.

For a `200` response, the adapter accepts a score only when `synthetic_probability` is a finite number from 0 through 1 inclusive, `latency_ms` is a finite non-negative number, and `detector.id` plus `detector.version` are non-empty strings. A valid result appears as `baseline_synthetic_detector` evidence with the raw score, reported inference latency, and model version; `model_versions.baseline_detector` is `id@version`. A `503` model response or connection failure is `unavailable`; a malformed successful response or unexpected status is `invalid_response`. These outcomes always keep the report `inconclusive`.

## Calibration and final verdict policy

Set `VERITAS_FACE_CALIBRATION_PATH` to a local JSON artifact only after a calibration run has produced it from a held-out, licensed portrait manifest. The artifact is bounded to 256 KiB and must use this exact schema; its manifest checksum is an audit link, not a claim that the API has inspected the underlying portraits.

```json
{
  "schema_version": "1.0",
  "calibration_version": "heldout-portraits-2026.09",
  "validation_manifest_sha256": "sha256:<64 lowercase hexadecimal characters>",
  "detectors": [
    {
      "id": "synthetic-portrait-classifier",
      "version": "model-release-id",
      "family": "classifier-family-id",
      "points": [
        {"raw_probability": 0.0, "calibrated_probability": 0.01},
        {"raw_probability": 0.5, "calibrated_probability": 0.48},
        {"raw_probability": 1.0, "calibrated_probability": 0.99}
      ]
    }
  ]
}
```

Each detector release must appear once. Its calibration curve must cover 0 through 1, use strictly increasing raw probabilities, and never decrease calibrated probabilities. The API applies only an exact `id` plus `version` match; a missing, unreadable, invalid, or mismatched artifact leaves the report `inconclusive` and returns calibrated-evidence status explaining why. A successful transformation appears as `detector_calibration` evidence with the calibration version and calibrated probability, never the calibration path or manifest contents.

Face-quality failure always overrides score policy. For usable portraits, the policy averages releases within each detector `family`, then averages the independent family scores. A consensus at least `0.85` returns `likely_synthetic`; a consensus at most `0.15` returns `likely_authentic`; intermediate consensus is `inconclusive`. When at least two independent families are available and their family scores differ by `0.20` or more, the report is `inconclusive` with `meaningful_detector_disagreement`, regardless of the average. A single calibrated family can inform a probabilistic verdict, but it does not establish origin; C2PA and metadata facts are never inputs to these thresholds.

## C++ inference service

The inference service defaults to `127.0.0.1:8080`; pass `--host` and `--port` to configure its IPv4 listener. A service without a configured model provides its liveness and unavailable-model contract. To load a CPU ONNX model, start it with:

```text
veritas-inference --model /private/path/portrait-classifier.onnx \
  --model-version model-release-id \
  [--model-id synthetic-portrait-classifier] [--threads 1]
```

`--model-version` is required with `--model`, so reports always identify the configured artifact. A model-load or model-contract failure stops the process rather than silently serving a different model. The CMake build downloads the pinned official ONNX Runtime CPU SDK for local macOS and CI Linux targets on its first configure; for an offline or separately provisioned build, set `ONNXRUNTIME_ROOT` to an extracted official SDK instead. No model artifact is stored in this repository.

### `GET /health` and `GET /healthz`

Both liveness endpoints return:

```json
{
  "status": "ok",
  "service": "veritas-face-inference",
  "version": "0.1.0"
}
```

They intentionally report process liveness only. Model availability belongs to the model-info endpoint.

### `GET /v1/model-info`

Returns the current model identity, readiness, and input contract without accepting image data. Without `--model`, it returns:

```json
{
  "service": "veritas-face-inference",
  "version": "0.1.0",
  "model": {
    "id": "synthetic-portrait-classifier",
    "version": "not_loaded",
    "status": "unavailable"
  },
  "input": {
    "color_space": "RGB",
    "width": 224,
    "height": 224,
    "channels": 3,
    "layout": "HWC",
    "value_range": "0_to_255",
    "normalization": "not_configured"
  }
}
```

With a configured, validated CPU model it returns `status: "ready"` with the supplied model identifier and version:

```json
{
  "service": "veritas-face-inference",
  "version": "0.1.0",
  "model": {
    "id": "synthetic-portrait-classifier",
    "version": "model-release-id",
    "status": "ready",
    "runtime": "onnxruntime-cpu"
  },
  "input": {
    "color_space": "RGB",
    "width": 224,
    "height": 224,
    "channels": 3,
    "layout": "HWC",
    "value_range": "0_to_255",
    "normalization": "imagenet_rgb_v1"
  }
}
```

All responses use `Cache-Control: no-store`. Unknown routes return a JSON `not_found` error; methods other than `GET` on these endpoints return `405` with `Allow: GET`.

### `POST /v1/infer`

This endpoint returns a JSON `503 model_unavailable` response until an ONNX model is loaded. It does not retain a request body in that state. With a ready model, it accepts `application/json` with one standardized, primary-face crop:

```json
{
  "face_crop": {
    "color_space": "RGB",
    "width": 224,
    "height": 224,
    "channels": 3,
    "layout": "HWC",
    "value_range": "0_to_255",
    "pixels_base64": "base64-encoded row-major uint8 pixels"
  }
}
```

The decoded crop must contain exactly `width × height × channels` bytes. The service verifies fixed RGB/HWC geometry before converting row-major uint8 pixels to one float32 NCHW tensor. It applies the documented ImageNet RGB normalization per channel: `(pixel / 255 - mean) / standard_deviation`, with means `[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`.

The configured ONNX model must expose exactly one `float32` input shaped `[1, 3, 224, 224]` and one `float32` output containing exactly one synthetic probability in the inclusive range `0` to `1`. The service rejects malformed JSON or missing fields with `400 invalid_inference_request`, an incompatible crop with `422 invalid_face_crop`, an unloaded model with `503 model_unavailable`, and a runtime failure with `503 inference_failed`.

A successful response is:

```json
{
  "synthetic_probability": 0.78,
  "detector": {
    "id": "synthetic-portrait-classifier",
    "version": "model-version"
  },
  "latency_ms": 14.2
}
```

`synthetic_probability` is a model estimate, not a final verdict. It will be calibrated and combined with independent evidence by Python's report builder; the inference service never returns an authenticity determination.
