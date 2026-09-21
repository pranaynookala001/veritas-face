"""Adapter for the local ONNX baseline synthetic-portrait detector.

The C++ service has a deliberately narrow HTTP contract.  This module owns the
Python side of that contract: it creates a normalized primary-face crop, keeps
the crop in memory only for the request, and normalizes every service outcome
into safe report evidence.  A detector probability is never a verdict here;
calibration is applied by a later report stage.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from enum import Enum
import json
import math
from typing import Mapping, Protocol
from urllib.error import HTTPError
from urllib.request import HTTPRedirectHandler, Request, build_opener

import cv2
import numpy as np

from .face_quality import FaceAssessment
from .storage import StoredArtifact


BASELINE_ADAPTER_VERSION = "1.0"
FACE_CROP_SIZE = 224
INFERENCE_TIMEOUT_SECONDS = 3.0


class BaselineDetectorStatus(str, Enum):
    """Normalized outcomes that are safe to expose in report evidence."""

    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    INVALID_RESPONSE = "invalid_response"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class BaselineDetectorResult:
    """Validated detector evidence, without any crop or response body."""

    status: BaselineDetectorStatus
    score: float | None = None
    detector_id: str | None = None
    detector_version: str | None = None
    latency_ms: float | None = None
    reason: str | None = None

    @classmethod
    def skipped(cls) -> "BaselineDetectorResult":
        return cls(
            status=BaselineDetectorStatus.SKIPPED,
            reason="face_quality_inconclusive",
        )

    @classmethod
    def unavailable(cls, reason: str) -> "BaselineDetectorResult":
        return cls(status=BaselineDetectorStatus.UNAVAILABLE, reason=reason)

    @classmethod
    def invalid_response(cls, reason: str) -> "BaselineDetectorResult":
        return cls(status=BaselineDetectorStatus.INVALID_RESPONSE, reason=reason)


@dataclass(frozen=True)
class JsonHttpResponse:
    """Small transport result so adapter tests need no listening HTTP process."""

    status_code: int
    payload: object | None


class JsonHttpTransport(Protocol):
    """Boundary for the one JSON request made to the local inference service."""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """POST a JSON payload and decode any JSON response body."""


class UrllibJsonHttpTransport:
    """Standard-library implementation that sends no request retry or redirect."""

    def post_json(
        self,
        url: str,
        payload: Mapping[str, object],
        *,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        body = json.dumps(payload, separators=(",", ":")).encode("utf-8")
        request = Request(
            url,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with build_opener(_NoRedirect()).open(
                request, timeout=timeout_seconds
            ) as response:  # noqa: S310 - configured local service
                return JsonHttpResponse(response.status, _decode_json(response.read()))
        except HTTPError as error:
            return JsonHttpResponse(error.code, _decode_json(error.read()))


class BaselineDetector(Protocol):
    """Detector boundary used by the report worker and its tests."""

    def score_primary_face(
        self,
        artifact: StoredArtifact,
        assessment: FaceAssessment,
    ) -> BaselineDetectorResult:
        """Return normalized synthetic-portrait evidence for a usable primary face."""


class OnnxInferenceBaselineDetector:
    """Prepare one primary face for the C++ CPU ONNX inference service."""

    def __init__(
        self,
        service_url: str,
        *,
        transport: JsonHttpTransport | None = None,
        timeout_seconds: float = INFERENCE_TIMEOUT_SECONDS,
    ) -> None:
        normalized_url = service_url.rstrip("/")
        if not normalized_url.startswith(("http://", "https://")):
            raise ValueError("the inference service URL must use HTTP or HTTPS")
        if timeout_seconds <= 0 or not math.isfinite(timeout_seconds):
            raise ValueError("the inference timeout must be a finite positive number")
        self._endpoint = f"{normalized_url}/v1/infer"
        self._transport = transport or UrllibJsonHttpTransport()
        self._timeout_seconds = timeout_seconds

    def score_primary_face(
        self,
        artifact: StoredArtifact,
        assessment: FaceAssessment,
    ) -> BaselineDetectorResult:
        if not assessment.is_usable or assessment.primary_face is None:
            return BaselineDetectorResult.skipped()

        crop = _primary_face_crop(artifact.content, assessment)
        if crop is None:
            return BaselineDetectorResult.unavailable("face_crop_unavailable")

        payload: dict[str, object] = {
            "face_crop": {
                "color_space": "RGB",
                "width": FACE_CROP_SIZE,
                "height": FACE_CROP_SIZE,
                "channels": 3,
                "layout": "HWC",
                "value_range": "0_to_255",
                "pixels_base64": base64.b64encode(crop.tobytes()).decode("ascii"),
            }
        }
        try:
            response = self._transport.post_json(
                self._endpoint,
                payload,
                timeout_seconds=self._timeout_seconds,
            )
        except OSError:
            return BaselineDetectorResult.unavailable("inference_service_unreachable")
        except Exception:
            # A worker must preserve its safe report when an adapter implementation fails.
            return BaselineDetectorResult.unavailable("inference_service_unreachable")

        if response.status_code == 503:
            return BaselineDetectorResult.unavailable("inference_model_unavailable")
        if response.status_code != 200:
            return BaselineDetectorResult.invalid_response("unexpected_inference_status")
        return _validated_response(response.payload)


def _primary_face_crop(content: bytes, assessment: FaceAssessment) -> np.ndarray | None:
    """Decode, crop, resize, and convert BGR pixels to the required RGB/HWC form."""
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    face = assessment.primary_face
    if image is None or face is None:
        return None
    crop = image[face.y : face.y + face.height, face.x : face.x + face.width]
    if crop.size == 0:
        return None
    interpolation = cv2.INTER_AREA if max(crop.shape[:2]) >= FACE_CROP_SIZE else cv2.INTER_LINEAR
    resized = cv2.resize(crop, (FACE_CROP_SIZE, FACE_CROP_SIZE), interpolation=interpolation)
    return cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)


class _NoRedirect(HTTPRedirectHandler):
    """Keep a configured local inference request from being redirected elsewhere."""

    def redirect_request(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        return None


def _validated_response(payload: object | None) -> BaselineDetectorResult:
    """Reject malformed, non-finite, and unversioned successful service responses."""
    if not isinstance(payload, Mapping):
        return BaselineDetectorResult.invalid_response("inference_response_not_an_object")

    score = _finite_number(payload.get("synthetic_probability"))
    if score is None or not 0 <= score <= 1:
        return BaselineDetectorResult.invalid_response("invalid_synthetic_probability")

    latency_ms = _finite_number(payload.get("latency_ms"))
    if latency_ms is None or latency_ms < 0:
        return BaselineDetectorResult.invalid_response("invalid_inference_latency")

    detector = payload.get("detector")
    if not isinstance(detector, Mapping):
        return BaselineDetectorResult.invalid_response("missing_detector_identity")
    detector_id = _nonempty_string(detector.get("id"))
    detector_version = _nonempty_string(detector.get("version"))
    if detector_id is None or detector_version is None:
        return BaselineDetectorResult.invalid_response("missing_detector_identity")

    return BaselineDetectorResult(
        status=BaselineDetectorStatus.AVAILABLE,
        score=score,
        detector_id=detector_id,
        detector_version=detector_version,
        latency_ms=latency_ms,
    )


def _decode_json(body: bytes) -> object | None:
    try:
        return json.loads(body)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def _nonempty_string(value: object) -> str | None:
    return value.strip() if isinstance(value, str) and value.strip() else None
