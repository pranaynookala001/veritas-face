import base64
import sys
from dataclasses import dataclass, field
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.baseline_detector import (
    FACE_CROP_SIZE,
    BaselineDetectorStatus,
    JsonHttpResponse,
    OnnxInferenceBaselineDetector,
)
from app.face_quality import FaceAssessment, FaceQualityMetrics, FaceRectangle
from app.storage import StoredArtifact


def encoded_image() -> bytes:
    image = np.full((240, 240, 3), (3, 2, 1), dtype=np.uint8)
    encoded, content = cv2.imencode(".png", image)
    assert encoded
    return content.tobytes()


def usable_assessment() -> FaceAssessment:
    return FaceAssessment(
        image_width=240,
        image_height=240,
        face_count=1,
        primary_face=FaceRectangle(20, 20, 160, 160),
        quality=FaceQualityMetrics(brightness=120.0, sharpness_variance=80.0),
        inconclusive_reasons=(),
    )


@dataclass
class FixtureTransport:
    response: JsonHttpResponse | None = None
    error: Exception | None = None
    calls: list[tuple[str, dict[str, object], float]] = field(default_factory=list)

    def post_json(self, url: str, payload: dict[str, object], *, timeout_seconds: float) -> JsonHttpResponse:
        self.calls.append((url, payload, timeout_seconds))
        if self.error is not None:
            raise self.error
        assert self.response is not None
        return self.response


class BaselineDetectorTests(unittest.TestCase):
    def test_adapter_sends_a_normalized_rgb_primary_face_and_captures_valid_evidence(self) -> None:
        transport = FixtureTransport(
            response=JsonHttpResponse(
                200,
                {
                    "synthetic_probability": 0.78,
                    "detector": {"id": "baseline-portrait", "version": "2026.09"},
                    "latency_ms": 14.2,
                },
            )
        )
        detector = OnnxInferenceBaselineDetector("http://127.0.0.1:8080/", transport=transport)

        result = detector.score_primary_face(
            StoredArtifact(content=encoded_image(), media_type="image/png"),
            usable_assessment(),
        )

        self.assertEqual(result.status, BaselineDetectorStatus.AVAILABLE)
        self.assertEqual(result.score, 0.78)
        self.assertEqual(result.detector_id, "baseline-portrait")
        self.assertEqual(result.detector_version, "2026.09")
        self.assertEqual(result.latency_ms, 14.2)
        self.assertEqual(len(transport.calls), 1)
        url, payload, timeout_seconds = transport.calls[0]
        self.assertEqual(url, "http://127.0.0.1:8080/v1/infer")
        self.assertEqual(timeout_seconds, 3.0)
        face_crop = payload["face_crop"]
        self.assertIsInstance(face_crop, dict)
        assert isinstance(face_crop, dict)
        self.assertEqual(
            {name: face_crop[name] for name in ("color_space", "width", "height", "channels", "layout", "value_range")},
            {
                "color_space": "RGB",
                "width": FACE_CROP_SIZE,
                "height": FACE_CROP_SIZE,
                "channels": 3,
                "layout": "HWC",
                "value_range": "0_to_255",
            },
        )
        pixels = base64.b64decode(face_crop["pixels_base64"])
        self.assertEqual(len(pixels), FACE_CROP_SIZE * FACE_CROP_SIZE * 3)
        self.assertEqual(pixels[:3], b"\x01\x02\x03")

    def test_adapter_rejects_invalid_score_latency_and_model_identity(self) -> None:
        valid_response = {
            "synthetic_probability": 0.5,
            "detector": {"id": "baseline-portrait", "version": "2026.09"},
            "latency_ms": 2.0,
        }
        invalid_responses = (
            {**valid_response, "synthetic_probability": -0.01},
            {**valid_response, "synthetic_probability": float("nan")},
            {**valid_response, "synthetic_probability": True},
            {**valid_response, "latency_ms": -0.1},
            {**valid_response, "latency_ms": float("inf")},
            {**valid_response, "detector": {"id": "", "version": "2026.09"}},
            {**valid_response, "detector": {"id": "baseline-portrait", "version": ""}},
        )
        artifact = StoredArtifact(content=encoded_image(), media_type="image/png")

        for response_payload in invalid_responses:
            with self.subTest(response_payload=response_payload):
                detector = OnnxInferenceBaselineDetector(
                    "http://127.0.0.1:8080",
                    transport=FixtureTransport(response=JsonHttpResponse(200, response_payload)),
                )
                result = detector.score_primary_face(artifact, usable_assessment())
                self.assertEqual(result.status, BaselineDetectorStatus.INVALID_RESPONSE)
                self.assertIsNone(result.score)
                self.assertIsNone(result.detector_version)

    def test_adapter_normalizes_unavailable_service_and_never_sends_a_bad_face(self) -> None:
        artifact = StoredArtifact(content=encoded_image(), media_type="image/png")
        unavailable = OnnxInferenceBaselineDetector(
            "http://127.0.0.1:8080",
            transport=FixtureTransport(response=JsonHttpResponse(503, {"error": {}})),
        ).score_primary_face(artifact, usable_assessment())
        self.assertEqual(unavailable.status, BaselineDetectorStatus.UNAVAILABLE)
        self.assertEqual(unavailable.reason, "inference_model_unavailable")

        unreachable = OnnxInferenceBaselineDetector(
            "http://127.0.0.1:8080",
            transport=FixtureTransport(error=OSError("not listening")),
        ).score_primary_face(artifact, usable_assessment())
        self.assertEqual(unreachable.status, BaselineDetectorStatus.UNAVAILABLE)
        self.assertEqual(unreachable.reason, "inference_service_unreachable")

        transport = FixtureTransport(response=JsonHttpResponse(200, {}))
        skipped = OnnxInferenceBaselineDetector(
            "http://127.0.0.1:8080", transport=transport
        ).score_primary_face(
            artifact,
            FaceAssessment(
                image_width=240,
                image_height=240,
                face_count=0,
                primary_face=None,
                quality=None,
                inconclusive_reasons=("no_face_detected",),
            ),
        )
        self.assertEqual(skipped.status, BaselineDetectorStatus.SKIPPED)
        self.assertEqual(transport.calls, [])

    def test_adapter_requires_a_finite_positive_http_configuration(self) -> None:
        with self.assertRaises(ValueError):
            OnnxInferenceBaselineDetector("file:///tmp/inference")
        with self.assertRaises(ValueError):
            OnnxInferenceBaselineDetector("http://127.0.0.1:8080", timeout_seconds=0)


if __name__ == "__main__":
    unittest.main()
