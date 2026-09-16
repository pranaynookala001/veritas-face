import io
import sys
from datetime import datetime, timezone
from pathlib import Path
import unittest
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.intake import MAX_UPLOAD_BYTES
from app.main import APP_VERSION, app


def image_bytes(image_format: str = "PNG") -> bytes:
    """Create a deterministic in-memory image without adding image artifacts to git."""
    image = Image.new("RGB", (2, 2), color=(18, 52, 86))
    buffer = io.BytesIO()
    image.save(buffer, format=image_format)
    return buffer.getvalue()


def animated_png_bytes() -> bytes:
    """Create a small APNG fixture entirely in memory."""
    first_frame = Image.new("RGB", (2, 2), color=(18, 52, 86))
    second_frame = Image.new("RGB", (2, 2), color=(135, 42, 27))
    buffer = io.BytesIO()
    first_frame.save(
        buffer,
        format="PNG",
        save_all=True,
        append_images=[second_frame],
        duration=100,
        loop=0,
    )
    return buffer.getvalue()


class ApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(app)

    def assert_error(self, response, status_code: int, code: str) -> dict:
        self.assertEqual(response.status_code, status_code)
        payload = response.json()
        self.assertEqual(payload["error"]["code"], code)
        self.assertIn("message", payload["error"])
        self.assertIn("issues", payload["error"])
        return payload

    def test_health_reports_service_and_version(self) -> None:
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "service": "veritas-face-api", "version": APP_VERSION},
        )

    def test_valid_png_receives_a_queued_job_receipt(self) -> None:
        response = self.client.post(
            "/v1/jobs",
            files={"portrait": ("private-portrait.png", image_bytes(), "image/png")},
        )

        self.assertEqual(response.status_code, 202)
        payload = response.json()
        self.assertEqual(payload["status"], "queued")
        UUID(payload["job_id"])
        expires_at = datetime.fromisoformat(payload["expires_at"].replace("Z", "+00:00"))
        self.assertGreater(expires_at, datetime.now(timezone.utc))
        self.assertNotIn("private-portrait.png", str(payload))

    def test_valid_image_with_a_generic_mime_type_uses_content_detection(self) -> None:
        response = self.client.post(
            "/v1/jobs",
            files={"portrait": ("portrait", image_bytes(), "application/octet-stream")},
        )

        self.assertEqual(response.status_code, 202)

    def test_openapi_describes_the_multipart_request_and_error_responses(self) -> None:
        schema = self.client.get("/openapi.json").json()
        create_job = schema["paths"]["/v1/jobs"]["post"]
        request_schema = create_job["requestBody"]["content"]["multipart/form-data"]["schema"]
        request_model = schema["components"]["schemas"][request_schema["$ref"].rsplit("/", 1)[-1]]

        self.assertEqual(request_model["properties"]["portrait"]["type"], "string")
        self.assertEqual(
            request_model["properties"]["portrait"]["contentMediaType"],
            "application/octet-stream",
        )
        self.assertEqual(create_job["responses"]["415"]["content"]["application/json"]["schema"]["$ref"], "#/components/schemas/ErrorResponse")

    def test_missing_upload_uses_the_error_envelope(self) -> None:
        response = self.client.post("/v1/jobs")

        payload = self.assert_error(response, 422, "request_validation_failed")
        self.assertEqual(payload["error"]["issues"][0]["field"], "portrait")

    def test_rejects_empty_oversized_and_unsupported_uploads(self) -> None:
        empty = self.client.post("/v1/jobs", files={"portrait": ("empty.png", b"", "image/png")})
        self.assert_error(empty, 400, "empty_upload")

        oversized = self.client.post(
            "/v1/jobs",
            files={"portrait": ("large.png", b"0" * (MAX_UPLOAD_BYTES + 1), "image/png")},
        )
        self.assert_error(oversized, 413, "upload_too_large")

        unsupported = self.client.post(
            "/v1/jobs",
            files={"portrait": ("portrait.gif", image_bytes(), "image/gif")},
        )
        self.assert_error(unsupported, 415, "unsupported_media_type")

    def test_rejects_invalid_or_mismatched_encoded_content(self) -> None:
        invalid = self.client.post(
            "/v1/jobs",
            files={"portrait": ("portrait.png", b"not an image", "image/png")},
        )
        self.assert_error(invalid, 422, "invalid_image")

        jpeg = image_bytes("JPEG")
        truncated = self.client.post(
            "/v1/jobs",
            files={"portrait": ("portrait.jpg", jpeg[: len(jpeg) // 2], "image/jpeg")},
        )
        self.assert_error(truncated, 422, "invalid_image")

        mismatched = self.client.post(
            "/v1/jobs",
            files={"portrait": ("portrait.png", image_bytes("PNG"), "image/jpeg")},
        )
        self.assert_error(mismatched, 415, "media_type_mismatch")

    def test_rejects_animated_images_outside_the_still_image_boundary(self) -> None:
        response = self.client.post(
            "/v1/jobs",
            files={"portrait": ("animated.png", animated_png_bytes(), "image/png")},
        )

        self.assert_error(response, 422, "animated_image_not_supported")

    def test_unknown_routes_use_the_error_envelope(self) -> None:
        response = self.client.get("/unknown")

        self.assert_error(response, 404, "not_found")


if __name__ == "__main__":
    unittest.main()
