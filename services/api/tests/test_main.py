import io
import stat
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
import unittest
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.domain import JobStatus, Report, Verdict
from app.intake import MAX_UPLOAD_BYTES
from app.main import APP_VERSION, app, get_job_store
from app.storage import JobExpiredError, JobStateError, StoredArtifact, TemporaryJobStore


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
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = TemporaryJobStore(timedelta(hours=24), directory=Path(self.temporary_directory.name))
        app.dependency_overrides[get_job_store] = lambda: self.store
        self.client = TestClient(app)

    def tearDown(self) -> None:
        app.dependency_overrides.clear()
        self.store.close()
        self.temporary_directory.cleanup()

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

    def test_local_web_origin_may_submit_and_read_job_reports(self) -> None:
        response = self.client.options(
            "/v1/jobs",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["access-control-allow-origin"], "http://localhost:3000")

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

        job = self.client.get(f"/v1/jobs/{payload['job_id']}")
        self.assertEqual(job.status_code, 200)
        self.assertEqual(job.json()["status"], "completed")
        self.assertNotIn("private-portrait.png", str(job.json()))
        self.assertNotIn(image_bytes().decode(errors="ignore"), str(job.json()))

    def test_uploaded_image_receives_a_completed_mock_quality_report(self) -> None:
        receipt = self.client.post(
            "/v1/jobs",
            files={"portrait": ("private-portrait.png", image_bytes(), "image/png")},
        )

        self.assertEqual(receipt.status_code, 202)
        self.assertEqual(receipt.json()["status"], "queued")
        response = self.client.get(f"/v1/jobs/{receipt.json()['job_id']}")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["status"], "completed")
        self.assertEqual(payload["report"]["verdict"], "inconclusive")
        self.assertIsNone(payload["report"]["confidence"])
        self.assertEqual(payload["report"]["report_version"], "mock-evidence-v1")
        self.assertEqual(payload["report"]["calibration_version"], "not_calibrated_mock_v1")
        self.assertIn("no_face_detected", payload["report"]["reasons"])
        self.assertEqual(
            [item["source"] for item in payload["report"]["evidence"]],
            ["image_metadata", "face_quality"],
        )
        self.assertIn("2 × 2 pixels", payload["report"]["evidence"][0]["detail"])
        self.assertIn("not an authenticity signal", payload["report"]["evidence"][0]["detail"])
        self.assertNotIn("private-portrait.png", str(payload))
        self.assertNotIn(image_bytes().decode(errors="ignore"), str(payload))

    def test_unknown_job_uses_the_error_envelope(self) -> None:
        response = self.client.get("/v1/jobs/00000000-0000-0000-0000-000000000000")

        self.assert_error(response, 404, "job_not_found")

    def test_expired_job_exposes_only_its_expired_status(self) -> None:
        job = self.store.create_job(
            StoredArtifact(content=b"temporary image", media_type="image/png"),
            now=datetime.now(timezone.utc) - timedelta(days=2),
        )

        response = self.client.get(f"/v1/jobs/{job.job_id}")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["status"], "expired")
        self.assertNotIn("temporary image", str(response.json()))
        with self.assertRaises(JobExpiredError):
            self.store.get_artifact(job.job_id)

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


class TemporaryJobStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.store = TemporaryJobStore(timedelta(minutes=5), directory=Path(self.temporary_directory.name))
        self.created_at = datetime(2026, 9, 15, tzinfo=timezone.utc)

    def tearDown(self) -> None:
        self.store.close()
        self.temporary_directory.cleanup()

    def test_state_machine_allows_only_monotonic_worker_transitions(self) -> None:
        job = self.store.create_job(
            StoredArtifact(content=b"validated image", media_type="image/png"),
            now=self.created_at,
        )

        with self.assertRaises(JobStateError):
            self.store.transition(job.job_id, JobStatus.COMPLETED, now=self.created_at)

        processing = self.store.transition(job.job_id, JobStatus.PROCESSING, now=self.created_at)
        completed = self.store.transition(job.job_id, JobStatus.COMPLETED, now=self.created_at)

        self.assertEqual(processing.status, JobStatus.PROCESSING)
        self.assertEqual(completed.status, JobStatus.COMPLETED)
        self.assertEqual(self.store.get_artifact(job.job_id, now=self.created_at).content, b"validated image")

        with self.assertRaises(JobStateError):
            self.store.transition(job.job_id, JobStatus.FAILED, now=self.created_at)

    def test_artifacts_are_private_and_expire_when_a_worker_reads_them(self) -> None:
        job = self.store.create_job(
            StoredArtifact(content=b"temporary image", media_type="image/png"),
            now=self.created_at,
        )

        self.assertIsNotNone(job.artifact_path)
        self.assertEqual(stat.S_IMODE(job.artifact_path.stat().st_mode), 0o600)
        with self.assertRaises(JobExpiredError):
            self.store.get_artifact(job.job_id, now=job.expires_at)
        self.assertEqual(self.store.get_job(job.job_id, now=job.expires_at).status, JobStatus.EXPIRED)

    def test_retention_deletes_the_artifact_and_marks_the_job_expired(self) -> None:
        job = self.store.create_job(
            StoredArtifact(content=b"temporary image", media_type="image/png"),
            now=self.created_at,
        )

        self.assertEqual(self.store.cleanup_expired(now=job.expires_at), 1)
        self.assertEqual(self.store.get_job(job.job_id, now=job.expires_at).status, JobStatus.EXPIRED)
        with self.assertRaises(JobExpiredError):
            self.store.get_artifact(job.job_id, now=job.expires_at)

    def test_expiration_removes_the_completed_report_with_the_artifact(self) -> None:
        job = self.store.create_job(
            StoredArtifact(content=b"validated image", media_type="image/png"),
            now=self.created_at,
        )
        report = Report(
            verdict=Verdict.INCONCLUSIVE,
            confidence=None,
            reasons=("mock_analysis_no_detector_score",),
        )

        completed = self.store.complete_with_report(job.job_id, report, now=self.created_at)
        expired = self.store.get_job(job.job_id, now=completed.expires_at)

        self.assertEqual(expired.status, JobStatus.EXPIRED)
        self.assertIsNone(expired.report)


if __name__ == "__main__":
    unittest.main()
