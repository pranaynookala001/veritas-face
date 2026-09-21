import io
import sys
from dataclasses import dataclass
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.baseline_detector import BaselineDetectorResult, BaselineDetectorStatus
from app.face_quality import FaceRectangle
from app.provenance import C2paSdkVerifier, C2paVerification, C2paVerificationStatus
from app.reports import build_local_report
from app.storage import StoredArtifact


@dataclass(frozen=True)
class FixtureDetector:
    faces: tuple[FaceRectangle, ...]

    def detect(self, _grayscale_image: np.ndarray) -> tuple[FaceRectangle, ...]:
        return self.faces


def detailed_image() -> np.ndarray:
    rows, columns = np.indices((240, 240))
    checkerboard = np.where((rows // 8 + columns // 8) % 2, 80, 180).astype(np.uint8)
    return cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)


def png_with_embedded_metadata() -> bytes:
    image = Image.new("RGB", (240, 240), color=(18, 52, 86))
    buffer = io.BytesIO()
    metadata = PngImagePlugin.PngInfo()
    metadata.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta>private-xmp-profile</x:xmpmeta>")
    exif = Image.Exif()
    exif[0x010F] = "private-camera-maker"
    image.save(buffer, format="PNG", pnginfo=metadata, exif=exif)
    return buffer.getvalue()


@dataclass(frozen=True)
class FixtureC2paVerifier:
    result: C2paVerification

    def verify(self, _content: bytes, _media_type: str) -> C2paVerification:
        return self.result


@dataclass(frozen=True)
class FixtureBaselineDetector:
    result: BaselineDetectorResult

    def score_primary_face(self, _artifact: StoredArtifact, _assessment) -> BaselineDetectorResult:
        return self.result


class LocalReportTests(unittest.TestCase):
    def test_quality_passing_image_reports_an_unavailable_opt_in_detector(self) -> None:
        encoded, content = cv2.imencode(".png", detailed_image())
        self.assertTrue(encoded)
        artifact = StoredArtifact(content=content.tobytes(), media_type="image/png")

        from app.face_quality import assess_encoded_image

        assessment = assess_encoded_image(
            artifact.content,
            detector=FixtureDetector((FaceRectangle(20, 20, 140, 140),)),
        )
        report = build_local_report(artifact, assessment).as_dict()

        self.assertEqual(report["verdict"], "inconclusive")
        self.assertIsNone(report["confidence"])
        self.assertEqual(report["reasons"], ["baseline_detector_unavailable"])
        self.assertEqual(report["evidence"][1]["status"], "not_present")
        self.assertEqual(report["evidence"][2]["status"], "passed")
        self.assertEqual(report["evidence"][3]["source"], "baseline_synthetic_detector")
        self.assertEqual(report["evidence"][3]["status"], "unavailable")
        self.assertIn("remains inconclusive", report["evidence"][3]["detail"])
        self.assertIn("240 × 240 pixels", report["evidence"][0]["detail"])

    def test_valid_baseline_score_is_versioned_latency_evidence_not_a_verdict(self) -> None:
        encoded, content = cv2.imencode(".png", detailed_image())
        self.assertTrue(encoded)
        artifact = StoredArtifact(content=content.tobytes(), media_type="image/png")

        from app.face_quality import assess_encoded_image

        assessment = assess_encoded_image(
            artifact.content,
            detector=FixtureDetector((FaceRectangle(20, 20, 140, 140),)),
        )
        report = build_local_report(
            artifact,
            assessment,
            baseline_detector=FixtureBaselineDetector(
                BaselineDetectorResult(
                    status=BaselineDetectorStatus.AVAILABLE,
                    score=0.78,
                    detector_id="baseline-portrait",
                    detector_version="2026.09",
                    latency_ms=14.2,
                )
            ),
        ).as_dict()

        self.assertEqual(report["verdict"], "inconclusive")
        self.assertIsNone(report["confidence"])
        self.assertEqual(report["reasons"], ["detector_score_uncalibrated"])
        evidence = report["evidence"][3]
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["score"], 0.78)
        self.assertEqual(evidence["version"], "2026.09")
        self.assertIn("14.200 ms", evidence["detail"])
        self.assertIn("not an authenticity verdict", evidence["detail"])
        self.assertEqual(report["model_versions"]["baseline_detector"], "baseline-portrait@2026.09")
        self.assertEqual(report["model_versions"]["baseline_detector_adapter"], "1.0")

    def test_metadata_evidence_extracts_exif_and_xmp_without_returning_values(self) -> None:
        artifact = StoredArtifact(content=png_with_embedded_metadata(), media_type="image/png")

        from app.face_quality import assess_encoded_image

        assessment = assess_encoded_image(artifact.content, detector=FixtureDetector(()))
        report = build_local_report(artifact, assessment).as_dict()

        metadata_detail = report["evidence"][0]["detail"]
        self.assertIn("embedded metadata is present", metadata_detail)
        self.assertIn("EXIF metadata is present", metadata_detail)
        self.assertIn("XMP metadata is present", metadata_detail)
        self.assertNotIn("private-camera-maker", metadata_detail)
        self.assertNotIn("private-xmp-profile", metadata_detail)
        self.assertIn("not an authenticity signal", metadata_detail)

    def test_real_sdk_normalizes_a_missing_manifest_without_contacting_remote_hosts(self) -> None:
        result = C2paSdkVerifier().verify(png_with_embedded_metadata(), "image/png")

        self.assertEqual(result.status, C2paVerificationStatus.NOT_PRESENT)
        self.assertTrue(result.version.startswith("c2pa-python-"))

    def test_verified_credential_is_reported_as_provenance_not_an_authenticity_verdict(self) -> None:
        encoded, content = cv2.imencode(".png", detailed_image())
        self.assertTrue(encoded)
        artifact = StoredArtifact(content=content.tobytes(), media_type="image/png")

        from app.face_quality import assess_encoded_image

        report = build_local_report(
            artifact,
            assess_encoded_image(artifact.content, detector=FixtureDetector(())),
            c2pa_verifier=FixtureC2paVerifier(
                C2paVerification(C2paVerificationStatus.VERIFIED, "c2pa-python-test")
            ),
        ).as_dict()

        credential = report["evidence"][1]
        self.assertEqual(report["verdict"], "inconclusive")
        self.assertEqual(credential["status"], "verified")
        self.assertIn("declared provenance", credential["detail"])
        self.assertIn("not whether this portrait is authentic or synthetic", credential["detail"])


if __name__ == "__main__":
    unittest.main()
