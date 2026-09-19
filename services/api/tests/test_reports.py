import io
import sys
from dataclasses import dataclass
from pathlib import Path
import unittest

import cv2
import numpy as np
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

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


class LocalReportTests(unittest.TestCase):
    def test_quality_passing_image_stays_inconclusive_without_detector_score(self) -> None:
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
        self.assertEqual(report["reasons"], ["no_synthetic_detector_score"])
        self.assertEqual(report["evidence"][1]["status"], "not_present")
        self.assertEqual(report["evidence"][2]["status"], "passed")
        self.assertIn("240 × 240 pixels", report["evidence"][0]["detail"])

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
