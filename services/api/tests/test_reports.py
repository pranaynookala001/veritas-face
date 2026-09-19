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
from app.reports import build_mock_report
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
    metadata.add_text("source", "mock-profile")
    image.save(buffer, format="PNG", pnginfo=metadata)
    return buffer.getvalue()


class MockReportTests(unittest.TestCase):
    def test_quality_passing_image_stays_inconclusive_without_detector_score(self) -> None:
        encoded, content = cv2.imencode(".png", detailed_image())
        self.assertTrue(encoded)
        artifact = StoredArtifact(content=content.tobytes(), media_type="image/png")

        from app.face_quality import assess_encoded_image

        assessment = assess_encoded_image(
            artifact.content,
            detector=FixtureDetector((FaceRectangle(20, 20, 140, 140),)),
        )
        report = build_mock_report(artifact, assessment).as_dict()

        self.assertEqual(report["verdict"], "inconclusive")
        self.assertIsNone(report["confidence"])
        self.assertEqual(report["reasons"], ["mock_analysis_no_detector_score"])
        self.assertEqual(report["evidence"][1]["status"], "passed")
        self.assertIn("240 × 240 pixels", report["evidence"][0]["detail"])

    def test_metadata_evidence_never_returns_embedded_values(self) -> None:
        artifact = StoredArtifact(content=png_with_embedded_metadata(), media_type="image/png")

        from app.face_quality import assess_encoded_image

        assessment = assess_encoded_image(artifact.content, detector=FixtureDetector(()))
        report = build_mock_report(artifact, assessment).as_dict()

        metadata_detail = report["evidence"][0]["detail"]
        self.assertIn("embedded metadata is present", metadata_detail)
        self.assertNotIn("mock-profile", metadata_detail)
        self.assertIn("not an authenticity signal", metadata_detail)


if __name__ == "__main__":
    unittest.main()
