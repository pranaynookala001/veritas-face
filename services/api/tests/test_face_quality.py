import sys
from dataclasses import dataclass
from pathlib import Path
import unittest

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.face_quality import (
    FaceRectangle,
    OpenCvHaarFaceDetector,
    assess_encoded_image,
    assess_image,
)


@dataclass(frozen=True)
class FixtureDetector:
    faces: tuple[FaceRectangle, ...]

    def detect(self, _grayscale_image: np.ndarray) -> tuple[FaceRectangle, ...]:
        return self.faces


def detailed_image(width: int = 240, height: int = 240) -> np.ndarray:
    """Create a deterministic high-detail BGR array without storing a portrait fixture."""
    rows, columns = np.indices((height, width))
    checkerboard = np.where((rows // 8 + columns // 8) % 2, 80, 180).astype(np.uint8)
    return cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)


class FaceQualityTests(unittest.TestCase):
    def test_baseline_detector_loads_and_rejects_a_blank_image(self) -> None:
        assessment = assess_image(
            np.zeros((256, 256, 3), dtype=np.uint8),
            detector=OpenCvHaarFaceDetector(),
        )

        self.assertEqual(assessment.inconclusive_reasons, ("no_face_detected",))

    def test_no_detected_face_is_inconclusive(self) -> None:
        assessment = assess_image(detailed_image(), detector=FixtureDetector(()))

        self.assertFalse(assessment.is_usable)
        self.assertEqual(assessment.inconclusive_reasons, ("no_face_detected",))
        self.assertIsNone(assessment.primary_face)

    def test_similarly_sized_faces_are_ambiguous(self) -> None:
        assessment = assess_image(
            detailed_image(),
            detector=FixtureDetector(
                (
                    FaceRectangle(10, 10, 120, 120),
                    FaceRectangle(130, 10, 110, 110),
                ),
            ),
        )

        self.assertEqual(assessment.face_count, 2)
        self.assertEqual(assessment.inconclusive_reasons, ("multiple_faces_ambiguous",))
        self.assertIsNone(assessment.primary_face)

    def test_largest_unambiguous_face_is_selected_and_quality_gates_pass(self) -> None:
        assessment = assess_image(
            detailed_image(),
            detector=FixtureDetector(
                (
                    FaceRectangle(160, 160, 100, 100),
                    FaceRectangle(20, 20, 140, 140),
                ),
            ),
        )

        self.assertTrue(assessment.is_usable)
        self.assertEqual(assessment.primary_face, FaceRectangle(20, 20, 140, 140))
        self.assertGreater(assessment.quality.sharpness_variance, 35)

    def test_low_resolution_and_blurry_face_are_inconclusive(self) -> None:
        image = np.full((180, 180, 3), 128, dtype=np.uint8)
        assessment = assess_image(
            image,
            detector=FixtureDetector((FaceRectangle(20, 20, 80, 80),)),
        )

        self.assertFalse(assessment.is_usable)
        self.assertIn("face_too_small", assessment.inconclusive_reasons)
        self.assertIn("face_too_blurry", assessment.inconclusive_reasons)

    def test_dark_and_bright_faces_are_inconclusive(self) -> None:
        detector = FixtureDetector((FaceRectangle(20, 20, 140, 140),))

        dark = assess_image(np.full((200, 200, 3), 10, dtype=np.uint8), detector=detector)
        bright = assess_image(np.full((200, 200, 3), 245, dtype=np.uint8), detector=detector)

        self.assertIn("face_too_dark", dark.inconclusive_reasons)
        self.assertIn("face_too_bright", bright.inconclusive_reasons)

    def test_encoded_image_uses_the_same_deterministic_detector_boundary(self) -> None:
        encoded, content = cv2.imencode(".png", detailed_image())
        self.assertTrue(encoded)

        assessment = assess_encoded_image(
            content.tobytes(),
            detector=FixtureDetector((FaceRectangle(20, 20, 140, 140),)),
        )

        self.assertTrue(assessment.is_usable)
