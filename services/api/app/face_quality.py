"""Local face selection and quality gates for a still-image analysis job."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


MIN_FACE_DIMENSION = 96
MIN_SHARPNESS_VARIANCE = 35.0
MIN_BRIGHTNESS = 35.0
MAX_BRIGHTNESS = 220.0
AMBIGUOUS_FACE_AREA_RATIO = 0.65


@dataclass(frozen=True)
class FaceRectangle:
    """A face bounding box in pixel coordinates."""

    x: int
    y: int
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    def bounded_to(self, image_width: int, image_height: int) -> FaceRectangle | None:
        """Clip a detector result to the image and reject empty intersections."""
        left = max(self.x, 0)
        top = max(self.y, 0)
        right = min(self.x + self.width, image_width)
        bottom = min(self.y + self.height, image_height)
        if right <= left or bottom <= top:
            return None
        return FaceRectangle(x=left, y=top, width=right - left, height=bottom - top)


@dataclass(frozen=True)
class FaceQualityMetrics:
    """Non-biometric quality signals that decide whether analysis can proceed."""

    brightness: float
    sharpness_variance: float


@dataclass(frozen=True)
class FaceAssessment:
    """Primary face selection with explicit reasons that require an inconclusive report."""

    image_width: int
    image_height: int
    face_count: int
    primary_face: FaceRectangle | None
    quality: FaceQualityMetrics | None
    inconclusive_reasons: tuple[str, ...]

    @property
    def is_usable(self) -> bool:
        return not self.inconclusive_reasons


class FaceDetector(Protocol):
    """Injectable detector boundary used by local inference and deterministic tests."""

    def detect(self, grayscale_image: np.ndarray) -> tuple[FaceRectangle, ...]:
        """Return candidate face bounds in the supplied grayscale image."""


class OpenCvHaarFaceDetector:
    """CPU-only baseline detector using OpenCV's distributed frontal-face cascade."""

    def __init__(self, cascade_path: Path | None = None) -> None:
        resolved_path = cascade_path or Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(resolved_path))
        if self._cascade.empty():
            raise RuntimeError("the OpenCV frontal-face cascade could not be loaded")

    def detect(self, grayscale_image: np.ndarray) -> tuple[FaceRectangle, ...]:
        detections = self._cascade.detectMultiScale(
            grayscale_image,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(24, 24),
        )
        return tuple(FaceRectangle(int(x), int(y), int(width), int(height)) for x, y, width, height in detections)


def assess_encoded_image(
    content: bytes,
    *,
    detector: FaceDetector | None = None,
) -> FaceAssessment:
    """Decode validated bytes and produce a no-claim face-quality assessment."""
    image = cv2.imdecode(np.frombuffer(content, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("validated image bytes could not be decoded for face analysis")
    return assess_image(image, detector=detector)


def assess_image(image: np.ndarray, *, detector: FaceDetector | None = None) -> FaceAssessment:
    """Choose a dominant face and reject inputs that cannot support a reliable verdict."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("face analysis requires a three-channel color image")

    image_height, image_width = image.shape[:2]
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_detector = detector or OpenCvHaarFaceDetector()
    faces = _bounded_faces(face_detector.detect(grayscale), image_width, image_height)

    if not faces:
        return FaceAssessment(
            image_width=image_width,
            image_height=image_height,
            face_count=0,
            primary_face=None,
            quality=None,
            inconclusive_reasons=("no_face_detected",),
        )

    primary_face = faces[0]
    if len(faces) > 1 and faces[1].area >= primary_face.area * AMBIGUOUS_FACE_AREA_RATIO:
        return FaceAssessment(
            image_width=image_width,
            image_height=image_height,
            face_count=len(faces),
            primary_face=None,
            quality=None,
            inconclusive_reasons=("multiple_faces_ambiguous",),
        )

    face_crop = grayscale[
        primary_face.y : primary_face.y + primary_face.height,
        primary_face.x : primary_face.x + primary_face.width,
    ]
    quality = FaceQualityMetrics(
        brightness=float(face_crop.mean()),
        sharpness_variance=float(cv2.Laplacian(face_crop, cv2.CV_64F).var()),
    )
    reasons: list[str] = []
    if min(primary_face.width, primary_face.height) < MIN_FACE_DIMENSION:
        reasons.append("face_too_small")
    if quality.sharpness_variance < MIN_SHARPNESS_VARIANCE:
        reasons.append("face_too_blurry")
    if quality.brightness < MIN_BRIGHTNESS:
        reasons.append("face_too_dark")
    if quality.brightness > MAX_BRIGHTNESS:
        reasons.append("face_too_bright")

    return FaceAssessment(
        image_width=image_width,
        image_height=image_height,
        face_count=len(faces),
        primary_face=primary_face,
        quality=quality,
        inconclusive_reasons=tuple(reasons),
    )


def _bounded_faces(
    candidates: tuple[FaceRectangle, ...],
    image_width: int,
    image_height: int,
) -> tuple[FaceRectangle, ...]:
    bounded = [
        face
        for candidate in candidates
        if (face := candidate.bounded_to(image_width, image_height)) is not None
    ]
    return tuple(sorted(bounded, key=lambda face: (-face.area, face.y, face.x)))
