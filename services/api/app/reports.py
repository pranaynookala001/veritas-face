"""Assemble the temporary v1 mock report without making authenticity claims."""

from __future__ import annotations

from io import BytesIO

from PIL import Image

from .domain import Evidence, Report, Verdict
from .face_quality import FaceAssessment
from .storage import StoredArtifact


REPORT_VERSION = "mock-evidence-v1"
CALIBRATION_VERSION = "not_calibrated_mock_v1"
FACE_QUALITY_VERSION = "1.0"


def build_mock_report(artifact: StoredArtifact, assessment: FaceAssessment) -> Report:
    """Build a completed, explicitly inconclusive report from local eligibility evidence.

    V1 has no provenance verifier or forensic model yet. A quality-passing image is
    therefore still inconclusive; quality failures only add their concrete reason
    codes and never become authenticity evidence.
    """
    reasons = assessment.inconclusive_reasons or ("mock_analysis_no_detector_score",)
    quality_status = "inconclusive" if assessment.inconclusive_reasons else "passed"
    quality_detail = (
        "Face-quality gates require an inconclusive result: "
        + ", ".join(assessment.inconclusive_reasons)
        if assessment.inconclusive_reasons
        else "One dominant face passed local eligibility checks."
    )

    return Report(
        report_version=REPORT_VERSION,
        calibration_version=CALIBRATION_VERSION,
        verdict=Verdict.INCONCLUSIVE,
        confidence=None,
        reasons=reasons,
        evidence=(
            Evidence(
                source="image_metadata",
                status="observed",
                detail=_metadata_detail(artifact),
                version="pillow-decoder",
            ),
            Evidence(
                source="face_quality",
                status=quality_status,
                detail=quality_detail,
                version=FACE_QUALITY_VERSION,
            ),
        ),
        model_versions={"face_quality": FACE_QUALITY_VERSION},
    )


def _metadata_detail(artifact: StoredArtifact) -> str:
    """Summarize safe decoded-image facts without returning metadata values."""
    with Image.open(BytesIO(artifact.content)) as image:
        image.load()
        has_embedded_metadata = bool(image.info or image.getexif())
        metadata_presence = "embedded metadata is present" if has_embedded_metadata else "no embedded metadata is present"
        format_name = (image.format or artifact.media_type).upper()
        return (
            f"Decoded {format_name} image, {image.width} × {image.height} pixels; "
            f"{metadata_presence}. Metadata presence or absence is not an authenticity signal."
        )
