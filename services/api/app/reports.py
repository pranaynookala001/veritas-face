"""Assemble the temporary local evidence report without authenticity claims."""

from __future__ import annotations

from .domain import Evidence, Report, Verdict
from .face_quality import FaceAssessment
from .provenance import (
    C2paVerifier,
    PROVENANCE_ADAPTER_VERSION,
    c2pa_detail,
    inspect_provenance,
    metadata_detail,
)
from .storage import StoredArtifact


REPORT_VERSION = "local-evidence-v2"
CALIBRATION_VERSION = "not_calibrated_v1"
FACE_QUALITY_VERSION = "1.0"


def build_local_report(
    artifact: StoredArtifact,
    assessment: FaceAssessment,
    *,
    c2pa_verifier: C2paVerifier | None = None,
) -> Report:
    """Build an explicitly inconclusive report from local eligibility and provenance.

    A verified credential only attests to declared provenance, and V1 has no
    synthetic-portrait detector score. A quality-passing image therefore remains
    inconclusive; quality failures only add concrete reason codes.
    """
    provenance = inspect_provenance(
        artifact.content,
        artifact.media_type,
        c2pa_verifier=c2pa_verifier,
    )
    reasons = assessment.inconclusive_reasons or ("no_synthetic_detector_score",)
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
                detail=metadata_detail(provenance.metadata),
                version=f"provenance-adapter-{PROVENANCE_ADAPTER_VERSION}",
            ),
            Evidence(
                source="c2pa_content_credentials",
                status=provenance.c2pa.status.value,
                detail=c2pa_detail(provenance.c2pa),
                version=provenance.c2pa.version,
            ),
            Evidence(
                source="face_quality",
                status=quality_status,
                detail=quality_detail,
                version=FACE_QUALITY_VERSION,
            ),
        ),
        model_versions={
            "c2pa": provenance.c2pa.version,
            "face_quality": FACE_QUALITY_VERSION,
            "provenance_adapter": PROVENANCE_ADAPTER_VERSION,
        },
    )
