"""Assemble the temporary local evidence report without authenticity claims."""

from __future__ import annotations

from .baseline_detector import (
    BASELINE_ADAPTER_VERSION,
    BaselineDetector,
    BaselineDetectorResult,
    BaselineDetectorStatus,
)
from .calibration import (
    CalibrationApplication,
    CalibrationRegistry,
    CalibrationStatus,
    VerdictDecision,
    apply_calibration,
    decide_final_verdict,
)
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


REPORT_VERSION = "local-evidence-v4"
FACE_QUALITY_VERSION = "1.0"


def build_local_report(
    artifact: StoredArtifact,
    assessment: FaceAssessment,
    *,
    c2pa_verifier: C2paVerifier | None = None,
    baseline_detector: BaselineDetector | None = None,
    calibration_registry: CalibrationRegistry | None = None,
) -> Report:
    """Build a report from local eligibility and explicitly versioned evidence.

    A verified credential only attests to declared provenance. A validated detector
    score is preserved as evidence until an exact detector-release calibration is
    available. The final policy remains probabilistic and never treats metadata or
    provenance absence as evidence of either origin.
    """
    provenance = inspect_provenance(
        artifact.content,
        artifact.media_type,
        c2pa_verifier=c2pa_verifier,
    )
    detector_result = _detector_result(artifact, assessment, baseline_detector)
    calibration = apply_calibration(detector_result, calibration_registry)
    decision = _final_decision(assessment, detector_result, calibration)
    quality_status = "inconclusive" if assessment.inconclusive_reasons else "passed"
    quality_detail = (
        "Face-quality gates require an inconclusive result: "
        + ", ".join(assessment.inconclusive_reasons)
        if assessment.inconclusive_reasons
        else "One dominant face passed local eligibility checks."
    )

    evidence = (
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
        _detector_evidence(detector_result),
        _calibration_evidence(calibration),
    )
    model_versions = {
        "baseline_detector_adapter": BASELINE_ADAPTER_VERSION,
        "c2pa": provenance.c2pa.version,
        "face_quality": FACE_QUALITY_VERSION,
        "provenance_adapter": PROVENANCE_ADAPTER_VERSION,
    }
    if detector_result.status is BaselineDetectorStatus.AVAILABLE:
        assert detector_result.detector_id is not None
        assert detector_result.detector_version is not None
        model_versions["baseline_detector"] = (
            f"{detector_result.detector_id}@{detector_result.detector_version}"
        )
    if calibration_registry is not None:
        model_versions["detector_calibration"] = calibration_registry.calibration_version

    return Report(
        report_version=REPORT_VERSION,
        calibration_version=calibration.calibration_version,
        verdict=decision.verdict,
        confidence=decision.confidence,
        reasons=decision.reasons,
        evidence=evidence,
        model_versions=model_versions,
    )


def _detector_result(
    artifact: StoredArtifact,
    assessment: FaceAssessment,
    baseline_detector: BaselineDetector | None,
) -> BaselineDetectorResult:
    """Never let optional detector availability turn a safe report into a job failure."""
    if assessment.inconclusive_reasons:
        return BaselineDetectorResult.skipped()
    if baseline_detector is None:
        return BaselineDetectorResult.unavailable("inference_service_not_configured")
    try:
        return baseline_detector.score_primary_face(artifact, assessment)
    except Exception:
        return BaselineDetectorResult.unavailable("inference_service_unreachable")


def _detector_reasons(result: BaselineDetectorResult) -> tuple[str, ...]:
    if result.status is BaselineDetectorStatus.INVALID_RESPONSE:
        return ("baseline_detector_response_invalid",)
    return ("baseline_detector_unavailable",)


def _final_decision(
    assessment: FaceAssessment,
    detector_result: BaselineDetectorResult,
    calibration: CalibrationApplication,
) -> VerdictDecision:
    """Keep quality failures and unavailable evidence ahead of score policy."""
    if assessment.inconclusive_reasons:
        return VerdictDecision(Verdict.INCONCLUSIVE, None, assessment.inconclusive_reasons)
    if detector_result.status is not BaselineDetectorStatus.AVAILABLE:
        return VerdictDecision(Verdict.INCONCLUSIVE, None, _detector_reasons(detector_result))
    if calibration.status is CalibrationStatus.NOT_CONFIGURED:
        return VerdictDecision(
            Verdict.INCONCLUSIVE,
            None,
            ("detector_calibration_not_configured",),
        )
    if calibration.status is not CalibrationStatus.APPLIED or calibration.score is None:
        return VerdictDecision(
            Verdict.INCONCLUSIVE,
            None,
            ("detector_calibration_not_applicable",),
        )
    return decide_final_verdict((calibration.score,))


def _detector_evidence(result: BaselineDetectorResult) -> Evidence:
    if result.status is BaselineDetectorStatus.AVAILABLE:
        assert result.score is not None
        assert result.detector_version is not None
        assert result.latency_ms is not None
        return Evidence(
            source="baseline_synthetic_detector",
            status=result.status.value,
            score=result.score,
            version=result.detector_version,
            detail=(
                f"The baseline detector returned a {result.score:.3f} synthetic-portrait probability "
                f"in {result.latency_ms:.3f} ms. This raw probability is detector evidence, "
                "not an authenticity verdict."
            ),
        )
    if result.status is BaselineDetectorStatus.SKIPPED:
        detail = "The baseline detector was not run because face-quality gates require an inconclusive result."
    elif result.status is BaselineDetectorStatus.INVALID_RESPONSE:
        detail = (
            "The baseline detector returned an invalid response, so no detector probability was used. "
            "The report remains inconclusive."
        )
    else:
        detail = (
            "The baseline detector is unavailable, so no detector probability was used. "
            "The report remains inconclusive."
        )
    return Evidence(
        source="baseline_synthetic_detector",
        status=result.status.value,
        detail=detail,
        version=BASELINE_ADAPTER_VERSION,
    )


def _calibration_evidence(calibration: CalibrationApplication) -> Evidence:
    """Expose policy readiness without exposing deployment paths or calibration data."""
    if calibration.status is CalibrationStatus.APPLIED:
        assert calibration.score is not None
        score = calibration.score
        return Evidence(
            source="detector_calibration",
            status=calibration.status.value,
            score=score.calibrated_probability,
            version=calibration.calibration_version,
            detail=(
                f"Calibration {calibration.calibration_version} transformed the "
                f"{score.detector_id}@{score.detector_version} raw synthetic probability "
                f"from {score.raw_probability:.3f} to {score.calibrated_probability:.3f}. "
                "The result is probabilistic evidence, not proof of origin."
            ),
        )
    if calibration.status is CalibrationStatus.NOT_CONFIGURED:
        detail = (
            "No validated calibration artifact is configured for detector evidence, "
            "so a raw detector probability cannot determine a verdict."
        )
    elif calibration.status is CalibrationStatus.NOT_APPLICABLE:
        detail = (
            "The configured calibration artifact does not apply to this exact detector release, "
            "so a raw detector probability cannot determine a verdict."
        )
    else:
        detail = (
            "Calibration was not evaluated because no usable detector probability was available."
        )
    return Evidence(
        source="detector_calibration",
        status=calibration.status.value,
        detail=detail,
        version=calibration.calibration_version,
    )
