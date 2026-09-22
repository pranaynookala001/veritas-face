import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.baseline_detector import BaselineDetectorResult, BaselineDetectorStatus
from app.calibration import (
    CalibrationConfigurationError,
    CalibratedDetectorScore,
    CalibrationStatus,
    VerdictDecision,
    apply_calibration,
    calibration_registry_from_mapping,
    decide_final_verdict,
    load_calibration_registry,
)
from app.domain import Verdict


CHECKSUM = "sha256:" + "a" * 64


def calibration_artifact(*, version: str = "model-2026.09", family: str = "cnn-family") -> dict:
    return {
        "schema_version": "1.0",
        "calibration_version": "heldout-portraits-2026.09",
        "validation_manifest_sha256": CHECKSUM,
        "detectors": [
            {
                "id": "baseline-portrait",
                "version": version,
                "family": family,
                "points": [
                    {"raw_probability": 0, "calibrated_probability": 0.02},
                    {"raw_probability": 0.5, "calibrated_probability": 0.45},
                    {"raw_probability": 1, "calibrated_probability": 0.98},
                ],
            }
        ],
    }


def available_result(score: float = 0.9) -> BaselineDetectorResult:
    return BaselineDetectorResult(
        status=BaselineDetectorStatus.AVAILABLE,
        score=score,
        detector_id="baseline-portrait",
        detector_version="model-2026.09",
        latency_ms=4.0,
    )


class CalibrationRegistryTests(unittest.TestCase):
    def test_exact_release_calibration_interpolates_and_retains_family(self) -> None:
        registry = calibration_registry_from_mapping(calibration_artifact())

        application = registry.apply(available_result(0.75))

        self.assertEqual(application.status, CalibrationStatus.APPLIED)
        self.assertEqual(application.calibration_version, "heldout-portraits-2026.09")
        self.assertIsNotNone(application.score)
        assert application.score is not None
        self.assertEqual(application.score.detector_family, "cnn-family")
        self.assertAlmostEqual(application.score.calibrated_probability, 0.715)

    def test_missing_or_mismatched_artifact_never_calibrates_a_detector_score(self) -> None:
        unconfigured = apply_calibration(available_result(), None)
        mismatched = calibration_registry_from_mapping(calibration_artifact(version="other-model")).apply(
            available_result()
        )
        malformed = calibration_registry_from_mapping(calibration_artifact()).apply(
            BaselineDetectorResult(
                status=BaselineDetectorStatus.AVAILABLE,
                score=float("nan"),
                detector_id="baseline-portrait",
                detector_version="model-2026.09",
                latency_ms=4.0,
            )
        )

        self.assertEqual(unconfigured.status, CalibrationStatus.NOT_CONFIGURED)
        self.assertEqual(mismatched.status, CalibrationStatus.NOT_APPLICABLE)
        self.assertIsNone(mismatched.score)
        self.assertEqual(malformed.status, CalibrationStatus.NOT_APPLICABLE)

    def test_rejects_non_monotonic_and_untraceable_calibration_artifacts(self) -> None:
        non_monotonic = calibration_artifact()
        non_monotonic["detectors"][0]["points"][2]["calibrated_probability"] = 0.2
        untraceable = calibration_artifact()
        untraceable["validation_manifest_sha256"] = "not-a-checksum"

        with self.assertRaisesRegex(CalibrationConfigurationError, "must not decrease"):
            calibration_registry_from_mapping(non_monotonic)
        with self.assertRaisesRegex(CalibrationConfigurationError, "sha256"):
            calibration_registry_from_mapping(untraceable)

    def test_loads_only_bounded_valid_json_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "calibration.json"
            path.write_text(json.dumps(calibration_artifact()), encoding="utf-8")

            registry = load_calibration_registry(path)

        self.assertEqual(registry.validation_manifest_sha256, CHECKSUM)


class FinalVerdictPolicyTests(unittest.TestCase):
    def test_decisive_consensus_generates_probabilistic_verdicts(self) -> None:
        synthetic = decide_final_verdict(
            (
                score("cnn-family", 0.94),
                score("spectral-family", 0.89),
            )
        )
        authentic = decide_final_verdict((score("cnn-family", 0.08),))

        self.assertEqual(
            synthetic,
            VerdictDecision(Verdict.LIKELY_SYNTHETIC, 0.915, ("calibrated_synthetic_evidence",)),
        )
        self.assertEqual(
            authentic,
            VerdictDecision(Verdict.LIKELY_AUTHENTIC, 0.08, ("calibrated_authentic_evidence",)),
        )

    def test_material_independent_detector_disagreement_forces_inconclusive(self) -> None:
        decision = decide_final_verdict(
            (
                score("cnn-family", 0.94),
                score("spectral-family", 0.74),
            )
        )

        self.assertEqual(decision.verdict, Verdict.INCONCLUSIVE)
        self.assertIsNone(decision.confidence)
        self.assertEqual(decision.reasons, ("meaningful_detector_disagreement",))

    def test_thresholds_use_unrounded_consensus(self) -> None:
        just_below_synthetic = decide_final_verdict((score("cnn-family", 0.8496),))
        just_above_authentic = decide_final_verdict((score("cnn-family", 0.1504),))

        self.assertEqual(just_below_synthetic.verdict, Verdict.INCONCLUSIVE)
        self.assertEqual(just_above_authentic.verdict, Verdict.INCONCLUSIVE)

    def test_related_detector_releases_do_not_outvote_an_independent_family(self) -> None:
        decision = decide_final_verdict(
            (
                score("cnn-family", 0.9, detector_version="release-a"),
                score("cnn-family", 0.8, detector_version="release-b"),
                score("spectral-family", 0.4),
            )
        )

        self.assertEqual(decision.verdict, Verdict.INCONCLUSIVE)
        self.assertEqual(decision.reasons, ("meaningful_detector_disagreement",))


def score(
    family: str,
    probability: float,
    *,
    detector_version: str = "model-2026.09",
) -> CalibratedDetectorScore:
    return CalibratedDetectorScore(
        detector_id="detector",
        detector_version=detector_version,
        detector_family=family,
        raw_probability=probability,
        calibrated_probability=probability,
    )


if __name__ == "__main__":
    unittest.main()
