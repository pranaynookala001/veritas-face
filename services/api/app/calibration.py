"""Versioned calibration artifacts and conservative final-verdict policy.

Detector output is only eligible for a verdict when a deployment supplies a
validated calibration artifact for that exact detector release.  Calibration
artifacts are deliberately small JSON documents so training can publish them
alongside an immutable held-out manifest, without embedding any portraits in
the API service.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import Enum
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .baseline_detector import BaselineDetectorResult, BaselineDetectorStatus
from .domain import Verdict


CALIBRATION_SCHEMA_VERSION = "1.0"
CALIBRATION_NOT_AVAILABLE = "not_available"
SYNTHETIC_VERDICT_THRESHOLD = 0.85
AUTHENTIC_VERDICT_THRESHOLD = 0.15
MEANINGFUL_DISAGREEMENT = 0.20


class CalibrationConfigurationError(ValueError):
    """Raised when a local calibration artifact is not safe to use."""


class CalibrationStatus(str, Enum):
    """Whether a detector score could be transformed by a trusted artifact."""

    APPLIED = "applied"
    NOT_CONFIGURED = "not_configured"
    NOT_APPLICABLE = "not_applicable"
    SKIPPED = "skipped"


@dataclass(frozen=True)
class CalibrationPoint:
    raw_probability: float
    calibrated_probability: float


@dataclass(frozen=True)
class DetectorCalibration:
    """Monotonic piecewise-linear calibration for one immutable detector release."""

    detector_id: str
    detector_version: str
    detector_family: str
    points: tuple[CalibrationPoint, ...]

    def calibrate(self, raw_probability: float) -> float:
        """Interpolate a bounded detector probability using the published curve."""
        for lower, upper in zip(self.points, self.points[1:]):
            if raw_probability <= upper.raw_probability:
                if raw_probability == lower.raw_probability:
                    return lower.calibrated_probability
                fraction = (raw_probability - lower.raw_probability) / (
                    upper.raw_probability - lower.raw_probability
                )
                return lower.calibrated_probability + fraction * (
                    upper.calibrated_probability - lower.calibrated_probability
                )
        return self.points[-1].calibrated_probability


@dataclass(frozen=True)
class CalibratedDetectorScore:
    """A score whose source release and calibration release are both known."""

    detector_id: str
    detector_version: str
    detector_family: str
    raw_probability: float
    calibrated_probability: float


@dataclass(frozen=True)
class CalibrationApplication:
    """Safe normalization of a calibration lookup for report assembly."""

    status: CalibrationStatus
    calibration_version: str
    score: CalibratedDetectorScore | None = None


@dataclass(frozen=True)
class CalibrationRegistry:
    """A validated, versioned set of release-specific calibration curves."""

    calibration_version: str
    validation_manifest_sha256: str
    detector_calibrations: tuple[DetectorCalibration, ...]

    def apply(self, result: BaselineDetectorResult) -> CalibrationApplication:
        """Apply only an exact release match; never guess a compatible calibration."""
        if result.status is not BaselineDetectorStatus.AVAILABLE:
            return CalibrationApplication(CalibrationStatus.SKIPPED, self.calibration_version)
        if (
            result.score is None
            or result.detector_id is None
            or result.detector_version is None
            or not _is_probability(result.score)
            or not isinstance(result.detector_id, str)
            or not result.detector_id.strip()
            or not isinstance(result.detector_version, str)
            or not result.detector_version.strip()
        ):
            return CalibrationApplication(CalibrationStatus.NOT_APPLICABLE, self.calibration_version)

        calibration = next(
            (
                item
                for item in self.detector_calibrations
                if item.detector_id == result.detector_id
                and item.detector_version == result.detector_version
            ),
            None,
        )
        if calibration is None:
            return CalibrationApplication(CalibrationStatus.NOT_APPLICABLE, self.calibration_version)

        return CalibrationApplication(
            status=CalibrationStatus.APPLIED,
            calibration_version=self.calibration_version,
            score=CalibratedDetectorScore(
                detector_id=result.detector_id,
                detector_version=result.detector_version,
                detector_family=calibration.detector_family,
                raw_probability=result.score,
                calibrated_probability=calibration.calibrate(result.score),
            ),
        )


@dataclass(frozen=True)
class VerdictDecision:
    """The explainable, policy-derived conclusion for calibrated evidence."""

    verdict: Verdict
    confidence: float | None
    reasons: tuple[str, ...]


def apply_calibration(
    result: BaselineDetectorResult,
    registry: CalibrationRegistry | None,
) -> CalibrationApplication:
    """Represent an absent artifact explicitly instead of silently using a score."""
    if registry is None:
        status = (
            CalibrationStatus.SKIPPED
            if result.status is not BaselineDetectorStatus.AVAILABLE
            else CalibrationStatus.NOT_CONFIGURED
        )
        return CalibrationApplication(status, CALIBRATION_NOT_AVAILABLE)
    return registry.apply(result)


def decide_final_verdict(scores: Sequence[CalibratedDetectorScore]) -> VerdictDecision:
    """Turn calibrated evidence into a cautious, explainable probabilistic verdict.

    Scores from the same detector family are averaged before aggregation so a
    family cannot gain extra influence merely by publishing closely related
    releases.  A spread of at least 0.20 between independent detector families
    is material disagreement and blocks a conclusion.
    """
    if not scores:
        return VerdictDecision(
            Verdict.INCONCLUSIVE,
            None,
            ("no_calibrated_detector_score",),
        )

    family_scores: dict[str, list[float]] = defaultdict(list)
    for score in scores:
        family_scores[score.detector_family].append(score.calibrated_probability)
    independent_scores = tuple(
        math.fsum(values) / len(values) for values in family_scores.values()
    )

    if (
        len(independent_scores) >= 2
        and (
            max(independent_scores) - min(independent_scores) >= MEANINGFUL_DISAGREEMENT
            or math.isclose(
                max(independent_scores) - min(independent_scores),
                MEANINGFUL_DISAGREEMENT,
                rel_tol=0,
                abs_tol=1e-12,
            )
        )
    ):
        return VerdictDecision(
            Verdict.INCONCLUSIVE,
            None,
            ("meaningful_detector_disagreement",),
        )

    consensus = math.fsum(independent_scores) / len(independent_scores)
    reported_consensus = round(consensus, 3)
    if consensus >= SYNTHETIC_VERDICT_THRESHOLD:
        return VerdictDecision(
            Verdict.LIKELY_SYNTHETIC,
            reported_consensus,
            ("calibrated_synthetic_evidence",),
        )
    if consensus <= AUTHENTIC_VERDICT_THRESHOLD:
        return VerdictDecision(
            Verdict.LIKELY_AUTHENTIC,
            reported_consensus,
            ("calibrated_authentic_evidence",),
        )
    return VerdictDecision(
        Verdict.INCONCLUSIVE,
        None,
        ("calibrated_detector_evidence_not_decisive",),
    )


def load_calibration_registry(path: str | Path) -> CalibrationRegistry:
    """Read a bounded local JSON artifact and reject any ambiguous structure."""
    artifact_path = Path(path)
    try:
        content = artifact_path.read_bytes()
    except OSError as error:
        raise CalibrationConfigurationError("calibration artifact could not be read") from error
    if len(content) > 256 * 1024:
        raise CalibrationConfigurationError("calibration artifact exceeds 256 KiB")
    try:
        payload = json.loads(content)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise CalibrationConfigurationError("calibration artifact is not valid JSON") from error
    return calibration_registry_from_mapping(payload)


def calibration_registry_from_mapping(payload: object) -> CalibrationRegistry:
    """Validate a calibration artifact loaded by a deployment or test fixture."""
    root = _mapping(payload, "calibration artifact")
    _exact_keys(
        root,
        {"schema_version", "calibration_version", "validation_manifest_sha256", "detectors"},
        "calibration artifact",
    )
    if _nonempty_string(root["schema_version"], "schema_version") != CALIBRATION_SCHEMA_VERSION:
        raise CalibrationConfigurationError(
            f"unsupported calibration schema version: {root['schema_version']!r}"
        )
    calibration_version = _nonempty_string(root["calibration_version"], "calibration_version")
    validation_manifest_sha256 = _sha256(
        root["validation_manifest_sha256"], "validation_manifest_sha256"
    )
    detectors = root["detectors"]
    if not isinstance(detectors, list) or not detectors:
        raise CalibrationConfigurationError("detectors must be a non-empty list")

    parsed = tuple(_detector_calibration(item) for item in detectors)
    releases = {(item.detector_id, item.detector_version) for item in parsed}
    if len(releases) != len(parsed):
        raise CalibrationConfigurationError("detector calibration releases must be unique")
    return CalibrationRegistry(calibration_version, validation_manifest_sha256, parsed)


def _detector_calibration(payload: object) -> DetectorCalibration:
    item = _mapping(payload, "detector calibration")
    _exact_keys(
        item,
        {"id", "version", "family", "points"},
        "detector calibration",
    )
    points_payload = item["points"]
    if not isinstance(points_payload, list) or len(points_payload) < 2:
        raise CalibrationConfigurationError("calibration points must contain at least two entries")
    points = tuple(_calibration_point(point) for point in points_payload)
    if points[0].raw_probability != 0 or points[-1].raw_probability != 1:
        raise CalibrationConfigurationError("calibration points must start at 0 and end at 1")
    for lower, upper in zip(points, points[1:]):
        if lower.raw_probability >= upper.raw_probability:
            raise CalibrationConfigurationError("calibration raw probabilities must strictly increase")
        if lower.calibrated_probability > upper.calibrated_probability:
            raise CalibrationConfigurationError("calibration probabilities must not decrease")
    return DetectorCalibration(
        detector_id=_nonempty_string(item["id"], "detector id"),
        detector_version=_nonempty_string(item["version"], "detector version"),
        detector_family=_nonempty_string(item["family"], "detector family"),
        points=points,
    )


def _calibration_point(payload: object) -> CalibrationPoint:
    point = _mapping(payload, "calibration point")
    _exact_keys(
        point,
        {"raw_probability", "calibrated_probability"},
        "calibration point",
    )
    return CalibrationPoint(
        raw_probability=_probability(point["raw_probability"], "raw_probability"),
        calibrated_probability=_probability(
            point["calibrated_probability"], "calibrated_probability"
        ),
    )


def _mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise CalibrationConfigurationError(f"{label} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise CalibrationConfigurationError(f"{label} keys must be strings")
    return value


def _exact_keys(value: Mapping[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise CalibrationConfigurationError(f"{label} has unexpected or missing fields")


def _nonempty_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CalibrationConfigurationError(f"{label} must be a non-empty string")
    return value.strip()


def _sha256(value: object, label: str) -> str:
    checksum = _nonempty_string(value, label)
    if not checksum.startswith("sha256:") or len(checksum) != 71:
        raise CalibrationConfigurationError(f"{label} must be a sha256 checksum")
    if any(character not in "0123456789abcdef" for character in checksum[7:].lower()):
        raise CalibrationConfigurationError(f"{label} must be a sha256 checksum")
    return checksum.lower()


def _probability(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise CalibrationConfigurationError(f"{label} must be a finite number")
    probability = float(value)
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise CalibrationConfigurationError(f"{label} must be between 0 and 1")
    return probability


def _is_probability(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and 0 <= float(value) <= 1
    )
