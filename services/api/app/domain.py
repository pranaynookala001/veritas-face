"""Stable, dependency-free evidence contract shared by API and workers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Verdict(str, Enum):
    LIKELY_SYNTHETIC = "likely_synthetic"
    LIKELY_AUTHENTIC = "likely_authentic"
    INCONCLUSIVE = "inconclusive"


class JobStatus(str, Enum):
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    EXPIRED = "expired"


@dataclass(frozen=True)
class Evidence:
    source: str
    status: str
    detail: str
    version: Optional[str] = None
    score: Optional[float] = None

    def as_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {"source": self.source, "status": self.status, "detail": self.detail}
        if self.version is not None:
            payload["version"] = self.version
        if self.score is not None:
            payload["score"] = self.score
        return payload


@dataclass(frozen=True)
class Report:
    verdict: Verdict
    confidence: Optional[float]
    reasons: tuple[str, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    report_version: str = "1.0"
    calibration_version: str = "not_available"
    model_versions: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.report_version.strip():
            raise ValueError("reports require a report version")
        if not self.calibration_version.strip():
            raise ValueError("reports require calibration information")
        if self.confidence is not None and not 0 <= self.confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if self.verdict is Verdict.INCONCLUSIVE and not self.reasons:
            raise ValueError("inconclusive reports require one or more reasons")
        for item in self.evidence:
            if item.score is not None and not 0 <= item.score <= 1:
                raise ValueError("evidence scores must be between 0 and 1")

    def as_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "report_version": self.report_version,
            "calibration_version": self.calibration_version,
            "verdict": self.verdict.value,
            "confidence": self.confidence,
            "reasons": list(self.reasons),
            "evidence": [item.as_dict() for item in self.evidence],
            "model_versions": self.model_versions,
        }
