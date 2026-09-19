"""HTTP schemas shared by the FastAPI routes and their OpenAPI contract."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_serializer


class HealthResponse(BaseModel):
    """Liveness response for the API process."""

    status: Literal["ok"] = "ok"
    service: Literal["veritas-face-api"] = "veritas-face-api"
    version: str


class EvidenceResponse(BaseModel):
    """One non-sensitive, versioned source used to construct a report."""

    model_config = ConfigDict(extra="forbid")

    source: str
    status: str
    detail: str
    version: str | None = None
    score: float | None = Field(default=None, ge=0, le=1)


class ReportResponse(BaseModel):
    """Completed evidence report without image bytes or identifying metadata."""

    model_config = ConfigDict(extra="forbid")

    report_version: str
    calibration_version: str
    verdict: Literal["likely_synthetic", "likely_authentic", "inconclusive"]
    confidence: float | None = Field(default=None, ge=0, le=1)
    reasons: list[str]
    evidence: list[EvidenceResponse]
    model_versions: dict[str, str]


class JobStatusResponse(BaseModel):
    """Public metadata for a temporary asynchronous job."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    status: Literal["queued", "processing", "completed", "failed", "expired"]
    expires_at: datetime

    @field_serializer("expires_at")
    def serialize_expiry(self, value: datetime) -> str:
        """Keep the public contract in unambiguous UTC RFC 3339 form."""
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class JobAcceptedResponse(JobStatusResponse):
    """Receipt returned after an image has passed intake validation."""

    status: Literal["queued"] = "queued"


class JobCompletedResponse(JobStatusResponse):
    """Completed job metadata together with the non-sensitive evidence report."""

    status: Literal["completed"] = "completed"
    report: ReportResponse


class ValidationIssue(BaseModel):
    """One machine-readable location and message for an invalid request."""

    model_config = ConfigDict(extra="forbid")

    field: str | None = None
    message: str
    type: str | None = None


class ErrorBody(BaseModel):
    """Stable error payload consumed by web and API clients."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable programmatic error code.")
    message: str
    issues: list[ValidationIssue] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    """Envelope used for all documented API errors."""

    model_config = ConfigDict(extra="forbid")

    error: ErrorBody
