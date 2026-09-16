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


class JobAcceptedResponse(BaseModel):
    """Receipt returned after an image has passed intake validation."""

    model_config = ConfigDict(extra="forbid")

    job_id: UUID
    status: Literal["queued"] = "queued"
    expires_at: datetime

    @field_serializer("expires_at")
    def serialize_expiry(self, value: datetime) -> str:
        """Keep the public contract in unambiguous UTC RFC 3339 form."""
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


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
