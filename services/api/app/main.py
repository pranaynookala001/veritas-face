"""FastAPI application for validated Veritas Face job intake."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Annotated
from uuid import uuid4

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .intake import UploadValidationError, validate_upload
from .schemas import ErrorBody, ErrorResponse, HealthResponse, JobAcceptedResponse, ValidationIssue


APP_VERSION = "0.1.0"
JOB_RETENTION = timedelta(hours=24)

app = FastAPI(
    title="Veritas Face API",
    version=APP_VERSION,
    description="Validated, temporary intake for synthetic portrait analysis jobs.",
)


def error_response(
    status_code: int,
    code: str,
    message: str,
    *,
    issues: list[ValidationIssue] | None = None,
) -> JSONResponse:
    """Render the single error-envelope shape used by public endpoints."""
    payload = ErrorResponse(error=ErrorBody(code=code, message=message, issues=issues or []))
    return JSONResponse(status_code=status_code, content=payload.model_dump(mode="json"))


@app.exception_handler(UploadValidationError)
async def upload_validation_error_handler(_: Request, exc: UploadValidationError) -> JSONResponse:
    return error_response(
        exc.status_code,
        exc.code,
        exc.message,
        issues=[ValidationIssue(field=exc.field, message=exc.message, type=exc.code)],
    )


@app.exception_handler(RequestValidationError)
async def request_validation_error_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    issues = [
        ValidationIssue(
            field=".".join(str(part) for part in error["loc"] if part != "body") or None,
            message=error["msg"],
            type=error["type"],
        )
        for error in exc.errors()
    ]
    return error_response(
        422,
        "request_validation_failed",
        "The request could not be validated.",
        issues=issues,
    )


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    if exc.status_code == 404:
        code, message = "not_found", "The requested endpoint does not exist."
    elif exc.status_code == 405:
        code, message = "method_not_allowed", "This HTTP method is not supported for the endpoint."
    else:
        code, message = "http_error", str(exc.detail)
    return error_response(exc.status_code, code, message)


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Report that the HTTP process is available."""
    return HealthResponse(version=APP_VERSION)


@app.get("/healthz", response_model=HealthResponse, include_in_schema=False)
async def healthz() -> HealthResponse:
    """Compatibility liveness alias for container platforms."""
    return await health()


@app.post(
    "/v1/jobs",
    status_code=202,
    response_model=JobAcceptedResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
    },
    tags=["jobs"],
)
async def create_job(
    portrait: Annotated[
        UploadFile,
        File(description="One JPEG, PNG, or WebP portrait, no larger than 10 MiB."),
    ],
) -> JobAcceptedResponse:
    """Validate an upload and issue the temporary job receipt.

    This intake boundary intentionally keeps no original filename or image bytes in
    the response. Artifact persistence and status retrieval arrive with the job
    storage milestone.
    """
    await validate_upload(portrait)
    return JobAcceptedResponse(
        job_id=uuid4(),
        expires_at=datetime.now(timezone.utc) + JOB_RETENTION,
    )
