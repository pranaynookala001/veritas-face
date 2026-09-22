"""FastAPI application for validated Veritas Face job intake."""

from __future__ import annotations

from datetime import timedelta
import os
from pathlib import Path
from typing import Annotated
from uuid import UUID

from fastapi import BackgroundTasks, Depends, FastAPI, File, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from .baseline_detector import BaselineDetector, OnnxInferenceBaselineDetector
from .calibration import (
    CalibrationConfigurationError,
    CalibrationRegistry,
    load_calibration_registry,
)
from .domain import Report
from .face_quality import assess_encoded_image
from .intake import UploadValidationError, validate_upload
from .schemas import (
    ErrorBody,
    ErrorResponse,
    HealthResponse,
    JobAcceptedResponse,
    JobCompletedResponse,
    JobStatusResponse,
    ReportResponse,
    ValidationIssue,
)
from .reports import build_local_report
from .storage import JobExpiredError, JobStateError, StoredArtifact, TemporaryJobStore


APP_VERSION = "0.1.0"
JOB_RETENTION = timedelta(hours=24)


def configured_job_store() -> TemporaryJobStore:
    """Create temporary storage, optionally at an operator-provided private mount.

    Containers can mount an owner-only volume at this path so accepted uploads
    survive an in-process background task without becoming part of the image.
    An empty setting deliberately retains the safe OS-managed temporary directory
    used by local development and unit tests.
    """
    configured_directory = os.getenv("VERITAS_FACE_ARTIFACT_DIRECTORY", "").strip()
    if configured_directory:
        return TemporaryJobStore(JOB_RETENTION, directory=Path(configured_directory))
    return TemporaryJobStore(JOB_RETENTION)


job_store = configured_job_store()
_inference_service_url = os.getenv("VERITAS_FACE_INFERENCE_URL", "").strip()
baseline_detector: BaselineDetector | None = (
    OnnxInferenceBaselineDetector(_inference_service_url) if _inference_service_url else None
)
_calibration_path = os.getenv("VERITAS_FACE_CALIBRATION_PATH", "").strip()
try:
    calibration_registry: CalibrationRegistry | None = (
        load_calibration_registry(_calibration_path) if _calibration_path else None
    )
except CalibrationConfigurationError:
    # A bad local artifact must remove verdict eligibility, not block safe reports.
    calibration_registry = None
WEB_ORIGINS = tuple(
    origin.strip()
    for origin in os.getenv("VERITAS_FACE_WEB_ORIGINS", "http://localhost:3000").split(",")
    if origin.strip()
)

app = FastAPI(
    title="Veritas Face API",
    version=APP_VERSION,
    description="Validated temporary intake, local quality checks, and evidence reports for synthetic portrait analysis.",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(WEB_ORIGINS),
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
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


def get_job_store() -> TemporaryJobStore:
    """Expose local storage as an overridable dependency for API and worker tests."""
    return job_store


def get_baseline_detector() -> BaselineDetector | None:
    """Expose the opt-in local inference adapter as an overridable dependency."""
    return baseline_detector


def get_calibration_registry() -> CalibrationRegistry | None:
    """Expose an optional validated calibration artifact for worker tests and routes."""
    return calibration_registry


def process_local_job(
    store: TemporaryJobStore,
    job_id: UUID,
    detector: BaselineDetector | None = None,
    calibration: CalibrationRegistry | None = None,
) -> None:
    """Run local quality/provenance analysis and retain only its safe report."""
    try:
        artifact = store.get_artifact(job_id)
        assessment = assess_encoded_image(artifact.content)
        report: Report = build_local_report(
            artifact,
            assessment,
            baseline_detector=detector,
            calibration_registry=calibration,
        )
        store.complete_with_report(job_id, report)
    except (JobExpiredError, JobStateError):
        return
    except Exception:
        try:
            store.fail_job(job_id)
        except (JobExpiredError, JobStateError):
            pass


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
    background_tasks: BackgroundTasks,
    store: Annotated[TemporaryJobStore, Depends(get_job_store)],
    detector: Annotated[BaselineDetector | None, Depends(get_baseline_detector)],
    calibration: Annotated[CalibrationRegistry | None, Depends(get_calibration_registry)],
) -> JobAcceptedResponse:
    """Validate an upload and issue the temporary job receipt.

    This endpoint intentionally keeps original filenames and image bytes out of the
    response. Bytes are stored only in a private temporary artifact for the worker.
    """
    validated_upload = await validate_upload(portrait)
    job = store.create_job(
        StoredArtifact(content=validated_upload.content, media_type=validated_upload.media_type),
    )
    background_tasks.add_task(process_local_job, store, job.job_id, detector, calibration)
    return JobAcceptedResponse(
        job_id=job.job_id,
        expires_at=job.expires_at,
    )


@app.get(
    "/v1/jobs/{job_id}",
    response_model=JobStatusResponse | JobCompletedResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["jobs"],
)
async def get_job(
    job_id: UUID,
    store: Annotated[TemporaryJobStore, Depends(get_job_store)],
) -> JobStatusResponse | JobCompletedResponse | JSONResponse:
    """Return non-sensitive job state after deleting artifacts past retention."""
    job = store.get_job(job_id)
    if job is None:
        return error_response(404, "job_not_found", "No job exists for this identifier.")
    response = {
        "job_id": job.job_id,
        "status": job.status.value,
        "expires_at": job.expires_at,
    }
    if job.report is not None:
        return JobCompletedResponse(
            **response,
            report=ReportResponse(**job.report.as_dict()),
        )
    return JobStatusResponse(
        **response,
    )
