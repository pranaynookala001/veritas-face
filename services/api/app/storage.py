"""Temporary, private artifact storage and the v1 job state machine."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import RLock
from uuid import UUID, uuid4

from .domain import JobStatus, Report


_ALLOWED_TRANSITIONS: dict[JobStatus, frozenset[JobStatus]] = {
    JobStatus.QUEUED: frozenset({JobStatus.PROCESSING, JobStatus.FAILED}),
    JobStatus.PROCESSING: frozenset({JobStatus.COMPLETED, JobStatus.FAILED}),
    JobStatus.COMPLETED: frozenset(),
    JobStatus.FAILED: frozenset(),
    JobStatus.EXPIRED: frozenset(),
}


class JobStateError(ValueError):
    """Raised when a worker requests a transition outside the job state machine."""


class JobExpiredError(JobStateError):
    """Raised when a worker tries to read an artifact after its retention window."""


@dataclass(frozen=True)
class StoredArtifact:
    """Private bytes and normalized media type made available only to workers."""

    content: bytes
    media_type: str


@dataclass(frozen=True)
class StoredJob:
    """Metadata retained after a job's artifact has been deleted."""

    job_id: UUID
    status: JobStatus
    created_at: datetime
    expires_at: datetime
    media_type: str
    artifact_path: Path | None
    report: Report | None = None


class TemporaryJobStore:
    """Thread-safe local storage with a private directory and explicit retention cleanup."""

    def __init__(self, retention: timedelta, *, directory: Path | None = None) -> None:
        if retention <= timedelta():
            raise ValueError("retention must be positive")

        self._retention = retention
        self._lock = RLock()
        self._jobs: dict[UUID, StoredJob] = {}
        self._temporary_directory: TemporaryDirectory[str] | None = None

        if directory is None:
            self._temporary_directory = TemporaryDirectory(prefix="veritas-face-")
            self._directory = Path(self._temporary_directory.name)
        else:
            self._directory = directory
            self._directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            self._directory.chmod(0o700)

    def readiness_error(self) -> str | None:
        """Return a non-sensitive reason when the private artifact store is unusable."""
        try:
            details = self._directory.stat()
        except OSError:
            return "artifact_directory_unavailable"
        if not self._directory.is_dir():
            return "artifact_directory_unavailable"
        if details.st_mode & 0o077:
            return "artifact_directory_permissions_invalid"
        if not os.access(self._directory, os.W_OK | os.X_OK):
            return "artifact_directory_unavailable"
        return None

    def create_job(
        self,
        artifact: StoredArtifact,
        *,
        now: datetime | None = None,
    ) -> StoredJob:
        """Persist validated bytes with a generated identifier, never a client filename."""
        if not artifact.content:
            raise ValueError("artifacts must not be empty")

        created_at = _as_utc(now or datetime.now(timezone.utc))
        with self._lock:
            self._cleanup_expired_locked(created_at)
            job_id = uuid4()
            artifact_path = self._directory / str(job_id)
            descriptor = os.open(artifact_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            try:
                with os.fdopen(descriptor, "wb") as artifact_file:
                    artifact_file.write(artifact.content)
            except BaseException:
                artifact_path.unlink(missing_ok=True)
                raise

            job = StoredJob(
                job_id=job_id,
                status=JobStatus.QUEUED,
                created_at=created_at,
                expires_at=created_at + self._retention,
                media_type=artifact.media_type,
                artifact_path=artifact_path,
            )
            self._jobs[job_id] = job
            return replace(job)

    def get_job(self, job_id: UUID, *, now: datetime | None = None) -> StoredJob | None:
        """Return metadata only; raw artifact bytes use ``get_artifact`` internally."""
        with self._lock:
            self._cleanup_expired_locked(_as_utc(now or datetime.now(timezone.utc)))
            job = self._jobs.get(job_id)
            return replace(job) if job else None

    def get_artifact(self, job_id: UUID, *, now: datetime | None = None) -> StoredArtifact:
        """Read a non-expired artifact for the worker pipeline."""
        with self._lock:
            self._cleanup_expired_locked(_as_utc(now or datetime.now(timezone.utc)))
            job = self._require_job(job_id)
            if job.status is JobStatus.EXPIRED or job.artifact_path is None:
                raise JobExpiredError("the job artifact has expired")
            try:
                content = job.artifact_path.read_bytes()
            except FileNotFoundError as exc:
                raise JobExpiredError("the job artifact is no longer available") from exc
            return StoredArtifact(content=content, media_type=job.media_type)

    def transition(
        self,
        job_id: UUID,
        target: JobStatus,
        *,
        now: datetime | None = None,
    ) -> StoredJob:
        """Move a job through its monotonic v1 lifecycle."""
        with self._lock:
            self._cleanup_expired_locked(_as_utc(now or datetime.now(timezone.utc)))
            job = self._require_job(job_id)
            if target not in _ALLOWED_TRANSITIONS[job.status]:
                raise JobStateError(f"cannot transition {job.status.value} job to {target.value}")
            updated = replace(job, status=target)
            self._jobs[job_id] = updated
            return replace(updated)

    def complete_with_report(
        self,
        job_id: UUID,
        report: Report,
        *,
        now: datetime | None = None,
    ) -> StoredJob:
        """Atomically move a queued worker job through processing to a completed report."""
        report.validate()
        with self._lock:
            self._cleanup_expired_locked(_as_utc(now or datetime.now(timezone.utc)))
            job = self._require_job(job_id)
            if job.status is not JobStatus.QUEUED:
                raise JobStateError(f"cannot attach a report to {job.status.value} job")
            processing = replace(job, status=JobStatus.PROCESSING)
            completed = replace(processing, status=JobStatus.COMPLETED, report=report)
            self._jobs[job_id] = completed
            return replace(completed)

    def fail_job(self, job_id: UUID, *, now: datetime | None = None) -> StoredJob:
        """Record an internal worker failure without exposing implementation details publicly."""
        with self._lock:
            self._cleanup_expired_locked(_as_utc(now or datetime.now(timezone.utc)))
            job = self._require_job(job_id)
            if job.status not in {JobStatus.QUEUED, JobStatus.PROCESSING}:
                raise JobStateError(f"cannot fail {job.status.value} job")
            failed = replace(job, status=JobStatus.FAILED)
            self._jobs[job_id] = failed
            return replace(failed)

    def cleanup_expired(self, *, now: datetime | None = None) -> int:
        """Delete expired artifacts and retain only an ``expired`` metadata tombstone."""
        with self._lock:
            return self._cleanup_expired_locked(_as_utc(now or datetime.now(timezone.utc)))

    def close(self) -> None:
        """Remove all artifacts when the local process shuts down."""
        with self._lock:
            for job in self._jobs.values():
                self._delete_artifact(job)
            self._jobs.clear()
            if self._temporary_directory is not None:
                self._temporary_directory.cleanup()
                self._temporary_directory = None

    def _cleanup_expired_locked(self, now: datetime) -> int:
        expired_count = 0
        for job_id, job in tuple(self._jobs.items()):
            if job.status is JobStatus.EXPIRED or job.expires_at > now:
                continue
            self._delete_artifact(job)
            self._jobs[job_id] = replace(job, status=JobStatus.EXPIRED, artifact_path=None, report=None)
            expired_count += 1
        return expired_count

    def _require_job(self, job_id: UUID) -> StoredJob:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    @staticmethod
    def _delete_artifact(job: StoredJob) -> None:
        if job.artifact_path is not None:
            job.artifact_path.unlink(missing_ok=True)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("timestamps must include a timezone")
    return value.astimezone(timezone.utc)
