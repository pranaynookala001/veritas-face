"""Narrow HTTP hardening primitives for the public upload boundary.

The v1 API is intentionally single-process, so its upload rate limit is a
best-effort per-process guard rather than a distributed quota.  It is still
useful for bounding accidental or abusive local-demo traffic.  A deployment
that runs multiple API processes must put an equivalent shared limit at its
edge before treating the limit as global.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from threading import RLock
from time import monotonic
from typing import Callable
from uuid import uuid4

from starlette.datastructures import MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send


LOGGER = logging.getLogger("veritas_face.api")


def _configure_structured_logger() -> None:
    """Ensure application events remain JSON lines under Uvicorn's default logging."""
    if not LOGGER.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(message)s"))
        LOGGER.addHandler(handler)
    LOGGER.setLevel(logging.INFO)
    LOGGER.propagate = False


_configure_structured_logger()


@dataclass(frozen=True)
class RateLimitDecision:
    """The result of checking one upload attempt against a fixed window."""

    allowed: bool
    retry_after_seconds: int = 0


class FixedWindowRateLimiter:
    """Thread-safe, self-pruning fixed-window limiter for upload requests."""

    def __init__(
        self,
        *,
        limit: int,
        window_seconds: int,
        clock: Callable[[], float] = monotonic,
    ) -> None:
        if limit < 1:
            raise ValueError("limit must be at least one request")
        if window_seconds < 1:
            raise ValueError("window_seconds must be at least one second")
        self._limit = limit
        self._window_seconds = window_seconds
        self._clock = clock
        self._lock = RLock()
        self._windows: dict[str, tuple[float, int]] = {}

    def allow(self, client_key: str) -> RateLimitDecision:
        """Consume one request slot, returning a safe retry delay when full."""
        now = self._clock()
        with self._lock:
            self._discard_expired_locked(now)
            started_at, used = self._windows.get(client_key, (now, 0))
            elapsed = now - started_at
            if elapsed >= self._window_seconds:
                started_at, used, elapsed = now, 0, 0
            if used >= self._limit:
                retry_after = max(1, int(self._window_seconds - elapsed + 0.999))
                return RateLimitDecision(allowed=False, retry_after_seconds=retry_after)
            self._windows[client_key] = (started_at, used + 1)
            return RateLimitDecision(allowed=True)

    def _discard_expired_locked(self, now: float) -> None:
        for key, (started_at, _used) in tuple(self._windows.items()):
            if now - started_at >= self._window_seconds:
                del self._windows[key]


class RequestBodyTooLarge(Exception):
    """Internal signal raised when a streamed upload exceeds the request cap."""


class PublicApiHardeningMiddleware:
    """Apply size/rate guards, safe response headers, and privacy-safe access logs."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request_id = str(uuid4())
        started_at = monotonic()
        response_status: int | None = None
        route = _safe_route(scope)

        async def hardened_send(message: Message) -> None:
            nonlocal response_status
            if message["type"] == "http.response.start":
                response_status = int(message["status"])
                headers = MutableHeaders(scope=message)
                headers["Cache-Control"] = "no-store"
                headers["Permissions-Policy"] = "camera=(), geolocation=(), microphone=()"
                headers["Referrer-Policy"] = "no-referrer"
                headers["X-Content-Type-Options"] = "nosniff"
                headers["X-Frame-Options"] = "DENY"
                headers["X-Request-ID"] = request_id
            await send(message)

        try:
            if _is_upload_request(scope):
                max_request_bytes = int(scope["app"].state.max_upload_request_bytes)
                declared_length = _declared_content_length(scope)
                if declared_length is not None and declared_length > max_request_bytes:
                    await _error_response(
                        scope,
                        receive,
                        hardened_send,
                        status_code=413,
                        code="request_too_large",
                        message="Upload request exceeds the supported size limit.",
                    )
                    return

                limiter: FixedWindowRateLimiter = scope["app"].state.upload_rate_limiter
                decision = limiter.allow(_client_key(scope))
                if not decision.allowed:
                    await _error_response(
                        scope,
                        receive,
                        hardened_send,
                        status_code=429,
                        code="rate_limited",
                        message="Upload rate limit exceeded. Try again shortly.",
                        retry_after_seconds=decision.retry_after_seconds,
                    )
                    return

                received_bytes = 0

                async def limited_receive() -> Message:
                    nonlocal received_bytes
                    message = await receive()
                    if message["type"] == "http.request":
                        received_bytes += len(message.get("body", b""))
                        if received_bytes > max_request_bytes:
                            raise RequestBodyTooLarge
                    return message

                await self.app(scope, limited_receive, hardened_send)
            else:
                await self.app(scope, receive, hardened_send)
        except RequestBodyTooLarge:
            await _error_response(
                scope,
                receive,
                hardened_send,
                status_code=413,
                code="request_too_large",
                message="Upload request exceeds the supported size limit.",
            )
        finally:
            LOGGER.info(
                json.dumps(
                    {
                        "event": "http_request_completed",
                        "request_id": request_id,
                        "method": scope["method"],
                        "route": route,
                        "status_code": response_status or 500,
                        "duration_ms": round((monotonic() - started_at) * 1000, 3),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                )
            )


def _is_upload_request(scope: Scope) -> bool:
    return scope["method"] == "POST" and scope["path"] == "/v1/jobs"


def _safe_route(scope: Scope) -> str:
    path = scope["path"]
    if scope["method"] == "GET" and path.startswith("/v1/jobs/"):
        return "/v1/jobs/{job_id}"
    return path if path in {"/health", "/healthz", "/ready", "/readyz", "/v1/jobs", "/openapi.json"} else "other"


def _declared_content_length(scope: Scope) -> int | None:
    for name, value in scope.get("headers", []):
        if name.lower() != b"content-length":
            continue
        try:
            content_length = int(value)
        except ValueError:
            return None
        return content_length if content_length >= 0 else None
    return None


def _client_key(scope: Scope) -> str:
    """Use the direct peer address; forwarded headers are untrusted by default."""
    client = scope.get("client")
    return str(client[0]) if client else "unknown"


async def _error_response(
    scope: Scope,
    receive: Receive,
    send: Send,
    *,
    status_code: int,
    code: str,
    message: str,
    retry_after_seconds: int | None = None,
) -> None:
    headers = {"Retry-After": str(retry_after_seconds)} if retry_after_seconds else None
    response = JSONResponse(
        status_code=status_code,
        content={"error": {"code": code, "message": message, "issues": []}},
        headers=headers,
    )
    await response(scope, receive, send)
