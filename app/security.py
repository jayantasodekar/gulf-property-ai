"""Security controls: rate limiting, budget caps, headers, input validation.

The app is deliberately unauthenticated (it is a public demo), which makes
abuse control the primary concern: the OpenRouter key behind it is a real,
if free-tier, resource. Controls are layered so that no single failure is
catastrophic:

  per-IP minute bucket -> per-IP daily cap -> global daily budget

The API key is additionally created with a $0 spend limit, so even total
compromise of this process cannot generate a bill.
"""

from __future__ import annotations

import logging
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field

from fastapi import Request
from pydantic import BaseModel, Field, field_validator
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from .config import settings

log = logging.getLogger(__name__)

# Values that must never reach a log line or an error response.
SECRET_RE = re.compile(r"(sk-or-[a-zA-Z0-9\-_]{8,}|Bearer\s+[A-Za-z0-9\-._~+/]{10,})")


def redact(text: str) -> str:
    return SECRET_RE.sub("[REDACTED]", text or "")


class RedactingFilter(logging.Filter):
    """Belt-and-braces: strip anything key-shaped from every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            if isinstance(record.msg, str):
                record.msg = redact(record.msg)
            if record.args:
                record.args = tuple(
                    redact(a) if isinstance(a, str) else a for a in record.args
                )  # type: ignore[assignment]
        except Exception:  # noqa: BLE001, S110
            # A logging filter must never raise: doing so would break the very
            # logging path used to report the original error. Failing open here
            # is safe because redaction is defence in depth -- the key is never
            # deliberately logged in the first place.
            pass
        return True


# --------------------------------------------------------------------------- #
#  Rate limiting
# --------------------------------------------------------------------------- #
@dataclass
class RateLimitResult:
    allowed: bool
    reason: str = ""
    retry_after: int = 60


@dataclass
class RateLimiter:
    per_minute: int = field(default_factory=lambda: settings.rate_limit_per_minute)
    per_day: int = field(default_factory=lambda: settings.rate_limit_per_day)
    global_per_day: int = field(default_factory=lambda: settings.global_daily_budget)

    _minute: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    _day: dict[str, deque] = field(default_factory=lambda: defaultdict(deque))
    _global: deque = field(default_factory=deque)

    def _prune(self, dq: deque, window: float, now: float) -> None:
        while dq and now - dq[0] > window:
            dq.popleft()

    def check(self, key: str) -> RateLimitResult:
        now = time.time()

        self._prune(self._global, 86400, now)
        if len(self._global) >= self.global_per_day:
            return RateLimitResult(False, "The daily service budget is exhausted.", 3600)

        minute = self._minute[key]
        self._prune(minute, 60, now)
        if len(minute) >= self.per_minute:
            return RateLimitResult(False, "Too many requests. Please slow down.", 60)

        day = self._day[key]
        self._prune(day, 86400, now)
        if len(day) >= self.per_day:
            return RateLimitResult(False, "Daily limit reached for this client.", 3600)

        minute.append(now)
        day.append(now)
        self._global.append(now)

        # Opportunistic cleanup so idle clients do not accumulate forever.
        if len(self._minute) > 4096:
            for k in [k for k, v in self._minute.items() if not v][:2048]:
                self._minute.pop(k, None)
                self._day.pop(k, None)
        return RateLimitResult(True)

    def snapshot(self) -> dict:
        now = time.time()
        self._prune(self._global, 86400, now)
        return {
            "global_used_today": len(self._global),
            "global_budget": self.global_per_day,
            "tracked_clients": len(self._day),
            "per_minute": self.per_minute,
            "per_day": self.per_day,
        }


limiter = RateLimiter()


def client_key(request: Request) -> str:
    """Identify the caller.

    Behind the Hugging Face Spaces proxy the socket peer is always the proxy,
    so we take the left-most X-Forwarded-For entry. That header is client-
    controlled and therefore spoofable; it is fine for abuse *dampening* but is
    never treated as an identity, and the global budget is the real backstop.
    """
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()[:64]
    return (request.client.host if request.client else "unknown")[:64]


# --------------------------------------------------------------------------- #
#  Headers
# --------------------------------------------------------------------------- #
CSP = (
    "default-src 'self'; "
    "img-src 'self' data: https://imagedelivery.net https://cdn.darglobal.co.uk; "
    "style-src 'self' 'unsafe-inline'; "
    "script-src 'self'; "
    "connect-src 'self'; "
    "font-src 'self' data:; "
    "object-src 'none'; base-uri 'none'; frame-ancestors 'none'"
)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next) -> Response:
        response = await call_next(request)
        response.headers.setdefault("Content-Security-Policy", CSP)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "geolocation=(), microphone=(), camera=()"
        )
        response.headers.setdefault(
            "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
        )
        return response


class BodySizeLimitMiddleware(BaseHTTPMiddleware):
    """Reject oversized bodies before they are parsed."""

    MAX_BYTES = 32 * 1024

    async def dispatch(self, request: Request, call_next) -> Response:
        cl = request.headers.get("content-length")
        if cl and cl.isdigit() and int(cl) > self.MAX_BYTES:
            from starlette.responses import JSONResponse

            return JSONResponse({"error": "Request body too large."}, status_code=413)
        return await call_next(request)


# --------------------------------------------------------------------------- #
#  Input validation
# --------------------------------------------------------------------------- #
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


class Turn(BaseModel):
    role: str = Field(pattern="^(user|assistant)$")
    content: str = Field(max_length=4000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=settings.max_message_chars)
    history: list[Turn] = Field(default_factory=list, max_length=settings.max_history_turns)

    @field_validator("message")
    @classmethod
    def _clean(cls, v: str) -> str:
        v = CONTROL_CHARS.sub("", v).strip()
        if not v:
            raise ValueError("message must not be empty")
        return v
