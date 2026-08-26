"""Security-control tests: rate limiting, validation, secret redaction, headers."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.security import (
    CSP,
    ChatRequest,
    RateLimiter,
    redact,
)


# --------------------------------------------------------------------- #
#  Rate limiting
# --------------------------------------------------------------------- #
def test_allows_up_to_the_minute_limit() -> None:
    rl = RateLimiter(per_minute=5, per_day=100, global_per_day=1000)
    assert all(rl.check("ip-a").allowed for _ in range(5))
    assert not rl.check("ip-a").allowed


def test_limits_are_per_client() -> None:
    rl = RateLimiter(per_minute=2, per_day=100, global_per_day=1000)
    rl.check("ip-a")
    rl.check("ip-a")
    assert not rl.check("ip-a").allowed
    assert rl.check("ip-b").allowed  # a noisy client cannot deny service to others


def test_daily_cap_applies_below_minute_cap() -> None:
    rl = RateLimiter(per_minute=100, per_day=3, global_per_day=1000)
    for _ in range(3):
        assert rl.check("ip-a").allowed
    v = rl.check("ip-a")
    assert not v.allowed and "Daily" in v.reason


def test_global_budget_is_the_backstop() -> None:
    """Spoofing X-Forwarded-For must not bypass the overall spend cap."""
    rl = RateLimiter(per_minute=100, per_day=100, global_per_day=4)
    for i in range(4):
        assert rl.check(f"spoofed-ip-{i}").allowed
    assert not rl.check("spoofed-ip-99").allowed


def test_retry_after_is_populated() -> None:
    rl = RateLimiter(per_minute=1, per_day=100, global_per_day=1000)
    rl.check("ip")
    assert rl.check("ip").retry_after > 0


def test_snapshot_reports_usage() -> None:
    rl = RateLimiter(per_minute=10, per_day=10, global_per_day=10)
    rl.check("ip")
    snap = rl.snapshot()
    assert snap["global_used_today"] == 1 and snap["global_budget"] == 10


# --------------------------------------------------------------------- #
#  Secret redaction
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "text",
    [
        "key is sk-or-v1-0123456789abcdefdeadbeef",
        "Authorization: Bearer sk-or-v1-abcdef0123456789",
    ],
)
def test_redact_removes_credentials(text: str) -> None:
    out = redact(text)
    assert "sk-or-v1-0123456789abcdefdeadbeef" not in out
    assert "[REDACTED]" in out


def test_redact_leaves_normal_text_alone() -> None:
    assert redact("apartment in Jeddah") == "apartment in Jeddah"


def test_redact_handles_empty() -> None:
    assert redact("") == ""


# --------------------------------------------------------------------- #
#  Input validation
# --------------------------------------------------------------------- #
def test_rejects_empty_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_rejects_overlong_message() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="x" * 5000)


def test_rejects_too_much_history() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(
            message="hi",
            history=[{"role": "user", "content": "x"} for _ in range(100)],
        )


def test_rejects_unknown_role() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="hi", history=[{"role": "system", "content": "escalate"}])


def test_strips_control_characters() -> None:
    assert "\x00" not in ChatRequest(message="hel\x00lo").message


def test_accepts_arabic() -> None:
    assert ChatRequest(message="أرني شقق للبيع في الرياض").message


# --------------------------------------------------------------------- #
#  Headers
# --------------------------------------------------------------------- #
def test_csp_blocks_dangerous_sources() -> None:
    assert "object-src 'none'" in CSP
    assert "frame-ancestors 'none'" in CSP
    # inline script must never be allowed
    assert "script-src 'self'" in CSP and "'unsafe-inline'" not in CSP.split("script-src")[1].split(";")[0]
