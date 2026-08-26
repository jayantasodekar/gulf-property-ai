"""OpenRouter client with live model discovery and an ordered fallback chain.

Free-tier models are rate-limited and periodically retired. A single hardcoded
model id is therefore a guaranteed outage. Instead we:

1. ask OpenRouter at startup which of our candidates actually exist AND
   advertise tool-calling support, and
2. walk that list on 429/5xx/timeout so one exhausted model degrades to the
   next rather than to an error page.
"""

from __future__ import annotations

import json
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from .config import settings

log = logging.getLogger(__name__)


class NoModelAvailable(RuntimeError):
    """Every candidate model failed or no API key is configured."""


@dataclass
class ModelState:
    """Tracks per-model health so we skip models that just failed us."""

    model_id: str
    cooldown_until: float = 0.0
    failures: int = 0
    successes: int = 0
    disabled_reason: str = ""

    @property
    def available(self) -> bool:
        return not self.disabled_reason and time.monotonic() >= self.cooldown_until

    def penalize(self, seconds: float = 60.0) -> None:
        self.failures += 1
        self.cooldown_until = time.monotonic() + seconds

    def disable(self, reason: str) -> None:
        """Retire a model permanently.

        Some catalogue entries advertise `:free` + tool support but reject
        ordinary API calls with 403 ("only available on agentic harnesses").
        That is an entitlement, not a transient fault, so retrying it forever
        would burn a slot in the chain on every single request.
        """
        self.disabled_reason = reason
        log.warning("permanently disabling %s: %s", self.model_id, reason)

    def succeed(self) -> None:
        self.successes += 1
        self.failures = 0
        self.cooldown_until = 0.0


@dataclass
class OpenRouterClient:
    api_key: str = field(default_factory=lambda: settings.openrouter_api_key)
    base_url: str = field(default_factory=lambda: settings.openrouter_base_url)
    states: list[ModelState] = field(default_factory=list)
    discovered: bool = False
    # OpenRouter meters free models per ACCOUNT per day, not per model. When
    # that cap is hit every `:free` model returns 429 simultaneously, so
    # rotating through the chain is pointless and just wastes latency on a
    # user request. We record it once and skip straight to search mode until
    # the quota resets.
    account_quota_until: float = 0.0

    @property
    def account_quota_exhausted(self) -> bool:
        return time.monotonic() < self.account_quota_until

    def _note_429(self, body: str, state: ModelState) -> None:
        if "free-models-per-day" in body or "per-day" in body:
            self.account_quota_until = time.monotonic() + 900  # re-probe in 15 min
            log.warning(
                "OpenRouter free-tier DAILY quota exhausted (account-wide). "
                "Serving search mode until it resets; adding credits raises the cap."
            )
            for st in self.states:
                st.penalize(900)
        else:
            state.penalize(90)

    def __post_init__(self) -> None:
        if not self.states:
            self.states = [ModelState(m) for m in settings.models]

    @property
    def enabled(self) -> bool:
        return bool(self.api_key.strip())

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            # OpenRouter uses these for attribution on the free tier.
            "HTTP-Referer": settings.public_url,
            "X-Title": settings.app_name,
        }

    async def discover(self) -> list[str]:
        """Intersect our candidate list with models that are live AND tool-capable."""
        if not self.enabled:
            log.warning("OPENROUTER_API_KEY not set - running in search-only mode")
            return []
        try:
            async with httpx.AsyncClient(timeout=25.0) as c:
                r = await c.get(
                    f"{self.base_url}/models",
                    params={"supported_parameters": "tools"},
                    headers=self._headers(),
                )
                r.raise_for_status()
                live = {m["id"] for m in r.json().get("data", [])}
        except Exception as exc:  # noqa: BLE001
            log.warning("model discovery failed (%s); using configured list as-is", exc)
            self.discovered = True
            return [s.model_id for s in self.states]

        keep = [s for s in self.states if s.model_id in live]
        missing = [s.model_id for s in self.states if s.model_id not in live]
        if missing:
            log.warning("configured models not available: %s", missing)
        if not keep:
            # Nothing we planned for is live: fall back to any free tool model.
            free = sorted(m for m in live if m.endswith(":free"))
            log.warning("no configured model live; discovered %d free ones", len(free))
            keep = [ModelState(m) for m in free[:6]]
        self.states = keep
        self.discovered = True
        log.info("active model chain: %s", [s.model_id for s in self.states])
        return [s.model_id for s in self.states]

    def _next_model(self) -> ModelState | None:
        for s in self.states:
            if s.available:
                return s
        return None

    def status(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "discovered": self.discovered,
            "account_quota_exhausted": self.account_quota_exhausted,
            "models": [
                {
                    "id": s.model_id,
                    "available": s.available,
                    "failures": s.failures,
                    "successes": s.successes,
                    "disabled": s.disabled_reason or None,
                }
                for s in self.states
            ],
        }

    # ------------------------------------------------------------------ #
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        temperature: float = 0.2,
        max_tokens: int = 1200,
    ) -> dict:
        """Non-streaming completion. Returns the assistant message dict."""
        if not self.enabled:
            raise NoModelAvailable("no API key configured")
        if self.account_quota_exhausted:
            raise NoModelAvailable("free-tier daily quota exhausted")

        last_error: Exception | None = None
        for _ in range(len(self.states)):
            state = self._next_model()
            if state is None:
                break
            payload: dict[str, Any] = {
                "model": state.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
            if tools:
                payload["tools"] = tools
                payload["tool_choice"] = "auto"
            try:
                async with httpx.AsyncClient(timeout=settings.request_timeout) as c:
                    r = await c.post(
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    )
                if r.status_code == 429:
                    self._note_429(r.text, state)
                    last_error = RuntimeError("HTTP 429")
                    if self.account_quota_exhausted:
                        break
                    continue
                if r.status_code in (401, 403):
                    state.disable(f"HTTP {r.status_code}: {r.text[:120]}")
                    last_error = RuntimeError(f"HTTP {r.status_code}")
                    continue
                if r.status_code >= 400:
                    log.warning("%s HTTP %s: %s", state.model_id, r.status_code, r.text[:200])
                    state.penalize(45)
                    last_error = RuntimeError(f"HTTP {r.status_code}")
                    continue
                data = r.json()
                choices = data.get("choices") or []
                if not choices:
                    state.penalize(30)
                    last_error = RuntimeError("empty choices")
                    continue
                state.succeed()
                msg = choices[0].get("message") or {}
                msg["_model"] = state.model_id
                return msg
            except Exception as exc:  # noqa: BLE001
                log.warning("%s failed: %s", state.model_id, exc)
                state.penalize(45)
                last_error = exc

        raise NoModelAvailable(f"all models exhausted: {last_error}")

    async def stream(
        self,
        messages: list[dict],
        temperature: float = 0.3,
        max_tokens: int = 1200,
    ) -> AsyncIterator[str]:
        """Stream assistant text deltas. Used for the final answer turn."""
        if not self.enabled:
            raise NoModelAvailable("no API key configured")
        if self.account_quota_exhausted:
            raise NoModelAvailable("free-tier daily quota exhausted")

        last_error: Exception | None = None
        for _ in range(len(self.states)):
            state = self._next_model()
            if state is None:
                break
            payload = {
                "model": state.model_id,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": True,
            }
            try:
                async with httpx.AsyncClient(timeout=settings.request_timeout) as c:
                    async with c.stream(
                        "POST",
                        f"{self.base_url}/chat/completions",
                        headers=self._headers(),
                        json=payload,
                    ) as r:
                        if r.status_code == 429:
                            body = (await r.aread()).decode("utf-8", "ignore")
                            self._note_429(body, state)
                            last_error = RuntimeError("HTTP 429")
                            if self.account_quota_exhausted:
                                break
                            continue
                        if r.status_code in (401, 403):
                            body = (await r.aread()).decode("utf-8", "ignore")[:120]
                            state.disable(f"HTTP {r.status_code}: {body}")
                            last_error = RuntimeError(f"HTTP {r.status_code}")
                            continue
                        if r.status_code >= 400:
                            body = (await r.aread()).decode("utf-8", "ignore")[:200]
                            log.warning("%s stream HTTP %s: %s", state.model_id, r.status_code, body)
                            state.penalize(45)
                            last_error = RuntimeError(f"HTTP {r.status_code}")
                            continue

                        emitted = False
                        async for line in r.aiter_lines():
                            if not line or not line.startswith("data:"):
                                continue
                            chunk = line[5:].strip()
                            if chunk == "[DONE]":
                                break
                            try:
                                delta = (
                                    json.loads(chunk)["choices"][0]
                                    .get("delta", {})
                                    .get("content")
                                )
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue
                            if delta:
                                emitted = True
                                yield delta
                        if emitted:
                            state.succeed()
                            return
                        state.penalize(30)
                        last_error = RuntimeError("stream produced no content")
            except Exception as exc:  # noqa: BLE001
                log.warning("%s stream failed: %s", state.model_id, exc)
                state.penalize(45)
                last_error = exc

        raise NoModelAvailable(f"all models exhausted: {last_error}")


client = OpenRouterClient()
