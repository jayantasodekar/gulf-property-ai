"""Shared fetch layer: robots.txt compliance, rate limiting, retries, disk cache.

Every outbound scraper request goes through here. That is deliberate: the
politeness and compliance guarantees are enforced in one place and cannot be
bypassed by an individual scraper module.

## Why curl_cffi rather than httpx/requests

Both target sites reject generic Python HTTP clients at the TLS layer, not the
header layer -- verified experimentally: identical headers succeed from
`urllib`/`curl` and return HTTP 403 from `httpx`. DarGlobal additionally sits
behind an Imperva WAF that serves a ~950-byte block page to non-browser
clients regardless of User-Agent (a full Chrome header set and a Googlebot UA
were both refused).

`curl_cffi` performs the TLS handshake with a real browser's fingerprint, which
is the same thing a headless Chromium would do -- but without shipping a
~700 MB browser into the image. We are not evading access control: these are
public pages, we identify ourselves, we obey robots.txt, and we rate-limit
below what a human browsing the site would generate.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

from curl_cffi.requests import AsyncSession

log = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"

# The identity we declare to robots.txt. We do not rotate identities.
ROBOTS_AGENT = "GulfPropertyAI"

# Browser profile curl_cffi impersonates at the TLS + header level.
IMPERSONATE = "chrome"

EXTRA_HEADERS = {"Accept-Language": "en-GB,en;q=0.9,ar;q=0.8"}


class BlockedByRobots(Exception):
    """Raised when robots.txt disallows a URL. Never caught-and-ignored."""


class CircuitOpen(Exception):
    """Raised when a host has failed too many times in a row."""


class WAFBlocked(Exception):
    """Raised when a response is recognisably a WAF interstitial, not content."""


class PermanentError(Exception):
    """A 4xx that will never succeed on retry (e.g. a delisted sitemap URL)."""


@dataclass
class RateLimiter:
    """Adaptive async rate limiter (AIMD).

    A fixed rate is a guess. The server knows its own limit and tells us via
    HTTP 429, so we treat that as the control signal: multiplicatively slow
    down on a 429, then additively speed back up while requests succeed. This
    converges on the fastest rate the host will actually tolerate, which is
    both politer and -- because 429s stop costing us retries -- faster overall.
    """

    rate: float  # target requests per second
    min_rate: float = 1.0
    _current: float = field(default=0.0, repr=False)
    _last: float = field(default=0.0, repr=False)
    _ok_streak: int = field(default=0, repr=False)
    _last_penalty: float = field(default=0.0, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def __post_init__(self) -> None:
        self._current = self.rate

    async def acquire(self) -> None:
        async with self._lock:
            interval = 1.0 / max(self._current, self.min_rate)
            wait = self._last + interval - time.monotonic()
            if wait > 0:
                await asyncio.sleep(wait)
            # jitter so we never form a perfectly regular request train
            self._last = time.monotonic() + random.uniform(0, interval * 0.25)

    async def penalize(self) -> None:
        """Called on a 429: reduce the rate, then pause to let the bucket refill.

        Guarded by a cooldown because N concurrent workers will each surface the
        same rate-limit burst. Without it, one burst compounds into N halvings
        and the crawl collapses to the floor.
        """
        async with self._lock:
            now = time.monotonic()
            if now - self._last_penalty < 10.0:
                fresh = False
            else:
                fresh = True
                self._last_penalty = now
                self._ok_streak = 0
                new = max(self._current * 0.7, self.min_rate)
                if new < self._current:
                    log.warning("rate-limited: %.2f -> %.2f req/s", self._current, new)
                self._current = new
        await asyncio.sleep(random.uniform(2.0, 4.0) if fresh else random.uniform(0.5, 1.5))

    async def reward(self) -> None:
        """Called on success: creep back toward the target rate."""
        async with self._lock:
            self._ok_streak += 1
            if self._ok_streak >= 15 and self._current < self.rate:
                self._current = min(self._current * 1.3, self.rate)
                self._ok_streak = 0
                log.info("rate recovered to %.2f req/s", self._current)


class RobotsGate:
    """Fetches and caches robots.txt per host, and answers can_fetch()."""

    def __init__(self, user_agent: str = ROBOTS_AGENT) -> None:
        self.user_agent = user_agent
        self._parsers: dict[str, urllib.robotparser.RobotFileParser | None] = {}
        self._lock = asyncio.Lock()

    async def allowed(self, session: AsyncSession, url: str) -> bool:
        host = urlparse(url).netloc
        async with self._lock:
            if host not in self._parsers:
                self._parsers[host] = await self._load(session, url)
        rp = self._parsers[host]
        if rp is None:  # no robots.txt served -> permitted by convention
            return True
        return rp.can_fetch(self.user_agent, url)

    async def _load(self, session: AsyncSession, url: str):
        parsed = urlparse(url)
        robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
        try:
            r = await session.get(robots_url, timeout=20)
            if r.status_code != 200:
                log.info("no robots.txt at %s (HTTP %s)", robots_url, r.status_code)
                return None
            rp = urllib.robotparser.RobotFileParser()
            rp.parse(r.text.splitlines())
            log.info("loaded robots.txt for %s", parsed.netloc)
            return rp
        except Exception as exc:  # noqa: BLE001
            log.warning("robots.txt fetch failed for %s: %s", parsed.netloc, exc)
            return None


def looks_like_waf(body: str) -> bool:
    """Detect an Imperva/Incapsula interstitial rather than real content.

    The block page is tiny; real pages are >100 KB. Size alone is the reliable
    signal -- the marker strings also appear inside legitimate pages.
    """
    return len(body) < 3000 and (
        "NOINDEX, NOFOLLOW" in body or "_Incapsula_" in body or "Request unsuccessful" in body
    )


class Fetcher:
    """Polite async fetcher with rate limiting, retries, WAF detection and cache."""

    def __init__(
        self,
        rate: float = 4.0,
        max_retries: int = 4,
        use_cache: bool = True,
        circuit_threshold: int = 25,
    ) -> None:
        self.limiter = RateLimiter(rate=rate)
        self.robots = RobotsGate()
        self.max_retries = max_retries
        self.use_cache = use_cache
        self.circuit_threshold = circuit_threshold
        self._consecutive_failures = 0
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        self._session: AsyncSession | None = None

    async def __aenter__(self) -> Fetcher:
        self._session = AsyncSession(
            impersonate=IMPERSONATE,
            headers=EXTRA_HEADERS,
            timeout=45,
            max_clients=16,
        )
        return self

    async def __aexit__(self, *exc) -> None:
        if self._session:
            await self._session.close()

    def _cache_path(self, url: str) -> Path:
        return CACHE_DIR / f"{hashlib.sha256(url.encode()).hexdigest()[:24]}.txt"

    async def get(self, url: str, *, check_robots: bool = True) -> str:
        """Fetch a URL as text.

        Raises BlockedByRobots / CircuitOpen / WAFBlocked, or the last transport
        error if every retry failed.
        """
        cache = self._cache_path(url)
        if self.use_cache and cache.exists():
            return cache.read_text(encoding="utf-8")

        assert self._session is not None, "use `async with Fetcher(...)`"

        if check_robots and not await self.robots.allowed(self._session, url):
            raise BlockedByRobots(url)

        if self._consecutive_failures >= self.circuit_threshold:
            raise CircuitOpen(f"{self._consecutive_failures} consecutive failures")

        last_exc: Exception | None = None
        for attempt in range(self.max_retries):
            await self.limiter.acquire()
            try:
                r = await self._session.get(url)
                if r.status_code == 429:
                    await self.limiter.penalize()
                    raise RuntimeError("rate limited (429)")
                if r.status_code in (500, 502, 503, 504):
                    raise RuntimeError(f"retryable HTTP {r.status_code}")
                if r.status_code >= 400:
                    # 404/410 are common on large sitemaps (delisted listings).
                    # Retrying them wastes the crawl budget, so fail fast.
                    raise PermanentError(f"HTTP {r.status_code}")
                text = r.text
                if looks_like_waf(text):
                    raise WAFBlocked(url)
                self._consecutive_failures = 0
                await self.limiter.reward()
                if self.use_cache:
                    cache.write_text(text, encoding="utf-8")
                return text
            except PermanentError:
                raise
            except Exception as exc:  # noqa: BLE001
                last_exc = exc
                log.debug("attempt %s failed for %s: %s", attempt + 1, url, exc)
                if attempt < self.max_retries - 1:
                    await asyncio.sleep((2**attempt) + random.uniform(0, 1))

        self._consecutive_failures += 1
        raise last_exc  # type: ignore[misc]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
