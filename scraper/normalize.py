"""Unified data model + sanitization for both sources.

Security note: scraped text is UNTRUSTED INPUT. It is sanitized here, at
ingest, before it can ever reach a prompt. That is the first of two layers;
the second is the delimiting done in app/prompts.py.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata
from datetime import UTC, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator

# Static, dated FX table. We deliberately do NOT fetch live rates: a chatbot
# quoting an invented exchange rate is worse than one quoting a dated, labelled
# one. Rates as of 2026-08-01, base USD.
FX_TO_USD: dict[str, float] = {
    "USD": 1.0,
    "SAR": 0.2666,
    "AED": 0.2723,
    "GBP": 1.2750,
    "EUR": 1.0850,
    "QAR": 0.2747,
    "OMR": 2.5974,
}
FX_AS_OF = "2026-08-01"

ARABIC_RANGE = re.compile(r"[؀-ۿ]")

# Patterns that look like an attempt to steer a downstream LLM. Neutralized,
# not deleted, so the record stays readable and the redaction is auditable.
INJECTION_PATTERNS = [
    re.compile(p, re.I)
    for p in (
        r"ignore\s+(?:all\s+)?(?:previous|prior|above)\s+instructions?",
        r"disregard\s+(?:all\s+)?(?:previous|prior|above)",
        r"\byou\s+are\s+now\b",
        r"\bsystem\s*(?:prompt|message)\s*:",
        r"</?(?:system|assistant|user|untrusted_listing_data)>",
        r"\bnew\s+instructions?\s*:",
        r"\bact\s+as\s+(?:a|an)\b.{0,40}\binstead\b",
        r"reveal\s+(?:your|the)\s+(?:system\s+)?prompt",
    )
]

TAG_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"[ \t ]+")
NL_RE = re.compile(r"\n{3,}")


def strip_html(raw: str | None) -> str:
    if not raw:
        return ""
    text = raw.replace("</div>", "\n").replace("<br>", "\n").replace("<br/>", "\n")
    text = TAG_RE.sub(" ", text)
    text = (
        text.replace("&nbsp;", " ")
        .replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#039;", "'")
    )
    text = unicodedata.normalize("NFKC", text)
    text = WS_RE.sub(" ", text)
    return NL_RE.sub("\n\n", text).strip()


def sanitize(raw: str | None, *, max_len: int = 4000) -> str:
    """Strip markup and neutralize prompt-injection attempts.

    Injection patterns are applied BEFORE and AFTER HTML stripping. Before,
    so that tag-shaped attacks (a forged </untrusted_listing_data> delimiter)
    leave an auditable [redacted-instruction] marker instead of being silently
    swallowed by the tag stripper; after, because stripping tags can splice
    previously separated words into a new instruction.
    """
    text = raw or ""
    for pat in INJECTION_PATTERNS:
        text = pat.sub(" [redacted-instruction] ", text)
    text = strip_html(text)
    for pat in INJECTION_PATTERNS:
        text = pat.sub("[redacted-instruction]", text)
    return text[:max_len]


def detect_language(text: str) -> Literal["ar", "en", "mixed", "unknown"]:
    if not text:
        return "unknown"
    arabic = len(ARABIC_RANGE.findall(text))
    latin = len(re.findall(r"[A-Za-z]", text))
    if arabic == 0 and latin == 0:
        return "unknown"
    if arabic > latin * 2:
        return "ar"
    if latin > arabic * 2:
        return "en"
    return "mixed"


def to_float(v: Any) -> float | None:
    if v is None or v == "" or v == "-":
        return None
    try:
        f = float(str(v).replace(",", "").strip())
        return f if f > 0 else None
    except (ValueError, TypeError):
        return None


def to_int(v: Any) -> int | None:
    f = to_float(v)
    return int(f) if f is not None else None


class Property(BaseModel):
    """One normalized listing or development, from either source."""

    id: str
    source: Literal["darglobal", "wasalt"]
    source_id: str
    url: str
    title: str
    description: str = ""
    language: str = "unknown"

    listing_type: Literal["sale", "rent", "offplan", "unknown"] = "unknown"
    property_type: str | None = None
    property_usage: str | None = None

    price: float | None = None
    currency: str | None = None
    price_usd: float | None = None
    price_per_sqm: float | None = None

    bedrooms: int | None = None
    bathrooms: int | None = None
    area_sqm: float | None = None

    city: str | None = None
    district: str | None = None
    country: str | None = None
    latitude: float | None = None
    longitude: float | None = None

    developer: str | None = None
    project_name: str | None = None
    completion_status: str | None = None
    amenities: list[str] = Field(default_factory=list)
    images: list[str] = Field(default_factory=list)

    is_verified: bool | None = None
    published_at: str | None = None
    scraped_at: str = Field(
        default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds")
    )
    content_hash: str = ""

    @field_validator("title", "description")
    @classmethod
    def _clean(cls, v: str) -> str:
        return sanitize(v)

    def finalize(self) -> Property:
        """Derive computed fields. Call once, after construction."""
        if self.price and self.currency:
            rate = FX_TO_USD.get(self.currency.upper())
            if rate:
                self.price_usd = round(self.price * rate, 2)
        if self.price and self.area_sqm and self.area_sqm > 0:
            self.price_per_sqm = round(self.price / self.area_sqm, 2)
        if not self.language or self.language == "unknown":
            self.language = detect_language(f"{self.title} {self.description}")
        self.content_hash = hashlib.sha256(
            f"{self.title}|{self.description[:500]}|{self.price}|{self.city}".encode()
        ).hexdigest()[:16]
        return self

    def search_text(self) -> str:
        """The text that gets embedded and indexed for full-text search."""
        parts = [
            self.title,
            self.property_type or "",
            self.property_usage or "",
            f"{self.bedrooms} bedrooms" if self.bedrooms else "",
            f"{self.bathrooms} bathrooms" if self.bathrooms else "",
            f"{self.area_sqm:.0f} sqm" if self.area_sqm else "",
            self.district or "",
            self.city or "",
            self.country or "",
            self.project_name or "",
            self.developer or "",
            self.completion_status or "",
            " ".join(self.amenities[:20]),
            self.description[:1200],
        ]
        return " · ".join(p for p in parts if p).strip()


def dedupe(rows: list[Property]) -> list[Property]:
    """Drop exact duplicates by (source, source_id) then by content hash."""
    seen_ids: set[tuple[str, str]] = set()
    seen_hash: set[str] = set()
    out: list[Property] = []
    for r in rows:
        key = (r.source, r.source_id)
        if key in seen_ids or r.content_hash in seen_hash:
            continue
        seen_ids.add(key)
        seen_hash.add(r.content_hash)
        out.append(r)
    return out
