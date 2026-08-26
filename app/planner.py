"""Turn a natural-language question into structured Filters.

Two implementations, tried in order:

1. `llm_plan` - asks the model for a JSON plan. Handles paraphrase and
   implication ("somewhere cheap for a family" -> bedrooms >= 3).
2. `heuristic_plan` - pure regex. No API call, no key needed. This is what
   keeps the app answering when OpenRouter is down, and it also acts as a
   sanity net: anything the LLM plan misses, the heuristic can still catch.
"""

from __future__ import annotations

import json
import logging
import re

from .llm import NoModelAvailable, client
from .prompts import planner_prompt
from .retrieval import Filters

log = logging.getLogger(__name__)

# FX for user-stated prices -> USD (mirrors scraper/normalize.py FX_TO_USD)
FX = {"SAR": 0.2666, "AED": 0.2723, "GBP": 1.2750, "USD": 1.0, "EUR": 1.0850,
      "QAR": 0.2747, "OMR": 2.5974}

CITIES = [
    "riyadh", "jeddah", "dammam", "aldammam", "khobar", "mecca", "makkah",
    "medina", "madinah", "taif", "abha", "tabuk", "buraidah", "khamis",
    "dubai", "abu dhabi", "sharjah", "doha", "muscat", "marbella", "malaga",
    "london", "marrakech", "athens", "amman",
]

PROPERTY_TYPES = {
    "apartment": "Apartment", "flat": "Apartment", "villa": "Villa",
    "land": "Land", "plot": "Land", "office": "Office", "shop": "Shop",
    "building": "Building", "floor": "Floor", "studio": "Studio",
    "townhouse": "Townhouse", "penthouse": "Penthouse", "rest": "Rest House",
    "duplex": "Duplex", "warehouse": "Warehouse", "chalet": "Chalet",
}

SCALE = {"k": 1e3, "thousand": 1e3, "m": 1e6, "million": 1e6, "mn": 1e6,
         "bn": 1e9, "billion": 1e9}

_NUM = r"(\d[\d,]*(?:\.\d+)?)\s*(k|m|mn|bn|million|thousand|billion)?"
PRICE_MAX_RE = re.compile(
    rf"(?:under|below|less than|cheaper than|up to|max(?:imum)?|budget of|within)\s*"
    rf"(?:(SAR|AED|USD|GBP|EUR|QAR|OMR|\$|£|€)\s*)?{_NUM}\s*"
    rf"(?:(SAR|AED|USD|GBP|EUR|QAR|OMR)\b)?",
    re.I,
)
PRICE_MIN_RE = re.compile(
    rf"(?:over|above|more than|at least|min(?:imum)?|starting (?:at|from)|from)\s*"
    rf"(?:(SAR|AED|USD|GBP|EUR|QAR|OMR|\$|£|€)\s*)?{_NUM}\s*"
    rf"(?:(SAR|AED|USD|GBP|EUR|QAR|OMR)\b)?",
    re.I,
)
BEDS_RE = re.compile(r"(\d+)[\s-]*(?:bed|bedroom|br\b|bhk)", re.I)
SYMBOL_CCY = {"$": "USD", "£": "GBP", "€": "EUR"}

STATS_WORDS = re.compile(
    r"\b(average|avg|mean|median|typical|how many|count|cheapest|most expensive|"
    r"price range|market|statistics|stats|compare prices|distribution)\b",
    re.I,
)


def _amount(num: str, scale: str | None, ccy: str | None, default_ccy: str) -> float:
    val = float(num.replace(",", ""))
    if scale:
        val *= SCALE.get(scale.lower(), 1)
    code = SYMBOL_CCY.get(ccy or "", (ccy or "").upper()) or default_ccy
    return val * FX.get(code, FX[default_ccy])


def heuristic_plan(question: str, default_ccy: str = "SAR") -> tuple[Filters, bool]:
    """Regex-only plan. Returns (filters, wants_stats)."""
    q = question.lower()
    f = Filters()

    for c in CITIES:
        if re.search(rf"\b{re.escape(c)}\b", q):
            f.city = "Dammam" if c == "aldammam" else c.title()
            break

    for word, canonical in PROPERTY_TYPES.items():
        if re.search(rf"\b{word}s?\b", q):
            f.property_type = canonical
            break

    if re.search(r"\b(rent|rental|renting|lease|monthly)\b", q):
        f.listing_type = "rent"
    elif re.search(r"\b(off.?plan|under construction|new development|launch)\b", q):
        f.listing_type = "offplan"
    elif re.search(r"\b(buy|sale|purchase|for sale|investment)\b", q):
        f.listing_type = "sale"

    if "darglobal" in q or "dar global" in q:
        f.source = "darglobal"
    elif "wasalt" in q:
        f.source = "wasalt"

    m = BEDS_RE.search(q)
    if m:
        f.min_bedrooms = f.max_bedrooms = int(m.group(1))

    m = PRICE_MAX_RE.search(question)
    if m:
        f.max_price_usd = _amount(m.group(2), m.group(3), m.group(1) or m.group(4), default_ccy)
    m = PRICE_MIN_RE.search(question)
    if m:
        f.min_price_usd = _amount(m.group(2), m.group(3), m.group(1) or m.group(4), default_ccy)

    return f, bool(STATS_WORDS.search(q))


async def llm_plan(question: str) -> tuple[Filters, str, bool]:
    """Ask the model for a JSON plan; fall back to regex on any problem.

    Returns (filters, semantic_query, wants_stats).
    """
    fallback_f, fallback_stats = heuristic_plan(question)
    try:
        msg = await client.complete(
            [{"role": "user", "content": planner_prompt(question)}],
            temperature=0.0,
            max_tokens=400,
        )
        raw = (msg.get("content") or "").strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            raise ValueError("no JSON object in planner reply")
        plan = json.loads(m.group(0))
    except (NoModelAvailable, ValueError, json.JSONDecodeError) as exc:
        log.info("planner fell back to heuristics: %s", exc)
        return fallback_f, question, fallback_stats
    except Exception as exc:  # noqa: BLE001
        log.warning("planner error, using heuristics: %s", exc)
        return fallback_f, question, fallback_stats

    def pick(key: str, cast=None):
        v = plan.get(key)
        if v in (None, "", "null"):
            return None
        try:
            return cast(v) if cast else v
        except (TypeError, ValueError):
            return None

    f = Filters(
        city=pick("city") or fallback_f.city,
        country=pick("country"),
        source=pick("source") or fallback_f.source,
        listing_type=pick("listing_type") or fallback_f.listing_type,
        property_type=pick("property_type") or fallback_f.property_type,
        min_price_usd=pick("min_price_usd", float) or fallback_f.min_price_usd,
        max_price_usd=pick("max_price_usd", float) or fallback_f.max_price_usd,
        min_bedrooms=pick("min_bedrooms", int) or fallback_f.min_bedrooms,
        max_bedrooms=pick("max_bedrooms", int) or fallback_f.max_bedrooms,
    )
    # Only trust the model's listing_type/source when it is a legal value.
    if f.listing_type not in (None, "sale", "rent", "offplan"):
        f.listing_type = fallback_f.listing_type
    if f.source not in (None, "wasalt", "darglobal"):
        f.source = fallback_f.source

    query = pick("query") or question
    wants_stats = bool(plan.get("wants_stats")) or fallback_stats
    return f, str(query), wants_stats
