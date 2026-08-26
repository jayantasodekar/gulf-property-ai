"""Chat orchestration.

The pipeline is deliberately split into a RETRIEVAL phase and an ANSWER phase,
each with its own degradation ladder. That separation is what lets the app keep
working as capabilities disappear:

  retrieval:  tool-calling  ->  LLM JSON planner  ->  regex heuristics
  answer:     streamed LLM  ->  "search mode" (ranked listings, no prose)

The worst case is still a useful product: a property search engine with
citations. It never becomes an error page.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from .config import settings
from .llm import NoModelAvailable, client
from .planner import heuristic_plan, llm_plan
from .prompts import SEARCH_MODE_NOTICE, SYSTEM_PROMPT, wrap_untrusted
from .retrieval import Filters, get_retriever

log = logging.getLogger(__name__)

FILTER_PROPS = {
    "city": {"type": "string", "description": "City or district name, e.g. Riyadh, Jeddah, Dubai"},
    "country": {"type": "string"},
    "source": {"type": "string", "enum": ["wasalt", "darglobal"]},
    "listing_type": {"type": "string", "enum": ["sale", "rent", "offplan"]},
    "property_type": {"type": "string", "description": "Apartment, Villa, Land, Office, Floor..."},
    "min_price_usd": {"type": "number", "description": "Minimum price in USD"},
    "max_price_usd": {"type": "number", "description": "Maximum price in USD"},
    "min_bedrooms": {"type": "integer"},
    "max_bedrooms": {"type": "integer"},
}

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": (
                "Search the indexed DarGlobal + Wasalt corpus. Combines keyword, "
                "semantic and structured filtering. Use for any question about "
                "specific properties. Prices must be given in USD."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Descriptive free text, e.g. 'sea view family villa'",
                    },
                    **FILTER_PROPS,
                    "limit": {"type": "integer", "description": "Max results, default 8"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "market_stats",
            "description": (
                "Exact aggregates (count, min, max, average, median price) computed "
                "in SQL. ALWAYS use this for averages, medians, counts or price "
                "ranges - never calculate them yourself."
            ),
            "parameters": {"type": "object", "properties": FILTER_PROPS},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_property",
            "description": "Fetch one property's full record by its id.",
            "parameters": {
                "type": "object",
                "properties": {"property_id": {"type": "string"}},
                "required": ["property_id"],
            },
        },
    },
]


def _filters_from(args: dict) -> Filters:
    allowed = set(FILTER_PROPS)
    return Filters(**{k: v for k, v in args.items() if k in allowed and v not in (None, "")})


def run_tool(name: str, args: dict) -> tuple[Any, list[dict]]:
    """Execute a tool. Returns (payload_for_model, citations)."""
    r = get_retriever()
    if name == "search_properties":
        limit = min(int(args.get("limit") or 8), 12)
        rows = r.search(str(args.get("query") or ""), _filters_from(args), k=limit)
        return rows, rows
    if name == "market_stats":
        return r.market_stats(_filters_from(args)), []
    if name == "get_property":
        row = r.get(str(args.get("property_id") or ""))
        return (row or {"error": "not found"}), ([row] if row else [])
    return {"error": f"unknown tool {name}"}, []


def _history(history: list[dict]) -> list[dict]:
    out = []
    for m in (history or [])[-settings.max_history_turns :]:
        role = m.get("role")
        content = (m.get("content") or "").strip()
        if role in ("user", "assistant") and content:
            out.append({"role": role, "content": content[: settings.max_message_chars]})
    return out


async def _retrieve_via_tools(question: str, history: list[dict]) -> tuple[list[dict], list[str]]:
    """Path A: let the model choose tools. Raises NoModelAvailable if unusable."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history(history),
        {"role": "user", "content": question},
    ]
    citations: list[dict] = []
    notes: list[str] = []

    for _ in range(settings.max_tool_rounds):
        msg = await client.complete(messages, tools=TOOLS, max_tokens=700)
        calls = msg.get("tool_calls") or []
        if not calls:
            break
        messages.append(
            {
                "role": "assistant",
                "content": msg.get("content") or "",
                "tool_calls": calls,
            }
        )
        for call in calls[:4]:
            fn = (call.get("function") or {}).get("name", "")
            try:
                args = json.loads((call.get("function") or {}).get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            payload, cites = run_tool(fn, args)
            citations.extend(cites)
            notes.append(f"{fn}({json.dumps(args, ensure_ascii=False)[:120]})")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call.get("id", ""),
                    "content": wrap_untrusted(payload, label=fn),
                }
            )
    return citations, notes


async def _retrieve_via_planner(question: str) -> tuple[list[dict], Any, list[str]]:
    """Path B/C: JSON planner (or regex) -> direct search."""
    try:
        filters, query, wants_stats = await llm_plan(question)
    except Exception as exc:  # noqa: BLE001
        log.info("planner unavailable (%s); using heuristics", exc)
        filters, wants_stats = heuristic_plan(question)
        query = question

    r = get_retriever()
    rows = r.search(query, filters, k=8)
    if not rows and filters.active():
        # Over-constrained: relax price and bedroom bounds before giving up.
        relaxed = Filters(city=filters.city, source=filters.source,
                          listing_type=filters.listing_type,
                          property_type=filters.property_type)
        rows = r.search(query, relaxed, k=8)
        if rows:
            filters = relaxed
    stats = r.market_stats(filters) if wants_stats else None
    notes = [f"plan={json.dumps(filters.active(), default=str)[:160]}"]
    return rows, stats, notes


def _dedupe_citations(rows: list[dict], limit: int = 8) -> list[dict]:
    seen, out = set(), []
    for row in rows:
        if not row or row.get("id") in seen:
            continue
        seen.add(row["id"])
        out.append(row)
        if len(out) >= limit:
            break
    return out


def _search_mode_text(rows: list[dict]) -> str:
    if not rows:
        return (
            f"{SEARCH_MODE_NOTICE}\n\nI could not find matching listings in the "
            "indexed corpus. Try a broader query, e.g. a city name on its own."
        )
    lines = [SEARCH_MODE_NOTICE, "", f"Top {len(rows)} matches:", ""]
    for row in rows:
        price = (
            f"{row['price']:,.0f} {row.get('currency') or ''}".strip()
            if row.get("price")
            else "price on application"
        )
        bits = [b for b in (
            row.get("property_type"),
            f"{row['bedrooms']} bed" if row.get("bedrooms") else None,
            f"{row['area_sqm']:.0f} sqm" if row.get("area_sqm") else None,
            row.get("city"),
        ) if b]
        lines.append(f"- **{row.get('title')}** — {price} · {' · '.join(bits)}")
    return "\n".join(lines)


async def answer(question: str, history: list[dict] | None = None) -> AsyncIterator[dict]:
    """Yield SSE-shaped events for one user turn."""
    history = history or []
    citations: list[dict] = []
    stats: Any = None
    notes: list[str] = []
    mode = "agent"

    # ---------------- retrieval phase ----------------
    yield {"type": "status", "message": "Searching listings…"}
    try:
        if not client.enabled:
            raise NoModelAvailable("no API key")
        citations, notes = await _retrieve_via_tools(question, history)
        if not citations:
            rows, stats, n2 = await _retrieve_via_planner(question)
            citations, notes = rows, notes + n2
            mode = "planner"
    except NoModelAvailable as exc:
        log.info("tool path unavailable (%s); using planner path", exc)
        mode = "planner"
        try:
            citations, stats, notes = await _retrieve_via_planner(question)
        except Exception as exc2:  # noqa: BLE001
            log.warning("planner path failed too: %s", exc2)
            f, wants = heuristic_plan(question)
            citations = get_retriever().search(question, f, k=8)
            stats = get_retriever().market_stats(f) if wants else None
            mode = "heuristic"
    except Exception as exc:  # noqa: BLE001
        log.exception("retrieval failed: %s", exc)
        f, _ = heuristic_plan(question)
        citations = get_retriever().search(question, f, k=8)
        mode = "heuristic"

    citations = _dedupe_citations(citations)
    yield {"type": "citations", "properties": citations}

    # ---------------- answer phase ----------------
    context_parts = []
    if citations:
        context_parts.append(wrap_untrusted(citations, label="search_properties"))
    if stats:
        context_parts.append(wrap_untrusted(stats, label="market_stats"))
    if not context_parts:
        context_parts.append(
            wrap_untrusted({"result": "no matching listings in the indexed corpus"})
        )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *_history(history),
        {
            "role": "user",
            "content": (
                f"{question}\n\n"
                "Answer using only the retrieved data below.\n\n"
                + "\n\n".join(context_parts)
            ),
        },
    ]

    streamed = False
    try:
        async for delta in client.stream(messages, max_tokens=1000):
            streamed = True
            yield {"type": "token", "text": delta}
    except NoModelAvailable as exc:
        log.warning("answer stream unavailable: %s", exc)
    except Exception as exc:  # noqa: BLE001
        log.exception("answer stream failed: %s", exc)

    if not streamed:
        mode = "search"
        yield {"type": "token", "text": _search_mode_text(citations)}

    yield {
        "type": "done",
        "meta": {
            "mode": mode,
            "citations": len(citations),
            "tools": notes[:6],
            "model": next(
                (s.model_id for s in client.states if s.successes and s.available), None
            ),
        },
    }
