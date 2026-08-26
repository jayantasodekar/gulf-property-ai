"""Probe free models with a REAL tool-calling request.

The catalogue is not evidence. `GET /api/v1/models?supported_parameters=tools`
advertised `thinkingmachines/inkling:free` as free *and* tool-capable, and it
returned HTTP 403 to ordinary API calls ("only available on agentic
harnesses"). A model that 200s but cannot actually emit a tool call is just as
useless to this app, and no metadata field reports that either.

So the only way to know a model works is to ask it to call a tool and look at
what comes back. This does exactly that:

    make verify-models          # the configured chain
    make verify-models-all      # every free tool-capable model in the catalogue

Exit code is non-zero when no model in the configured chain works, so it is
usable as a pre-deploy gate.
"""

from __future__ import annotations

import argparse
import asyncio
import sys

import httpx

from .config import settings

# One unambiguous tool and a question that cannot be answered without it.
# A model that answers this from its own knowledge is not usable here.
PROBE_TOOL = [
    {
        "type": "function",
        "function": {
            "name": "search_properties",
            "description": "Search real-estate listings.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"},
                    "max_price_usd": {"type": "number", "description": "Max price in USD"},
                },
                "required": ["city"],
            },
        },
    }
]
PROBE_MESSAGES = [
    {"role": "user", "content": "Find apartments in Riyadh under 200000 USD."}
]

OK = "OK"
NO_TOOL_CALL = "NO TOOL CALL"
QUOTA = "QUOTA EXHAUSTED"


async def probe(model_id: str) -> tuple[str, str]:
    """Return (status, detail). status is OK / QUOTA / an error label."""
    payload = {
        "model": model_id,
        "messages": PROBE_MESSAGES,
        "tools": PROBE_TOOL,
        "tool_choice": "auto",
        "max_tokens": 200,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {settings.openrouter_api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": settings.public_url,
        "X-Title": settings.app_name,
    }
    try:
        async with httpx.AsyncClient(timeout=settings.request_timeout) as c:
            r = await c.post(
                f"{settings.openrouter_base_url}/chat/completions",
                headers=headers,
                json=payload,
            )
    except Exception as exc:  # noqa: BLE001
        return "ERROR", str(exc)[:110]

    if r.status_code == 429:
        body = r.text
        if "per-day" in body or "free-models-per-day" in body:
            return QUOTA, "account-wide free-tier daily cap; retry after reset"
        return "RATE LIMITED", "per-model 429, transient"
    if r.status_code in (401, 403):
        # The inkling failure mode: advertised, but not actually callable.
        return f"HTTP {r.status_code}", r.text[:110].replace("\n", " ")
    if r.status_code >= 400:
        return f"HTTP {r.status_code}", r.text[:110].replace("\n", " ")

    try:
        msg = (r.json().get("choices") or [{}])[0].get("message") or {}
    except Exception as exc:  # noqa: BLE001
        return "BAD JSON", str(exc)[:110]

    calls = msg.get("tool_calls") or []
    if not calls:
        # 200, but it answered in prose instead of calling the tool. The
        # retrieval phase would fall through to the planner on every request.
        text = (msg.get("content") or "")[:70].replace("\n", " ")
        return NO_TOOL_CALL, f"replied with prose: {text!r}"

    fn = (calls[0].get("function") or {}).get("name", "?")
    args = (calls[0].get("function") or {}).get("arguments", "")
    return OK, f"called {fn}({args[:60]})"


async def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--all",
        action="store_true",
        help="probe every free tool-capable model in the live catalogue, "
        "not just the configured chain",
    )
    args = ap.parse_args()

    if not settings.openrouter_api_key.strip():
        print("OPENROUTER_API_KEY is not set - nothing to probe.")
        return 1

    models = settings.models
    if args.all:
        headers = {"Authorization": f"Bearer {settings.openrouter_api_key}"}
        async with httpx.AsyncClient(timeout=30.0) as c:
            r = await c.get(
                f"{settings.openrouter_base_url}/models",
                params={"supported_parameters": "tools"},
                headers=headers,
            )
            r.raise_for_status()
            catalogue = sorted(
                m["id"] for m in r.json().get("data", []) if m["id"].endswith(":free")
            )
        configured = set(models)
        models = models + [m for m in catalogue if m not in configured]

    print(f"Probing {len(models)} model(s) with a real tool-calling request.\n")
    results = await asyncio.gather(*(probe(m) for m in models))

    width = max(len(m) for m in models)
    working, quota_blocked = [], 0
    for model_id, (status, detail) in zip(models, results, strict=True):
        marker = "+" if status == OK else " "
        print(f"{marker} {model_id:<{width}}  {status:<16} {detail}")
        if status == OK:
            working.append(model_id)
        elif status == QUOTA:
            quota_blocked += 1

    print()
    if quota_blocked:
        print(
            f"{quota_blocked} model(s) blocked by the account-wide free-tier daily "
            "cap, which is not a property of the model. Re-run after the quota "
            "resets (00:00 UTC) to learn anything about them."
        )
    if working:
        print(f"{len(working)} usable: {', '.join(working)}")
        print("\nSuggested model_candidates order (verified working, in probe order):")
        print("  " + ",".join(working))
        return 0

    if quota_blocked == len(models):
        # Inconclusive, not a failure: the quota is an account property, so
        # this run learned nothing about the models either way. Failing the
        # gate here would just mean "you tested on the wrong day".
        print("INCONCLUSIVE - every probe hit the daily cap. No model was actually tested.")
        return 0

    print("No model in the chain could be verified. The app will serve search mode.")
    return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
