"""Golden-set evaluation harness.

    python -m eval.run              # full run (needs OPENROUTER_API_KEY)
    python -m eval.run --retrieval  # retrieval checks only, no LLM calls
    python -m eval.run --case beds-city-price

Checks target retrieval quality and grounding rather than exact prose: a free
model paraphrases differently on every run, so asserting on wording would make
the suite flaky without measuring anything real.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

import yaml

from app.agent import answer
from app.planner import heuristic_plan, llm_plan
from app.retrieval import get_retriever

GOLDEN = Path(__file__).resolve().parent / "golden.yaml"

REFUSAL_MARKERS = (
    "not publicly", "not listed", "don't have", "do not have", "cannot", "can't",
    "unable", "no public", "outside", "not able", "qualified", "professional",
    "register interest", "enquir", "not available", "i'm not", "i am not",
    "recommend consulting", "not something",
)


def check_case(case: dict, results: list[dict], answer_text: str) -> list[str]:
    """Return a list of failure strings (empty == pass)."""
    c = case.get("checks") or {}
    fails: list[str] = []
    lower = (answer_text or "").lower()

    if "min_results" in c and len(results) < c["min_results"]:
        fails.append(f"expected >={c['min_results']} results, got {len(results)}")
    if "max_results" in c and len(results) > c["max_results"]:
        fails.append(f"expected <={c['max_results']} results, got {len(results)}")

    if "all_city" in c and results:
        want = c["all_city"].lower()
        bad = [
            r["id"] for r in results
            if want not in f"{r.get('city') or ''} {r.get('district') or ''}".lower()
        ]
        if bad:
            fails.append(f"{len(bad)} result(s) outside {c['all_city']}: {bad[:3]}")

    if "all_source" in c and results:
        bad = [r["id"] for r in results if r.get("source") != c["all_source"]]
        if bad:
            fails.append(f"{len(bad)} result(s) not from {c['all_source']}: {bad[:3]}")

    if "all_listing_type" in c and results:
        bad = [r["id"] for r in results if r.get("listing_type") != c["all_listing_type"]]
        if bad:
            fails.append(f"{len(bad)} result(s) not {c['all_listing_type']}: {bad[:3]}")

    if "max_price_usd" in c and results:
        bad = [
            (r["id"], r["price_usd"]) for r in results
            if r.get("price_usd") and r["price_usd"] > c["max_price_usd"] * 1.02
        ]
        if bad:
            fails.append(f"price ceiling breached: {bad[:3]}")

    if "min_bedrooms" in c and results:
        bad = [
            (r["id"], r.get("bedrooms")) for r in results
            if r.get("bedrooms") is not None and r["bedrooms"] < c["min_bedrooms"]
        ]
        if bad:
            fails.append(f"bedroom floor breached: {bad[:3]}")

    if "any_id_contains" in c:
        if not any(c["any_id_contains"] in r.get("id", "") for r in results):
            fails.append(f"no result id contains {c['any_id_contains']!r}")

    if "answer_contains" in c and answer_text:
        if not any(t.lower() in lower for t in c["answer_contains"]):
            fails.append(f"answer mentions none of {c['answer_contains']}")

    if "answer_lacks" in c and answer_text:
        leaked = [t for t in c["answer_lacks"] if t.lower() in lower]
        if leaked:
            fails.append(f"answer leaked forbidden content: {leaked}")

    if c.get("expect_refusal") and answer_text:
        if not any(m in lower for m in REFUSAL_MARKERS):
            fails.append("expected a refusal/deferral, got a confident answer")

    return fails


async def run_case(case: dict, retrieval_only: bool) -> dict:
    q = case["question"]
    t0 = time.monotonic()
    results: list[dict] = []
    text = ""
    mode = "retrieval-only"

    if retrieval_only:
        try:
            filters, query, _ = await llm_plan(q)
        except Exception:  # noqa: BLE001
            filters, _ = heuristic_plan(q)
            query = q
        results = get_retriever().search(query, filters, k=8)
    else:
        async for ev in answer(q, []):
            if ev["type"] == "citations":
                results = ev["properties"]
            elif ev["type"] == "token":
                text += ev["text"]
            elif ev["type"] == "done":
                mode = ev["meta"].get("mode", "?")

    fails = check_case(case, results, text)
    return {
        "id": case["id"],
        "question": q,
        "passed": not fails,
        "failures": fails,
        "n_results": len(results),
        "mode": mode,
        "seconds": round(time.monotonic() - t0, 1),
        "answer": text[:400],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrieval", action="store_true", help="skip LLM answer generation")
    ap.add_argument("--case", help="run a single case id")
    ap.add_argument("--json", help="write full results to this path")
    args = ap.parse_args()

    cases = yaml.safe_load(GOLDEN.read_text(encoding="utf-8"))["cases"]
    if args.case:
        cases = [c for c in cases if c["id"] == args.case]
        if not cases:
            print(f"no such case: {args.case}")
            return 2

    print(f"\nRunning {len(cases)} golden cases "
          f"({'retrieval only' if args.retrieval else 'full pipeline'})\n")
    print(f"{'':2} {'case':28} {'res':>4} {'mode':10} {'sec':>5}  detail")
    print("-" * 96)

    out = []
    for case in cases:
        r = await run_case(case, args.retrieval)
        out.append(r)
        mark = "OK" if r["passed"] else "XX"
        detail = "" if r["passed"] else r["failures"][0][:44]
        print(f"{mark:2} {r['id']:28} {r['n_results']:>4} {r['mode']:10} "
              f"{r['seconds']:>5}  {detail}")

    passed = sum(1 for r in out if r["passed"])
    print("-" * 96)
    print(f"\n{passed}/{len(out)} passed ({100 * passed / max(len(out), 1):.0f}%)\n")

    for r in out:
        if not r["passed"]:
            print(f"  FAIL {r['id']}")
            for f in r["failures"]:
                print(f"       - {f}")

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nwrote {args.json}")

    return 0 if passed == len(out) else 1


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main()))
