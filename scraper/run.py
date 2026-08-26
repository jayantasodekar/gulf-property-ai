"""Scraper CLI.

    python -m scraper.run --source both --limit 3000
    python -m scraper.run --source darglobal --no-cache
    python -m scraper.run --stats-only
"""

from __future__ import annotations

import argparse
import asyncio
import gzip
import json
import logging
import sys
from collections import Counter
from pathlib import Path

from .common import write_jsonl
from .normalize import Property, dedupe

DATA = Path(__file__).resolve().parent.parent / "data"
CORPUS = DATA / "corpus.jsonl.gz"
FAILURES = DATA / "failures.jsonl"

log = logging.getLogger("scraper")


def summarize(rows: list[Property]) -> str:
    if not rows:
        return "  (empty corpus)"
    by_source = Counter(r.source for r in rows)
    by_type = Counter(r.property_type or "?" for r in rows)
    by_city = Counter(r.city or "?" for r in rows)
    by_deal = Counter(r.listing_type for r in rows)
    by_lang = Counter(r.language for r in rows)
    priced = sum(1 for r in rows if r.price)
    geo = sum(1 for r in rows if r.latitude)
    beds = sum(1 for r in rows if r.bedrooms)
    area = sum(1 for r in rows if r.area_sqm)
    n = len(rows)

    def pct(x: int) -> str:
        return f"{x:5d} ({100 * x / n:4.1f}%)"

    lines = [
        f"  total records     {n}",
        f"  by source         {dict(by_source)}",
        f"  by listing_type   {dict(by_deal)}",
        f"  by language       {dict(by_lang)}",
        f"  top property_type {by_type.most_common(6)}",
        f"  top cities        {by_city.most_common(8)}",
        f"  with price        {pct(priced)}",
        f"  with bedrooms     {pct(beds)}",
        f"  with area_sqm     {pct(area)}",
        f"  with coordinates  {pct(geo)}",
    ]
    return "\n".join(lines)


def load_corpus() -> list[Property]:
    if not CORPUS.exists():
        return []
    with gzip.open(CORPUS, "rt", encoding="utf-8") as fh:
        return [Property(**json.loads(line)) for line in fh if line.strip()]


def save_corpus(rows: list[Property]) -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    with gzip.open(CORPUS, "wt", encoding="utf-8") as fh:
        for r in rows:
            fh.write(r.model_dump_json() + "\n")


async def main() -> int:
    ap = argparse.ArgumentParser(description="Scrape DarGlobal and Wasalt.")
    ap.add_argument("--source", choices=["both", "wasalt", "darglobal"], default="both")
    ap.add_argument("--limit", type=int, default=3000, help="Wasalt listing target")
    ap.add_argument("--no-cache", action="store_true", help="ignore the on-disk response cache")
    ap.add_argument("--stats-only", action="store_true", help="summarize the existing corpus")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("curl_cffi").setLevel(logging.WARNING)

    if args.stats_only:
        rows = load_corpus()
        print(f"\ncorpus: {CORPUS}")
        print(summarize(rows))
        return 0

    use_cache = not args.no_cache
    rows: list[Property] = []
    failures: list[str] = []

    if args.source in ("darglobal", "both"):
        from . import darglobal

        log.info("=== DarGlobal ===")
        r, f = await darglobal.scrape(use_cache=use_cache)
        rows += r
        failures += f

    if args.source in ("wasalt", "both"):
        from . import wasalt

        log.info("=== Wasalt (target %d) ===", args.limit)
        r, f = await wasalt.scrape(target=args.limit, use_cache=use_cache)
        rows += r
        failures += f

    before = len(rows)
    rows = dedupe(rows)
    log.info("deduped %d -> %d records", before, len(rows))

    save_corpus(rows)
    if failures:
        write_jsonl(FAILURES, [{"url": u} for u in failures])

    print(f"\nwrote {CORPUS}  ({CORPUS.stat().st_size / 1e6:.2f} MB)")
    print(summarize(rows))
    if failures:
        print(f"\n  {len(failures)} URLs failed -> {FAILURES}")
    return 0


if __name__ == "__main__":
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    raise SystemExit(asyncio.run(main()))
