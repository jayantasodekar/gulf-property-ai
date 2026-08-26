"""Wasalt (wasalt.sa) scraper.

Wasalt is a Next.js app that ships the full listing record inside the
__NEXT_DATA__ script tag, so we read that structured payload rather than
scraping the DOM -- a far more stable contract than CSS selectors.

Their robots.txt explicitly welcomes AI crawlers and publishes an llms.txt
policy, but we still honour `Disallow: /search` and rate-limit ourselves.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from collections import defaultdict
from typing import Any

from .common import Fetcher
from .normalize import Property, sanitize, to_float, to_int

log = logging.getLogger(__name__)

# The product sitemap already contains BOTH sale (~48k) and long-term rent
# (~12k) listings, and both use the same propertyDetailsV3 payload. The
# separate rental_pdp sitemap holds *daily-rental* booking pages, which are a
# different product with a different schema, so it is deliberately excluded.
SITEMAPS = {
    "product": "https://cdn.wasalt.sa/sitemap/product_sitemap_en_sa.xml.gz",
}

IMAGE_BASE = "https://imagedelivery.net/1DNKFJPRaeUdy_j8F7HT3w/production/properties"
LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
NEXT_DATA_RE = re.compile(r'id="__NEXT_DATA__"[^>]*>(\{.*?\})</script>', re.S)
ID_RE = re.compile(r"-(\d+)/?$")

# attribute key in Wasalt's payload -> our field name
ATTR_MAP = {
    "noOfBedrooms": "bedrooms",
    "bedrooms": "bedrooms",
    "noOfBathrooms": "bathrooms",
    "bathrooms": "bathrooms",
    "builtUpArea": "area_sqm",
    "landArea": "area_sqm",
    "plotArea": "area_sqm",
    "area": "area_sqm",
}


async def fetch_sitemap_urls(fetcher: Fetcher, url: str) -> list[str]:
    """Return every <loc> in a sitemap.

    The CDN serves these `.xml.gz` paths already decompressed, but we guard for
    both cases rather than relying on that.
    """
    text = await fetcher.get(url, check_robots=False)
    if "<loc>" not in text:
        log.warning("sitemap %s did not look like XML (%d bytes)", url, len(text))
        return []
    return LOC_RE.findall(text)


def deal_from_url(url: str) -> str:
    """`/en/property/rent/...` -> rent, otherwise sale."""
    return "rent" if "/property/rent/" in url else "sale"


def type_token(url: str) -> str:
    """Bucket key: the leading noun of the slug (apartment / villa / land ...)."""
    tail = url.rstrip("/").split("/")[-1]
    tail = ID_RE.sub("", tail)
    token = tail.split("-")[0].lower()
    return token if token.isalpha() and len(token) > 2 else "other"


def stratified_order(urls: list[str], seed: int = 42) -> list[tuple[str, str]]:
    """Order the whole URL pool so consuming any prefix is a stratified sample.

    A naive head-of-list sample returns thousands of near-identical Riyadh
    apartments. Instead we bucket by (deal, property type), shuffle within each
    bucket, then interleave the buckets round-robin. Taking the first N of the
    result is a stratified sample for ANY N -- which matters because the caller
    consumes URLs until it has enough *successes*, not a fixed count.

    Buckets are weighted by sqrt(size): the corpus stays roughly proportional
    to the real market while guaranteeing small categories (offices, chalets)
    are not crowded out entirely.
    """
    rng = random.Random(seed)
    buckets: dict[tuple[str, str], list[str]] = defaultdict(list)
    for u in urls:
        buckets[(deal_from_url(u), type_token(u))].append(u)
    for b in buckets.values():
        rng.shuffle(b)

    keys = sorted(buckets, key=lambda k: -len(buckets[k]))
    weights = {k: max(1.0, len(buckets[k]) ** 0.5) for k in keys}
    cursors = {k: 0 for k in keys}
    credit = {k: 0.0 for k in keys}

    ordered: list[tuple[str, str]] = []
    total = sum(len(v) for v in buckets.values())
    while len(ordered) < total:
        progressed = False
        for k in keys:
            credit[k] += weights[k]
            take = int(credit[k])
            if take <= 0:
                continue
            credit[k] -= take
            for _ in range(take):
                if cursors[k] < len(buckets[k]):
                    ordered.append((k[0], buckets[k][cursors[k]]))
                    cursors[k] += 1
                    progressed = True
        if not progressed:
            break
    return ordered


def parse_pdp(html: str, url: str, deal: str) -> Property | None:
    """Extract a Property from a Wasalt product detail page."""
    m = NEXT_DATA_RE.search(html)
    if not m:
        return None
    try:
        nd = json.loads(m.group(1))
        page = nd["props"]["pageProps"]
        pd = page["propertyDetailsV3"]
    except (json.JSONDecodeError, KeyError, TypeError):
        return None

    info: dict[str, Any] = pd.get("propertyInfo") or {}
    if not info:
        return None

    id_match = ID_RE.search(url)
    prop_id = str(pd.get("id") or (id_match.group(1) if id_match else ""))
    if not prop_id:
        return None

    # --- attributes (beds / baths / area) ---
    fields: dict[str, Any] = {}
    for attr in pd.get("attributes") or []:
        key = ATTR_MAP.get(attr.get("key", ""))
        if key and attr.get("value") not in (None, "", "-"):
            fields.setdefault(key, attr["value"])

    extras = {a.get("key"): a.get("value") for a in (pd.get("additionalAttributes") or [])}

    loc = pd.get("location") or {}
    price = to_float(info.get("salePrice")) or to_float(info.get("conversionPrice"))
    listing_type = "rent" if (deal == "rent" or info.get("propertyFor") == "rent") else "sale"

    gallery = ((page.get("galleryDetails") or {}).get("images") or {}).get("data") or []
    images = [
        f"{IMAGE_BASE}/{prop_id}/images/{g['content']}/quality=70,w=640"
        for g in gallery[:6]
        if g.get("content") and g.get("type") == "image"
    ]

    amenities = [
        f"{a.get('label')}: {a.get('value')}"
        for a in (pd.get("additionalAttributes") or [])
        if a.get("label") and a.get("value") and a.get("key") != "adSource"
    ][:15]

    return Property(
        id=f"wasalt-{prop_id}",
        source="wasalt",
        source_id=prop_id,
        url=url,
        title=sanitize(info.get("title") or info.get("propertyName") or "Property"),
        description=info.get("description") or "",
        listing_type=listing_type,
        property_type=info.get("propertySubType"),
        property_usage=info.get("propertyMainType"),
        price=price,
        currency=info.get("currencyType") or info.get("conversionUnit") or "SAR",
        bedrooms=to_int(fields.get("bedrooms")),
        bathrooms=to_int(fields.get("bathrooms")),
        area_sqm=to_float(fields.get("area_sqm")),
        city=info.get("city"),
        district=info.get("district") or info.get("zone"),
        country=info.get("country") or "Saudi Arabia",
        latitude=to_float(loc.get("lat")),
        longitude=to_float(loc.get("lon")),
        completion_status=extras.get("completionYear"),
        amenities=amenities,
        images=images,
        is_verified=bool(pd.get("isVerified")),
        published_at=pd.get("publishedAt"),
    ).finalize()


async def scrape(
    target: int = 3000, concurrency: int = 6, use_cache: bool = True
) -> tuple[list[Property], list[str]]:
    """Fetch until `target` listings parse successfully.

    Roughly a third of the sitemap 60k URLs are stale (HTTP 404 for delisted
    listings), so a fixed-size sample under-delivers by an unpredictable
    margin. Consuming a stratified stream until the target is met makes the
    corpus size deterministic regardless of how stale the sitemap is.
    """
    results: list[Property] = []
    failures: list[str] = []
    attempted = 0

    async with Fetcher(rate=2.5, use_cache=use_cache) as f:
        pool: list[tuple[str, str]] = []
        for name, sm in SITEMAPS.items():
            try:
                urls = await fetch_sitemap_urls(f, sm)
                log.info("wasalt sitemap %-8s -> %d urls", name, len(urls))
                pool.extend(stratified_order(urls))
            except Exception as exc:  # noqa: BLE001
                log.error("sitemap %s failed: %s", sm, exc)

        if not pool:
            return [], []
        log.info(
            "wasalt: %d candidate urls, streaming until %d parse successfully",
            len(pool), target,
        )

        sem = asyncio.Semaphore(concurrency)
        cursor = 0
        cursor_lock = asyncio.Lock()

        async def worker() -> None:
            nonlocal cursor, attempted
            while True:
                async with cursor_lock:
                    if len(results) >= target or cursor >= len(pool):
                        return
                    deal, url = pool[cursor]
                    cursor += 1
                async with sem:
                    try:
                        prop = parse_pdp(await f.get(url), url, deal)
                        if prop:
                            results.append(prop)
                        else:
                            failures.append(url)
                    except Exception as exc:  # noqa: BLE001
                        failures.append(url)
                        log.debug("wasalt fetch failed %s: %s", url, exc)
                    finally:
                        attempted += 1
                        if attempted % 250 == 0:
                            log.info(
                                "wasalt progress: %d ok / %d attempted (target %d)",
                                len(results), attempted, target,
                            )

        await asyncio.gather(*(worker() for _ in range(concurrency)))

    log.info(
        "wasalt done: %d ok, %d failed, %d attempted (%.0f%% yield)",
        len(results), len(failures), attempted,
        100 * len(results) / max(attempted, 1),
    )
    return results[:target], failures
