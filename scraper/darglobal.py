"""DarGlobal (darglobal.co.uk) scraper.

IMPORTANT: the developer is `darglobal.co.uk`. `darglobal.com` is an unrelated
Kazakh holding group -- scraping it would silently poison the corpus with the
wrong company's data.

DarGlobal sits behind an Imperva WAF (see scraper/common.py for how that is
handled). Project pages embed a complete schema.org graph -- RealEstateListing
+ ApartmentComplex + Place + FAQPage -- so extraction reads that graph rather
than CSS selectors.

Project pages are identified by *self-declaration*: a page is a project if its
JSON-LD contains a RealEstateListing node. That is more robust than maintaining
a hand-written slug allow-list against a site that adds developments.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from collections.abc import Iterator
from typing import Any

from selectolax.parser import HTMLParser

from .common import Fetcher
from .normalize import Property, sanitize, strip_html, to_float, to_int

log = logging.getLogger(__name__)

SITEMAP = "https://darglobal.co.uk/sitemap.xml"
LOC_RE = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")

# Paths that are definitively not developments. Everything else is probed and
# accepted only if it self-declares as a RealEstateListing.
EXCLUDE_PREFIXES = (
    "/blog", "/press", "/insights", "/careers", "/internship", "/about", "/faq",
    "/get-in-touch", "/pay-online", "/privacy", "/terms", "/cookie", "/sitemap",
    "/become-", "/campaigns", "/bookyourunit", "/dg-circle", "/exclusive-membership",
    "/development-management", "/media", "/news", "/investor", "/contact",
    "/win-a-trip", "/why-invest", "/search", "/thank", "/unsubscribe",
)

CITY_COUNTRY = {
    "dubai": "United Arab Emirates", "abu dhabi": "United Arab Emirates",
    "jeddah": "Saudi Arabia", "riyadh": "Saudi Arabia", "makkah": "Saudi Arabia",
    "mecca": "Saudi Arabia", "doha": "Qatar", "muscat": "Oman",
    "marbella": "Spain", "malaga": "Spain", "london": "United Kingdom",
    "marrakech": "Morocco", "athens": "Greece", "amman": "Jordan",
}

COUNTRY_NAMES = {
    "AE": "United Arab Emirates", "SA": "Saudi Arabia", "QA": "Qatar",
    "OM": "Oman", "ES": "Spain", "GB": "United Kingdom", "UK": "United Kingdom",
    "MA": "Morocco", "GR": "Greece", "US": "United States",
}

CURRENCY_RE = re.compile(
    r"(AED|USD|GBP|EUR|SAR|QAR|OMR)\s*([\d,]+(?:\.\d+)?)\s*(million|m\b|bn|billion)?",
    re.I,
)
BEDROOM_RE = re.compile(r"(\d+)\s*(?:,|&|and|to|-)?\s*(?:\d+\s*(?:,|&|and)?\s*)*bedrooms?", re.I)


def walk_json(obj: Any) -> Iterator[dict]:
    """Yield every dict in a nested JSON-LD structure."""
    if isinstance(obj, dict):
        yield obj
        for v in obj.values():
            yield from walk_json(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from walk_json(v)


def extract_ld_nodes(html: str) -> list[dict]:
    nodes: list[dict] = []
    for m in re.findall(
        r'<script[^>]*type=["\']application/ld\+json["\'][^>]*>(.*?)</script>', html, re.S
    ):
        try:
            nodes.extend(walk_json(json.loads(m.strip())))
        except json.JSONDecodeError:
            continue
    return nodes


def first_of_type(nodes: list[dict], *types: str) -> dict | None:
    for n in nodes:
        t = n.get("@type")
        t = t if isinstance(t, str) else (t[0] if isinstance(t, list) and t else None)
        if t in types:
            return n
    return None


def body_text(html: str) -> str:
    tree = HTMLParser(html)
    for tag in ("script", "style", "noscript", "svg"):
        for node in tree.css(tag):
            node.decompose()
    return re.sub(r"\s+", " ", tree.body.text() if tree.body else "").strip()


def parse_spec_table(text: str) -> dict[str, str]:
    """Pull the Key Features strip: completion date, unit type, area, status.

    Rendered as a flat run of label/value pairs, e.g.
    "Expected Completion Date September 2026 Unit Type 1, 2 & 3-bedrooms
     Area (SQM) 355 Status Sold Out"
    """
    labels = [
        "Expected Completion Date", "Completion Date", "Handover", "Delivery",
        "Unit Type", "Unit Types", "Units", "Area (SQM)", "Area", "Status",
        "Starting Price", "Starting From", "Price From", "Price", "Location",
        "Number of Floors", "Floors", "Stage", "Developer", "Property Type",
        "Payment Plan", "Down Payment", "Plot Size", "Total Units", "Views",
    ]
    # longest-first so "Unit Types" wins over "Unit Type", "Area (SQM)" over "Area"
    labels.sort(key=len, reverse=True)
    pattern = "|".join(re.escape(x) for x in labels)
    out: dict[str, str] = {}
    for m in re.finditer(rf"({pattern})\s*[:\-]?\s*(.{{1,60}}?)(?=(?:{pattern})|$)", text):
        key, val = m.group(1).strip(), m.group(2).strip(" :-·|")
        if val and key not in out:
            out[key] = val
    return out


def parse_price(text: str) -> tuple[float | None, str | None]:
    """Find a headline price. Returns (amount, currency)."""
    window = text[:6000]
    for m in CURRENCY_RE.finditer(window):
        cur = m.group(1).upper()
        amount = to_float(m.group(2))
        if amount is None:
            continue
        scale = (m.group(3) or "").lower()
        if scale in ("million", "m"):
            amount *= 1_000_000
        elif scale in ("bn", "billion"):
            amount *= 1_000_000_000
        # ignore obvious non-prices (golden-visa thresholds, phone numbers)
        if amount >= 50_000:
            return amount, cur
    return None, None


def _status_line(status: str, completion: str | None, floors: str | None) -> str | None:
    bits = []
    if status:
        bits.append(f"Availability: {status.strip()}")
    if completion:
        bits.append(f"Completion: {completion.strip()}")
    if floors:
        bits.append(f"Floors: {floors.strip()}")
    return " · ".join(bits) or None


def parse_project(html: str, url: str) -> Property | None:
    nodes = extract_ld_nodes(html)
    listing = first_of_type(nodes, "RealEstateListing")
    if not listing:
        return None  # not a development page

    complex_node = first_of_type(nodes, "ApartmentComplex", "Residence", "House") or {}
    place = first_of_type(nodes, "Place") or {}
    addr = complex_node.get("address") or place.get("address") or {}
    geo = complex_node.get("geo") or place.get("geo") or {}

    text = body_text(html)
    specs = parse_spec_table(text)
    price, currency = parse_price(text)

    slug = url.rstrip("/").split("/")[-1]
    name = listing.get("name") or complex_node.get("name") or slug.replace("-", " ").title()

    # Description: meta description is a one-liner, so enrich it with the FAQ
    # answers, which carry genuinely useful buyer information.
    desc_parts = [listing.get("description") or complex_node.get("description") or ""]
    for n in nodes:
        if n.get("@type") == "Question":
            q = strip_html(n.get("name", ""))
            a = n.get("acceptedAnswer") or {}
            a_text = strip_html(a.get("text", "")) if isinstance(a, dict) else ""
            if q and a_text:
                desc_parts.append(f"Q: {q} A: {a_text}")
    for key in ("Unit Type", "Unit Types", "Status", "Expected Completion Date"):
        if key in specs:
            desc_parts.append(f"{key}: {specs[key]}")

    amenities = [
        n["name"]
        for n in nodes
        if n.get("@type") == "LocationFeatureSpecification" and n.get("name")
    ][:25]

    images = listing.get("image") or complex_node.get("image") or []
    if isinstance(images, str):
        images = [images]

    floor = complex_node.get("floorSize") or {}
    area = to_float(floor.get("value")) if isinstance(floor, dict) else None
    if area is None:
        area = to_float(specs.get("Area (SQM)") or specs.get("Area"))

    beds = None
    unit_type = specs.get("Unit Type") or specs.get("Unit Types") or ""
    bed_nums = re.findall(r"(\d+)", unit_type)
    if bed_nums:
        beds = to_int(max(bed_nums, key=lambda x: int(x)))

    country_code = (addr.get("addressCountry") or "").upper()
    city = addr.get("addressLocality")
    country = COUNTRY_NAMES.get(country_code, country_code or None)
    if not country and city:
        country = CITY_COUNTRY.get(city.strip().lower())

    status = specs.get("Status") or specs.get("Stage") or ""
    completion = (
        specs.get("Expected Completion Date")
        or specs.get("Completion Date")
        or specs.get("Handover")
    )
    # Every DarGlobal page is a development, not an individual resale listing.
    # Availability lives in completion_status, so "sold out" does not become a
    # listing_type -- that would wrongly surface it in "for sale" queries.
    listing_type = "offplan"

    extra_props = {
        p.get("name"): p.get("value")
        for p in nodes
        if p.get("@type") == "PropertyValue" and p.get("name")
    }
    if not completion:
        completion = extra_props.get("estimatedCompletionDate")

    brand = listing.get("brand") or {}
    brand_name = brand.get("name") if isinstance(brand, dict) else None

    return Property(
        id=f"darglobal-{slug}",
        source="darglobal",
        source_id=slug,
        url=url,
        title=sanitize(name),
        description=" ".join(p for p in desc_parts if p),
        listing_type=listing_type,
        property_type=complex_node.get("@type") or "Development",
        property_usage="Residential",
        price=price,
        currency=currency,
        bedrooms=beds,
        area_sqm=area,
        city=city,
        district=addr.get("addressRegion"),
        country=country,
        latitude=to_float(geo.get("latitude")),
        longitude=to_float(geo.get("longitude")),
        developer="DarGlobal",
        project_name=name,
        completion_status=_status_line(status, completion, specs.get("Number of Floors")),
        amenities=([f"Brand: {brand_name}"] if brand_name else []) + amenities,
        images=[i for i in images if isinstance(i, str)][:8],
        published_at=listing.get("datePublished"),
    ).finalize()


def candidate_urls(all_urls: list[str]) -> list[str]:
    out = []
    for u in all_urls:
        path = u.replace("https://darglobal.co.uk", "").replace("http://darglobal.co.uk", "")
        path = path.rstrip("/") or "/"
        if path == "/":
            continue
        if any(path.lower().startswith(p) for p in EXCLUDE_PREFIXES):
            continue
        out.append(u)
    return sorted(set(out))


async def scrape(
    concurrency: int = 3, use_cache: bool = True, limit: int | None = None
) -> tuple[list[Property], list[str]]:
    results: list[Property] = []
    failures: list[str] = []

    # DarGlobal is WAF-protected and the corpus is small, so we crawl gently.
    async with Fetcher(rate=1.5, use_cache=use_cache) as f:
        sitemap = await f.get(SITEMAP, check_robots=False)
        all_urls = LOC_RE.findall(sitemap)
        cands = candidate_urls(all_urls)
        if limit:
            cands = cands[:limit]
        log.info("darglobal: %d sitemap urls -> %d project candidates", len(all_urls), len(cands))

        sem = asyncio.Semaphore(concurrency)
        done = 0
        skipped = 0

        async def one(url: str) -> None:
            nonlocal done, skipped
            async with sem:
                try:
                    html = await f.get(url)
                    prop = parse_project(html, url)
                    if prop:
                        results.append(prop)
                    else:
                        skipped += 1  # not a development page; expected
                except Exception as exc:  # noqa: BLE001
                    failures.append(url)
                    log.debug("darglobal fetch failed %s: %s", url, exc)
                finally:
                    done += 1
                    if done % 20 == 0:
                        log.info(
                            "darglobal progress %d/%d (projects=%d skipped=%d failed=%d)",
                            done, len(cands), len(results), skipped, len(failures),
                        )

        await asyncio.gather(*(one(u) for u in cands))
        log.info(
            "darglobal done: %d projects, %d non-project pages, %d failed",
            len(results), skipped, len(failures),
        )

    return results, failures
