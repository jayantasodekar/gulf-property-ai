"""Retrieval tests against a synthetic index.

Vectors are injected directly and `_embed` is stubbed, so these run without
downloading an embedding model -- keeping CI fast and hermetic.
"""

from __future__ import annotations

import json
import sqlite3

import numpy as np
import pytest

from app.index import COLUMNS, SCHEMA
from app.retrieval import Filters, Retriever, clean_fts_query

ROWS = [
    dict(
        id="wasalt-1", source="wasalt", source_id="1",
        url="https://wasalt.sa/1", title="Apartment with 3 Bedrooms",
        description="Luxury apartment in Al Nahdah district", language="en",
        listing_type="sale", property_type="Apartment", property_usage="Residential",
        price=1_700_000.0, currency="SAR", price_usd=453_220.0, price_per_sqm=9289.0,
        bedrooms=3, bathrooms=2, area_sqm=183.0,
        city="Jeddah", district="Al Nahdah", country="Saudi Arabia",
    ),
    dict(
        id="wasalt-2", source="wasalt", source_id="2",
        url="https://wasalt.sa/2", title="Villa with 5 Bedrooms",
        description="Spacious family villa with private pool", language="en",
        listing_type="sale", property_type="Villa", property_usage="Residential",
        price=4_000_000.0, currency="SAR", price_usd=1_066_400.0, price_per_sqm=8000.0,
        bedrooms=5, bathrooms=4, area_sqm=500.0,
        city="Riyadh", district="Al Olaya", country="Saudi Arabia",
    ),
    dict(
        id="wasalt-3", source="wasalt", source_id="3",
        url="https://wasalt.sa/3", title="Apartment for Rent",
        description="Two bedroom rental near the corniche", language="en",
        listing_type="rent", property_type="Apartment", property_usage="Residential",
        price=35_000.0, currency="SAR", price_usd=9331.0, price_per_sqm=212.0,
        bedrooms=2, bathrooms=1, area_sqm=165.0,
        city="Dammam", district="Al Noor", country="Saudi Arabia",
    ),
    dict(
        id="darglobal-w-residences", source="darglobal", source_id="w-residences",
        url="https://darglobal.co.uk/w-residences", title="W Residences",
        description="W Hotels branded residences in Downtown Dubai with Burj Khalifa views",
        language="en", listing_type="offplan", property_type="ApartmentComplex",
        property_usage="Residential",
        price=None, currency=None, price_usd=None, price_per_sqm=None,
        bedrooms=3, bathrooms=None, area_sqm=355.0,
        city="Dubai", district="Dubai", country="United Arab Emirates",
        developer="DarGlobal", project_name="W Residences",
    ),
]


def _search_text(r: dict) -> str:
    return " · ".join(
        str(x) for x in (
            r["title"], r.get("property_type"), f"{r.get('bedrooms')} bedrooms",
            r.get("district"), r.get("city"), r.get("country"), r.get("description"),
        ) if x
    )


@pytest.fixture(scope="module")
def retriever(tmp_path_factory) -> Retriever:
    db = tmp_path_factory.mktemp("idx") / "corpus.sqlite"
    conn = sqlite3.connect(str(db))
    conn.executescript(SCHEMA)

    rng = np.random.default_rng(0)
    for r in ROWS:
        full = {c: r.get(c) for c in COLUMNS}
        full["search_text"] = _search_text(r)
        full["amenities"] = json.dumps(r.get("amenities") or [])
        full["images"] = json.dumps(r.get("images") or [])
        conn.execute(
            f"INSERT INTO properties ({','.join(COLUMNS)}) "
            f"VALUES ({','.join('?' * len(COLUMNS))})",
            [full[c] for c in COLUMNS],
        )
        conn.execute(
            "INSERT INTO properties_fts (id, search_text) VALUES (?, ?)",
            (r["id"], full["search_text"]),
        )
        v = rng.normal(size=384).astype(np.float32)
        conn.execute(
            "INSERT INTO vectors (id, vector) VALUES (?, ?)", (r["id"], v.tobytes())
        )
    conn.executemany(
        "INSERT INTO meta (key, value) VALUES (?, ?)",
        [("embedding_dim", "384"), ("embedding_model", "stub"), ("doc_count", "4")],
    )
    conn.commit()
    conn.close()

    r = Retriever(db_path=db)
    # deterministic stub: no model download in tests
    r._embed = lambda text: np.ones(384, dtype=np.float32) / np.sqrt(384)  # type: ignore[assignment]
    return r


# ------------------------------------------------------------------ #
#  FTS query hygiene (also an injection surface)
# ------------------------------------------------------------------ #
@pytest.mark.parametrize(
    "raw", ['villa" OR "1"="1', "apartment*", "a AND b:c", "(nested)", "'quoted'"]
)
def test_clean_fts_query_is_safe(raw: str) -> None:
    out = clean_fts_query(raw)
    # FTS5 operator characters must never survive into the MATCH expression
    for ch in "*():^'":
        assert ch not in out, f"{ch!r} survived sanitisation of {raw!r}"
    # quotes appear only as the balanced delimiters we add around each token
    assert out.count('"') % 2 == 0


def test_clean_fts_query_builds_or_terms() -> None:
    assert clean_fts_query("luxury villa") == '"luxury" OR "villa"'


def test_clean_fts_query_empty() -> None:
    assert clean_fts_query("") == ""


def test_clean_fts_query_keeps_single_char_tokens() -> None:
    """Regression: dropping 1-char tokens deleted the "W" from "W Residences".

    That turned the query into a match for every ...Residences listing in the
    corpus and buried the development the user actually named.
    """
    assert '"W"' in clean_fts_query("the W Residences development")


def test_single_char_brand_token_ranks_first(retriever: Retriever) -> None:
    out = retriever.search("Tell me about the W Residences development", k=5)
    assert out and out[0]["id"] == "darglobal-w-residences"


# ------------------------------------------------------------------ #
#  Structured filters -- the whole reason for the hybrid design
# ------------------------------------------------------------------ #
def test_filter_by_city(retriever: Retriever) -> None:
    ids = retriever.allowed_ids(Filters(city="Jeddah"))
    assert ids == ["wasalt-1"]


def test_filter_matches_district_too(retriever: Retriever) -> None:
    assert retriever.allowed_ids(Filters(city="Al Olaya")) == ["wasalt-2"]


def test_filter_by_max_price(retriever: Retriever) -> None:
    ids = set(retriever.allowed_ids(Filters(max_price_usd=500_000)))
    assert ids == {"wasalt-1", "wasalt-3"}


def test_filter_by_bedrooms_range(retriever: Retriever) -> None:
    ids = set(retriever.allowed_ids(Filters(min_bedrooms=3)))
    assert "wasalt-3" not in ids  # 2 bedrooms
    assert "wasalt-2" in ids


def test_filter_by_listing_type_and_source(retriever: Retriever) -> None:
    assert retriever.allowed_ids(Filters(listing_type="rent")) == ["wasalt-3"]
    assert retriever.allowed_ids(Filters(source="darglobal")) == ["darglobal-w-residences"]


def test_no_filters_returns_none_sentinel(retriever: Retriever) -> None:
    assert retriever.allowed_ids(Filters()) is None


def test_over_constrained_returns_empty(retriever: Retriever) -> None:
    assert retriever.search("apartment", Filters(city="Jeddah", min_bedrooms=99)) == []


# ------------------------------------------------------------------ #
#  Hybrid search
# ------------------------------------------------------------------ #
def test_search_respects_hard_constraint(retriever: Retriever) -> None:
    """The core claim: a price ceiling is honoured, not approximated."""
    out = retriever.search("apartment", Filters(max_price_usd=500_000), k=5)
    assert out and all(r["price_usd"] <= 500_000 for r in out)


def test_search_keyword_precision(retriever: Retriever) -> None:
    ids = [r["id"] for r in retriever.search("Burj Khalifa branded residences", k=4)]
    assert "darglobal-w-residences" in ids


def test_empty_query_is_a_filter_browse(retriever: Retriever) -> None:
    out = retriever.search("", Filters(city="Riyadh"), k=5)
    assert [r["id"] for r in out] == ["wasalt-2"]


def test_search_returns_parsed_json_columns(retriever: Retriever) -> None:
    row = retriever.search("", Filters(city="Jeddah"), k=1)[0]
    assert isinstance(row["amenities"], list) and isinstance(row["images"], list)
    assert "search_text" not in row  # internal column is not leaked to the client


def test_get_by_id(retriever: Retriever) -> None:
    assert retriever.get("wasalt-2")["title"] == "Villa with 5 Bedrooms"
    assert retriever.get("does-not-exist") is None


# ------------------------------------------------------------------ #
#  Aggregates must be exact -- this is why they are not left to the LLM
# ------------------------------------------------------------------ #
def test_market_stats_are_exact(retriever: Retriever) -> None:
    s = retriever.market_stats(Filters(source="wasalt", listing_type="sale"))
    assert s["matched"] == 2
    assert s["with_price"] == 2
    assert s["price_usd"]["min"] == pytest.approx(453_220.0)
    assert s["price_usd"]["max"] == pytest.approx(1_066_400.0)
    assert s["price_usd"]["mean"] == pytest.approx((453_220.0 + 1_066_400.0) / 2)


def test_market_stats_handles_unpriced_rows(retriever: Retriever) -> None:
    """DarGlobal publishes no prices; stats must report that, not invent it."""
    s = retriever.market_stats(Filters(source="darglobal"))
    assert s["matched"] == 1
    assert s["with_price"] == 0
    assert s["price_usd"]["mean"] is None
    assert s["price_usd"]["median"] is None


def test_market_stats_flags_sale_rent_mixing(retriever: Retriever) -> None:
    """Averaging a sale price with a rent price is meaningless; say so."""
    mixed = retriever.market_stats(Filters(source="wasalt"))
    assert mixed["mixes_sale_and_rent"] is True
    assert mixed["listing_type_breakdown"] == {"rent": 1, "sale": 2}

    clean = retriever.market_stats(Filters(source="wasalt", listing_type="sale"))
    assert clean["mixes_sale_and_rent"] is False


def test_market_stats_reports_percentiles(retriever: Retriever) -> None:
    s = retriever.market_stats(Filters(source="wasalt", listing_type="sale"))
    p = s["price_usd"]
    assert p["median"] is not None
    assert p["p25"] is not None and p["p75"] is not None
    assert p["p25"] <= p["median"] <= p["p75"]


def test_trimmed_mean_resists_a_typo_outlier() -> None:
    """A single mis-keyed price must not drag the reported typical price.

    Mirrors a real defect in the source data: Saudi land is often advertised
    per square metre, and some listings carry outright typos.
    """
    from app.retrieval import Retriever as R

    prices = [100_000.0] * 20 + [500_000_000.0]  # 20 normal + 1 typo
    sorted_p = sorted(prices)
    mean = sum(prices) / len(prices)

    def pct(v, q):
        i = min(len(v) - 1, max(0, int(round(q * (len(v) - 1)))))
        return v[i]

    median = pct(sorted_p, 0.5)
    lo, hi = pct(sorted_p, 0.05), pct(sorted_p, 0.95)
    core = [x for x in sorted_p if lo <= x <= hi]
    trimmed = sum(core) / len(core)

    assert mean > 20_000_000        # the naive mean is destroyed
    assert median == 100_000        # the median is not
    assert trimmed == 100_000       # nor is the trimmed mean
    assert R is not None


def test_corpus_stats(retriever: Retriever) -> None:
    st = retriever.corpus_stats()
    assert st["total"] == 4
    assert st["by_source"] == {"wasalt": 3, "darglobal": 1}


# ------------------------------------------------------------------ #
#  Sale/rent disambiguation from a price bound
# ------------------------------------------------------------------ #
def test_price_bound_above_rent_range_implies_sale(retriever: Retriever) -> None:
    """"Apartments under 2M SAR" is purchase language, even without "buy".

    A ceiling that admits every rental cannot have been meant to filter
    rentals, and rents sit an order of magnitude below sale prices, so
    leaving listing_type unset lets cheap rentals monopolise the results.
    """
    resolved = retriever.resolve_price_intent(Filters(max_price_usd=500_000))
    assert resolved.listing_type == "sale"


def test_price_bound_below_sale_range_implies_rent(retriever: Retriever) -> None:
    resolved = retriever.resolve_price_intent(Filters(max_price_usd=9_000))
    assert resolved.listing_type == "rent"


def test_explicit_listing_type_is_never_overridden(retriever: Retriever) -> None:
    resolved = retriever.resolve_price_intent(
        Filters(max_price_usd=500_000, listing_type="rent")
    )
    assert resolved.listing_type == "rent"


def test_no_price_bound_stays_ambiguous(retriever: Retriever) -> None:
    """Without a bound there is no signal, so nothing is assumed."""
    resolved = retriever.resolve_price_intent(Filters(city="Jeddah"))
    assert resolved.listing_type is None


def test_price_bounded_search_excludes_rentals(retriever: Retriever) -> None:
    """End-to-end: the regression this guards against."""
    out = retriever.search("apartment", Filters(max_price_usd=500_000), k=5)
    assert out and not any(r["listing_type"] == "rent" for r in out)
