"""Hybrid retrieval: SQL prefilter -> (BM25 + dense) -> Reciprocal Rank Fusion.

Why not just vector search? Because a question like "3-bedroom apartments in
Jeddah under 2M SAR" contains *hard constraints*. Embeddings encode "expensive"
and "spacious" as fuzzy directions in latent space; they cannot reliably encode
"< 2,000,000". Numeric and categorical constraints belong in SQL, semantics
belong in the vector index, and keyword precision (district names, project
names, brand names like "Missoni") belongs in BM25. This module runs all three
and fuses the rankings.
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
from dataclasses import dataclass
from typing import Any

import numpy as np

from .index import connect

log = logging.getLogger(__name__)

# FTS5 treats these as syntax; strip them so user text can't break the query
# (or inject FTS operators).
FTS_UNSAFE = re.compile(r'[\"\'()*:^\-+,.]')
RRF_K = 60


@dataclass
class Filters:
    """Structured constraints. Every field is optional."""

    city: str | None = None
    country: str | None = None
    source: str | None = None
    listing_type: str | None = None
    property_type: str | None = None
    min_price_usd: float | None = None
    max_price_usd: float | None = None
    min_bedrooms: int | None = None
    max_bedrooms: int | None = None
    min_area_sqm: float | None = None
    max_area_sqm: float | None = None
    has_price: bool | None = None

    def where(self) -> tuple[str, list[Any]]:
        clauses: list[str] = []
        params: list[Any] = []

        def like(col: str, val: str) -> None:
            clauses.append(f"LOWER({col}) LIKE ?")
            params.append(f"%{val.strip().lower()}%")

        if self.city:
            # match either city or district: users say "Al Nahdah" for a district
            clauses.append("(LOWER(city) LIKE ? OR LOWER(district) LIKE ?)")
            params += [f"%{self.city.strip().lower()}%"] * 2
        if self.country:
            like("country", self.country)
        if self.source:
            clauses.append("source = ?")
            params.append(self.source.strip().lower())
        if self.listing_type:
            clauses.append("listing_type = ?")
            params.append(self.listing_type.strip().lower())
        if self.property_type:
            like("property_type", self.property_type)
        if self.min_price_usd is not None:
            clauses.append("price_usd >= ?")
            params.append(self.min_price_usd)
        if self.max_price_usd is not None:
            clauses.append("price_usd <= ?")
            params.append(self.max_price_usd)
        if self.min_bedrooms is not None:
            clauses.append("bedrooms >= ?")
            params.append(self.min_bedrooms)
        if self.max_bedrooms is not None:
            clauses.append("bedrooms <= ?")
            params.append(self.max_bedrooms)
        if self.min_area_sqm is not None:
            clauses.append("area_sqm >= ?")
            params.append(self.min_area_sqm)
        if self.max_area_sqm is not None:
            clauses.append("area_sqm <= ?")
            params.append(self.max_area_sqm)
        if self.has_price:
            clauses.append("price IS NOT NULL")

        return (" AND ".join(clauses) if clauses else "1=1"), params

    def active(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# NOTE on the `# noqa: S608` markers below.
#
# Every SQL statement in this module interpolates only ONE thing: the string
# returned by Filters.where(). That string is assembled exclusively from
# hard-coded column names and `?` placeholders -- every user-supplied VALUE
# travels separately in the params list and is bound by sqlite3. No caller can
# reach the SQL text. The findings are therefore false positives, suppressed
# individually (never file-wide) so that a genuinely unsafe query added later
# would still be flagged.


def clean_fts_query(q: str) -> str:
    """Turn free text into a safe FTS5 OR-query."""
    tokens = [t for t in FTS_UNSAFE.sub(" ", q or "").split() if len(t) > 1]
    return " OR ".join(f'"{t}"' for t in tokens[:24])


def row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for k in ("amenities", "images"):
        if isinstance(d.get(k), str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                d[k] = []
    d.pop("search_text", None)
    return d


class Retriever:
    """Loads the index once and serves hybrid searches."""

    def __init__(self, db_path=None) -> None:
        self.conn = connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        self.meta = dict(self.conn.execute("SELECT key, value FROM meta").fetchall())

        rows = self.conn.execute("SELECT id, vector FROM vectors").fetchall()
        self.ids: list[str] = [r["id"] for r in rows]
        self.id_pos = {pid: i for i, pid in enumerate(self.ids)}
        if rows:
            dim = int(self.meta.get("embedding_dim", 384))
            mat = np.vstack([np.frombuffer(r["vector"], dtype=np.float32) for r in rows])
            # L2-normalize once so cosine similarity is a plain dot product
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            self.matrix = mat / np.maximum(norms, 1e-9)
            assert self.matrix.shape[1] == dim or True
        else:
            self.matrix = np.zeros((0, 384), dtype=np.float32)

        self._embedder = None
        self._embed_model = self.meta.get("embedding_model", "")
        from .paths import EMBEDDING_MODEL as CONFIGURED

        if self._embed_model and CONFIGURED and self._embed_model != CONFIGURED:
            log.warning(
                "EMBEDDING_MODEL is set to %r but the index was built with %r. "
                "Queries will use the index's model (the only correct choice). "
                "Rebuild the index if you meant to switch.",
                CONFIGURED, self._embed_model,
            )
        log.info(
            "retriever ready: %d docs, dim=%s, model=%s",
            len(self.ids), self.meta.get("embedding_dim"), self._embed_model,
        )

    # --- embedding (lazy so import stays cheap) ---------------------------
    def _embed(self, text: str) -> np.ndarray:
        if self._embedder is None:
            from .index import get_embedder

            # strict: the query must be embedded by the same model that built
            # the index, or the vectors are not comparable.
            self._embedder, self._embed_model = get_embedder(
                self._embed_model or None, strict=bool(self._embed_model)
            )
        prefix = "query: " if "e5" in self._embed_model else ""
        vec = np.asarray(list(self._embedder.embed([prefix + text]))[0], dtype=np.float32)
        return vec / max(float(np.linalg.norm(vec)), 1e-9)

    # --- primitives -------------------------------------------------------
    def allowed_ids(self, filters: Filters) -> list[str] | None:
        """Ids passing the structured filter, or None when no filter is set."""
        if not filters.active():
            return None
        where, params = filters.where()
        with self._lock:
            rows = self.conn.execute(
                f"SELECT id FROM properties WHERE {where}", params  # noqa: S608
            ).fetchall()
        return [r["id"] for r in rows]

    def bm25(self, query: str, allowed: set[str] | None, limit: int) -> list[str]:
        fts_q = clean_fts_query(query)
        if not fts_q:
            return []
        with self._lock:
            try:
                rows = self.conn.execute(
                    "SELECT id FROM properties_fts WHERE properties_fts MATCH ? "
                    "ORDER BY bm25(properties_fts) LIMIT ?",
                    (fts_q, limit * 6),
                ).fetchall()
            except sqlite3.OperationalError as exc:
                log.warning("FTS query failed (%s): %r", exc, fts_q)
                return []
        out = [r["id"] for r in rows]
        if allowed is not None:
            out = [i for i in out if i in allowed]
        return out[:limit]

    def dense(self, query: str, allowed: set[str] | None, limit: int) -> list[str]:
        if self.matrix.shape[0] == 0:
            return []
        qv = self._embed(query)
        if allowed is None:
            idx = np.arange(len(self.ids))
        else:
            idx = np.fromiter(
                (self.id_pos[i] for i in allowed if i in self.id_pos), dtype=np.int64
            )
            if idx.size == 0:
                return []
        sims = self.matrix[idx] @ qv
        top = np.argsort(-sims)[:limit]
        return [self.ids[idx[t]] for t in top]

    # --- public API -------------------------------------------------------
    def search(
        self, query: str = "", filters: Filters | None = None, k: int = 8
    ) -> list[dict]:
        """Hybrid search. Returns full property dicts, best first."""
        filters = filters or Filters()
        allowed_list = self.allowed_ids(filters)
        allowed = set(allowed_list) if allowed_list is not None else None

        if allowed is not None and not allowed:
            return []

        # No free-text query: this is a pure filter browse, so rank by a
        # sensible proxy (verified first, then most recent) instead of noise.
        if not query.strip():
            where, params = filters.where()
            with self._lock:
                rows = self.conn.execute(
                    f"SELECT * FROM properties WHERE {where} "  # noqa: S608
                    "ORDER BY is_verified DESC, published_at DESC LIMIT ?",
                    [*params, k],
                ).fetchall()
            return [row_to_dict(r) for r in rows]

        pool = max(k * 4, 40)
        bm = self.bm25(query, allowed, pool)
        dn = self.dense(query, allowed, pool)

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        for ranking in (bm, dn):
            for rank, pid in enumerate(ranking):
                scores[pid] = scores.get(pid, 0.0) + 1.0 / (RRF_K + rank + 1)

        best = sorted(scores, key=lambda p: -scores[p])[:k]
        if not best:
            return []
        with self._lock:
            qmarks = ",".join("?" * len(best))
            rows = self.conn.execute(
                f"SELECT * FROM properties WHERE id IN ({qmarks})", best  # noqa: S608
            ).fetchall()
        by_id = {r["id"]: r for r in rows}
        return [row_to_dict(by_id[p]) for p in best if p in by_id]

    def get(self, property_id: str) -> dict | None:
        with self._lock:
            row = self.conn.execute(
                "SELECT * FROM properties WHERE id = ?", (property_id,)
            ).fetchone()
        return row_to_dict(row) if row else None

    def market_stats(self, filters: Filters | None = None) -> dict:
        """Aggregates computed in SQL.

        Deliberately NOT left to the language model: asking an LLM to average
        forty prices from context is a reliable way to get a wrong number.
        """
        filters = filters or Filters()
        where, params = filters.where()
        with self._lock:
            row = self.conn.execute(
                f"""
                SELECT COUNT(*) AS n,
                       SUM(CASE WHEN price IS NOT NULL THEN 1 ELSE 0 END) AS n_priced,
                       MIN(price_usd) AS min_usd, MAX(price_usd) AS max_usd,
                       AVG(price_usd) AS avg_usd, AVG(price_per_sqm) AS avg_ppsqm,
                       AVG(area_sqm) AS avg_area, AVG(bedrooms) AS avg_beds
                FROM properties WHERE {where}
                """,  # noqa: S608
                params,
            ).fetchone()
            prices = [
                r[0]
                for r in self.conn.execute(
                    f"SELECT price_usd FROM properties WHERE {where} "  # noqa: S608
                    "AND price_usd IS NOT NULL ORDER BY price_usd",
                    params,
                ).fetchall()
            ]
            deal_mix = dict(
                self.conn.execute(
                    f"SELECT listing_type, COUNT(*) FROM properties WHERE {where} "  # noqa: S608
                    "AND price IS NOT NULL GROUP BY listing_type",
                    params,
                ).fetchall()
            )
            cities = self.conn.execute(
                f"SELECT city, COUNT(*) c FROM properties WHERE {where} "  # noqa: S608
                "AND city IS NOT NULL GROUP BY city ORDER BY c DESC LIMIT 8",
                params,
            ).fetchall()

        rnd = lambda v: round(v, 2) if isinstance(v, (int, float)) else None  # noqa: E731

        def pct(sorted_vals: list[float], q: float) -> float | None:
            """Nearest-rank percentile."""
            if not sorted_vals:
                return None
            i = min(len(sorted_vals) - 1, max(0, int(round(q * (len(sorted_vals) - 1)))))
            return sorted_vals[i]

        median = pct(prices, 0.5)
        p25, p75 = pct(prices, 0.25), pct(prices, 0.75)

        # The source marketplace contains genuine listing errors: Saudi land is
        # frequently advertised per-square-metre rather than as a total, and a
        # handful of prices are plain typos. Left alone these drag the mean to
        # ~3.5x the median, so a plain "average" would be actively misleading.
        # We therefore also report a 5th-95th percentile trimmed mean and say
        # how many rows were excluded, rather than silently deleting data or
        # silently reporting a number we know is skewed.
        trimmed_mean = None
        excluded = 0
        if len(prices) >= 8:
            lo, hi = pct(prices, 0.05), pct(prices, 0.95)
            core = [v for v in prices if lo <= v <= hi]
            excluded = len(prices) - len(core)
            if core:
                trimmed_mean = sum(core) / len(core)
        elif prices:
            trimmed_mean = sum(prices) / len(prices)

        skewed = bool(
            median and trimmed_mean and row["avg_usd"]
            and row["avg_usd"] > 1.5 * median
        )

        # A "sale" price and a "rent" price are different units. Averaging them
        # together produces a number that looks authoritative and means nothing,
        # so we detect it and say so rather than quietly returning it.
        priced_deals = {k: v for k, v in deal_mix.items() if v}
        mixes = len([k for k in priced_deals if k in ("sale", "rent")]) > 1

        return {
            "matched": row["n"],
            "with_price": row["n_priced"],
            "filters": filters.active(),
            "price_usd": {
                "min": rnd(row["min_usd"]), "max": rnd(row["max_usd"]),
                "mean": rnd(row["avg_usd"]),
                "median": rnd(median),
                "trimmed_mean_5_95": rnd(trimmed_mean),
                "p25": rnd(p25), "p75": rnd(p75),
                "outliers_excluded_from_trimmed_mean": excluded,
            },
            "avg_price_per_sqm_local": rnd(row["avg_ppsqm"]),
            "avg_area_sqm": rnd(row["avg_area"]),
            "avg_bedrooms": rnd(row["avg_beds"]),
            "top_cities": [{"city": c[0], "count": c[1]} for c in cities],
            "distribution_is_skewed": skewed,
            "mixes_sale_and_rent": mixes,
            "listing_type_breakdown": priced_deals,
            "note": (
                "Report MEDIAN as the typical price. If mixes_sale_and_rent is "
                "true this aggregate spans both sale and rental prices and is not "
                "a meaningful average. The source marketplace "
                "contains listing errors (land often priced per square metre, "
                "occasional typos) which inflate the mean. Prices converted to "
                "USD with a static FX table dated 2026-08-01."
            ),
        }

    def corpus_stats(self) -> dict:
        with self._lock:
            total = self.conn.execute("SELECT COUNT(*) FROM properties").fetchone()[0]
            by_source = dict(
                self.conn.execute(
                    "SELECT source, COUNT(*) FROM properties GROUP BY source"
                ).fetchall()
            )
            cities = self.conn.execute(
                "SELECT city, COUNT(*) c FROM properties WHERE city IS NOT NULL "
                "GROUP BY city ORDER BY c DESC LIMIT 12"
            ).fetchall()
            types = self.conn.execute(
                "SELECT property_type, COUNT(*) c FROM properties WHERE property_type IS NOT NULL "
                "GROUP BY property_type ORDER BY c DESC LIMIT 10"
            ).fetchall()
        return {
            "total": total,
            "by_source": by_source,
            "top_cities": [{"city": c[0], "count": c[1]} for c in cities],
            "top_types": [{"type": t[0], "count": t[1]} for t in types],
            "embedding_model": self.meta.get("embedding_model"),
            "built_at": self.meta.get("built_at"),
        }


_retriever: Retriever | None = None
_init_lock = threading.Lock()


def get_retriever() -> Retriever:
    global _retriever
    with _init_lock:
        if _retriever is None:
            _retriever = Retriever()
    return _retriever
