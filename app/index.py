"""Build the searchable corpus: SQLite table + FTS5 index + dense vectors.

Run at Docker *build* time so the container starts instantly:

    python -m app.index --build
    python -m app.index --verify

Design note: with a few thousand listings the vectors are only a handful of
megabytes, so a brute-force NumPy cosine over an in-memory matrix takes well
under 10 ms. Adding a vector database here would be cost without benefit.
"""

from __future__ import annotations

import argparse
import gzip
import json
import logging
import os
import sqlite3
from collections.abc import Iterable
from pathlib import Path

import numpy as np

from .config import settings

log = logging.getLogger(__name__)

COLUMNS = [
    "id", "source", "source_id", "url", "title", "description", "language",
    "listing_type", "property_type", "property_usage", "price", "currency",
    "price_usd", "price_per_sqm", "bedrooms", "bathrooms", "area_sqm",
    "city", "district", "country", "latitude", "longitude", "developer",
    "project_name", "completion_status", "amenities", "images",
    "is_verified", "published_at", "scraped_at", "search_text",
]

SCHEMA = """
PRAGMA journal_mode=WAL;

CREATE TABLE IF NOT EXISTS properties (
    id            TEXT PRIMARY KEY,
    source        TEXT NOT NULL,
    source_id     TEXT,
    url           TEXT,
    title         TEXT,
    description   TEXT,
    language      TEXT,
    listing_type  TEXT,
    property_type TEXT,
    property_usage TEXT,
    price         REAL,
    currency      TEXT,
    price_usd     REAL,
    price_per_sqm REAL,
    bedrooms      INTEGER,
    bathrooms     INTEGER,
    area_sqm      REAL,
    city          TEXT,
    district      TEXT,
    country       TEXT,
    latitude      REAL,
    longitude     REAL,
    developer     TEXT,
    project_name  TEXT,
    completion_status TEXT,
    amenities     TEXT,
    images        TEXT,
    is_verified   INTEGER,
    published_at  TEXT,
    scraped_at    TEXT,
    search_text   TEXT
);

CREATE INDEX IF NOT EXISTS idx_city     ON properties(city);
CREATE INDEX IF NOT EXISTS idx_source   ON properties(source);
CREATE INDEX IF NOT EXISTS idx_deal     ON properties(listing_type);
CREATE INDEX IF NOT EXISTS idx_price    ON properties(price_usd);
CREATE INDEX IF NOT EXISTS idx_beds     ON properties(bedrooms);
CREATE INDEX IF NOT EXISTS idx_ptype    ON properties(property_type);

CREATE VIRTUAL TABLE IF NOT EXISTS properties_fts USING fts5(
    id UNINDEXED,
    search_text,
    tokenize='unicode61 remove_diacritics 2'
);

CREATE TABLE IF NOT EXISTS vectors (
    id     TEXT PRIMARY KEY,
    vector BLOB NOT NULL
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);
"""


def connect(path: Path | None = None) -> sqlite3.Connection:
    p = path or settings.db_path
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def iter_corpus(path: Path) -> Iterable[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[operator]
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def _search_text(rec: dict) -> str:
    """Mirror of Property.search_text() for raw dicts."""
    parts = [
        rec.get("title") or "",
        rec.get("property_type") or "",
        rec.get("property_usage") or "",
        f"{rec['bedrooms']} bedrooms" if rec.get("bedrooms") else "",
        f"{rec['bathrooms']} bathrooms" if rec.get("bathrooms") else "",
        f"{rec['area_sqm']:.0f} sqm" if rec.get("area_sqm") else "",
        rec.get("district") or "",
        rec.get("city") or "",
        rec.get("country") or "",
        rec.get("project_name") or "",
        rec.get("developer") or "",
        rec.get("completion_status") or "",
        " ".join((rec.get("amenities") or [])[:20]),
        (rec.get("description") or "")[:1200],
    ]
    return " · ".join(p for p in parts if p).strip()


def get_embedder(model_name: str | None = None):
    """Load the embedding model, falling back through multilingual options.

    Listing descriptions are frequently Arabic even on the English pages, so an
    English-only model would silently degrade half the corpus.
    """
    from fastembed import TextEmbedding

    preferred = [
        model_name or settings.embedding_model,
        "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",  # 384d, 0.22GB
        "intfloat/multilingual-e5-large",  # 1024d, 2.24GB - better but heavy
        "BAAI/bge-small-en-v1.5",  # last resort: English-only
    ]
    available = {m["model"] for m in TextEmbedding.list_supported_models()}
    # ONNX Runtime defaults to a single thread here, which makes a full corpus
    # rebuild ~4x slower than it needs to be and risks tripping cloud build
    # timeouts. Use every core the builder has.
    threads = os.cpu_count() or 4
    for name in preferred:
        if name in available:
            log.info("embedding model: %s (threads=%d)", name, threads)
            return TextEmbedding(model_name=name, threads=threads), name
    raise RuntimeError(f"no usable embedding model; available={sorted(available)[:10]}")


def build(corpus_path: Path | None = None, db_path: Path | None = None) -> dict:
    corpus_path = corpus_path or settings.corpus_path
    db_path = db_path or settings.db_path
    if not corpus_path.exists():
        raise FileNotFoundError(f"corpus not found: {corpus_path}. Run scraper/run.py first.")

    if db_path.exists():
        db_path.unlink()
    conn = connect(db_path)
    conn.executescript(SCHEMA)

    records = list(iter_corpus(corpus_path))
    log.info("loaded %d records from %s", len(records), corpus_path.name)

    rows, fts_rows, texts = [], [], []
    for r in records:
        st = _search_text(r)
        texts.append(st)
        fts_rows.append((r["id"], st))
        rows.append(
            tuple(
                json.dumps(r.get(c), ensure_ascii=False)
                if c in ("amenities", "images")
                else (st if c == "search_text" else r.get(c))
                for c in COLUMNS
            )
        )

    placeholders = ",".join("?" * len(COLUMNS))
    # COLUMNS is a module-level constant, not user input; values are bound.
    conn.executemany(
        f"INSERT OR REPLACE INTO properties ({','.join(COLUMNS)}) "  # noqa: S608
        f"VALUES ({placeholders})",
        rows,
    )
    conn.executemany("INSERT INTO properties_fts (id, search_text) VALUES (?, ?)", fts_rows)
    conn.commit()
    log.info("inserted %d rows + FTS index", len(rows))

    embedder, model_name = get_embedder()
    log.info("embedding %d documents ...", len(texts))
    # e5 models expect a "passage: " prefix on indexed documents.
    prefix = "passage: " if "e5" in model_name else ""
    vectors = list(embedder.embed([prefix + t for t in texts], batch_size=64))
    dim = len(vectors[0])
    conn.executemany(
        "INSERT OR REPLACE INTO vectors (id, vector) VALUES (?, ?)",
        [
            (r["id"], np.asarray(v, dtype=np.float32).tobytes())
            for r, v in zip(records, vectors, strict=False)
        ],
    )
    conn.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [
            ("embedding_model", model_name),
            ("embedding_dim", str(dim)),
            ("doc_count", str(len(records))),
            ("built_at", __import__("datetime").datetime.utcnow().isoformat(timespec="seconds")),
        ],
    )
    conn.commit()
    conn.close()

    stats = {"records": len(records), "model": model_name, "dim": dim}
    log.info("index built: %s", stats)
    return stats


def verify(db_path: Path | None = None) -> dict:
    conn = connect(db_path)
    q = lambda s: conn.execute(s).fetchone()[0]  # noqa: E731
    stats = {
        "properties": q("SELECT COUNT(*) FROM properties"),
        "fts": q("SELECT COUNT(*) FROM properties_fts"),
        "vectors": q("SELECT COUNT(*) FROM vectors"),
        "sources": dict(
            conn.execute("SELECT source, COUNT(*) FROM properties GROUP BY source").fetchall()
        ),
        "meta": dict(conn.execute("SELECT key, value FROM meta").fetchall()),
    }
    conn.close()
    ok = stats["properties"] == stats["fts"] == stats["vectors"] and stats["properties"] > 0
    stats["consistent"] = ok
    return stats


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-5s | %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args()
    if a.build or not a.verify:
        build()
    print(json.dumps(verify(), indent=2, ensure_ascii=False))
