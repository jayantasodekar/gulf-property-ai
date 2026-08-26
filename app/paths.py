"""Settings that determine the CONTENT of the built index.

Deliberately separate from config.py.

The Docker image builds the SQLite + vector index in an expensive (~12 minute)
layer. That layer must be invalidated when something changes what the index
*contains* -- the corpus, or the embedding model -- and must NOT be invalidated
by ordinary application changes such as swapping an LLM candidate or adjusting
a rate limit. Splitting these few values out of config.py is what lets the
Dockerfile copy only `paths.py` + `index.py` into the cached layer.
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# config.py loads .env through pydantic-settings, but this module is imported
# by the index builder without config.py (that is the point of the split), so
# it loads .env itself. Without this, a local .env would silently fail to
# override EMBEDDING_MODEL here while appearing to work everywhere else.
try:
    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env", override=False)
except ImportError:  # python-dotenv is optional at build time
    pass

DB_PATH = Path(os.getenv("DB_PATH", ROOT / "data" / "corpus.sqlite"))
CORPUS_PATH = Path(os.getenv("CORPUS_PATH", ROOT / "data" / "corpus.jsonl.gz"))

# Listing descriptions are frequently Arabic even on the English pages, so the
# default is multilingual. An English-only model roughly a third of the size is
# viable where memory is tight, because FTS5 (unicode61) indexes Arabic script
# correctly and BM25 alone still retrieves Arabic listings -- only semantic
# paraphrase matching degrades. See README "Deployment".
EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
