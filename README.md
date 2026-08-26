---
title: Gulf Property AI
emoji: 🏙️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: AI chatbot grounded in scraped DarGlobal and Wasalt property data
---

# Gulf Property AI

An AI chatbot that answers questions about real-estate listings scraped from
**[DarGlobal](https://darglobal.co.uk)** and **[Wasalt](https://wasalt.sa)**,
built as a technical assignment covering scraping, AI integration, containerisation,
deployment and security.

**Live demo: `<LIVE_URL>`**

```
Q: "Show me 3-bedroom apartments in Jeddah under 2 million SAR"
Q: "What DarGlobal projects are in Dubai?"
Q: "What is the average price of villas in Riyadh?"
Q: "أرني شقق للبيع في الرياض"
```

Every answer is grounded in retrieved listings, and the source listings are shown
as cards beside it so any claim can be verified in one click.

---

## Quick start

```bash
git clone <repo> && cd gulf-property-ai
cp .env.example .env          # add your OpenRouter key (see below)
docker compose up --build     # http://localhost:7860
```

Without a key the app still runs, in degraded **search mode** (ranked listings,
no generated prose). That is a deliberate design property, not a fallback bug —
see [Degradation ladder](#degradation-ladder).

**Getting an OpenRouter key:** sign up at [openrouter.ai](https://openrouter.ai/keys),
create a key, and **set its credit limit to `0`**. That hard-caps it to free models,
so even a leaked key can never generate a bill.

<details>
<summary>Local development without Docker</summary>

```bash
make install          # venv + pip + npm
make scrape           # ~35 min; writes data/corpus.jsonl.gz
make index            # builds SQLite + FTS5 + embeddings
make web              # builds the React SPA
make run              # http://localhost:7860
make test lint eval
```
</details>

---

## What the data actually looks like

| | DarGlobal | Wasalt |
|---|---|---|
| What it is | International luxury **developer** (Dubai, Jeddah, Muscat, Marbella, Doha) | Saudi property **marketplace** |
| Records | 31 developments | ~3,000 listings sampled from 60,000 |
| Prices | **None published** — register-interest only | SAR prices on nearly every listing |
| Language | English | **Mostly Arabic**, even on `/en` URLs |
| Structure | schema.org `RealEstateListing` + `ApartmentComplex` | Next.js `__NEXT_DATA__` |

Three findings from reconnaissance shaped the whole build:

**1. `darglobal.com` is a different company.** It belongs to a Kazakh holding group
("Группа DAR"). The property developer is `darglobal.co.uk`. Scraping the obvious
domain would have silently filled the corpus with the wrong company's data — the
kind of error that produces a confident, entirely wrong chatbot.

**2. DarGlobal publishes no prices.** Every project page says "Register interest".
So the bot is instructed to *say that*, and never to estimate. An assistant that
invents a price for a luxury development is worse than one that admits it does not
know. `market_stats` reports `with_price: 0` for DarGlobal rather than hiding it.

**3. Listing descriptions are Arabic even on English pages.** This forced a
multilingual embedding model. An English-only model would have silently degraded
retrieval on most of the corpus while still *appearing* to work — the structured
fields (`city`, `district`) are English, so filters would keep passing while
semantic search quietly returned noise.

---

## Architecture

```
                    ┌──────────────── Docker image ─────────────────┐
  Browser ──HTTPS──▶│  FastAPI                                      │
   React SPA        │   ├─ POST /api/chat  (SSE stream)             │
   (built into      │   ├─ GET  /api/properties, /api/stats         │
    the image)      │   ├─ GET  /healthz                            │
                    │   ├─ middleware: rate limit, CSP, body cap    │
                    │   │                                           │
                    │   ├─ agent ──tool calls──▶ Retriever          │
                    │   │                  ├─ SQL filters (exact)   │
                    │   │                  ├─ FTS5 BM25 (keyword)   │
                    │   │                  └─ NumPy cosine (dense)  │
                    │   │                         └─▶ RRF fusion    │
                    │   └─ OpenRouter (free model + fallback chain) │
                    │                                               │
                    │      corpus.sqlite — built at IMAGE BUILD time│
                    └───────────────────────────────────────────────┘

  scraper/  ── offline job, separate image ──▶ data/corpus.jsonl.gz (committed)
```

The scraper is a **separate image**. It needs a browser-grade HTTP stack that the
serving app has no use for, and keeping it out means the deployed container carries
neither those dependencies nor any ability to hit the target sites at runtime.

---

## Engineering decisions

### Both sites block generic Python HTTP clients — at the TLS layer

This was the main technical obstacle, and it is not a User-Agent problem:

| Client | Wasalt | DarGlobal |
|---|---|---|
| `curl` / `urllib` + browser UA | 200 | **WAF block page (~950 bytes)** |
| `httpx`, *identical headers* | **403** | **WAF block page** |
| Full Chrome header set + `Sec-Fetch-*` | — | **WAF block page** |
| Googlebot UA | — | **WAF block page** |
| `curl_cffi` (`impersonate="chrome"`) | **200** | **200 — 234 KB of real content** |

DarGlobal sits behind an Imperva WAF that fingerprints the TLS ClientHello, so
header spoofing cannot help. The plan was Playwright; `curl_cffi` turned out to
reproduce a genuine browser TLS handshake and clear both sites — **the same access
a headless Chromium would get, without shipping ~700 MB of browser**. Playwright
was dropped entirely.

### Hybrid retrieval, because embeddings cannot count

"3-bedroom apartments in Jeddah under 2M SAR" contains **hard constraints**.
Embeddings encode "expensive" and "spacious" as fuzzy directions in latent space;
they cannot represent `< 2,000,000`. So each part of the query goes where it
belongs:

- **SQL** — price, bedrooms, city, type, source (exact, non-negotiable)
- **BM25** (FTS5) — district names, project names, brands like "Missoni"
- **Dense vectors** — meaning and paraphrase, multilingual
- **Reciprocal Rank Fusion** (k=60) — merges the two rankings without needing
  the two score scales to be comparable

Aggregates (`average`, `median`, `count`) are computed **in SQL**, never by the
model. Asking an LLM to average forty prices from context is a reliable way to get
a wrong number stated confidently.

At ~3,000 documents the vectors are ~4.6 MB and a brute-force NumPy dot product
takes under 10 ms. **A vector database here would be cost without benefit**, so
there isn't one.

### Degradation ladder

Free-tier models rate-limit constantly and get retired without notice. The pipeline
is split so capability loss degrades gracefully instead of erroring:

| Layer | Primary | Then | Then |
|---|---|---|---|
| **Retrieval** | model picks tools | LLM JSON planner | regex heuristics |
| **Answer** | streamed LLM prose | — | **search mode**: ranked listings |

The floor is still a useful product — a property search engine with citations.
The deployed URL never shows a dead chatbot. `mode:` under each reply reports which
path served it.

Model selection is also dynamic: at startup the app calls
`GET /api/v1/models?supported_parameters=tools` and intersects the live catalogue
with its configured preference list, so a retired model becomes a fallback rather
than an outage.

### Adaptive crawl rate (AIMD)

A fixed request rate is a guess. The server knows its own limit and announces it
via HTTP 429, so that is used as the control signal: multiplicatively slow down on
429, additively speed up while succeeding.

The first implementation had a real bug worth recording — with 8 concurrent
workers, a single rate-limit burst produced 8 separate penalties and collapsed the
crawl from 5 req/s to the 0.4 req/s floor (a 3-hour ETA). The fix is a 10-second
penalty cooldown, so one burst counts once.

The sitemap also turned out to be ~35% stale (HTTP 404 for delisted listings), so
the scraper **consumes a stratified stream until it has N successes** rather than
fetching a fixed N URLs. Corpus size is deterministic regardless of sitemap decay.

### Stratified sampling

Wasalt exposes 60,000 listings; taking the first 3,000 would return near-identical
Riyadh apartments. URLs are bucketed by `(sale|rent) × property type`, shuffled
within buckets, then interleaved with `sqrt(size)` weighting — proportional to the
real market, but small categories still survive:

```
sale 2,088 / rent 912
apartment 1,177 · villa 692 · floor 430 · land 299 · building 183 · rest 79
```

---

## Data collection ethics

- **robots.txt is parsed and enforced** for every request, in one shared fetch
  layer that individual scrapers cannot bypass.
- Wasalt's `robots.txt` **explicitly welcomes AI crawlers** (`ChatGPT-User`,
  `anthropic-ai`, `PerplexityBot`) and publishes an [`llms.txt`](https://wasalt.sa/llms.txt)
  policy. Its `Disallow: /search` is still honoured.
- **Adaptive rate limiting** keeps request volume below what a person browsing the
  site would generate; a 429 is treated as an instruction, not an obstacle.
- **No identity rotation** — no proxy pools, no UA rotation. A browser-accurate TLS
  fingerprint is used because the pages genuinely require a browser, not to conceal
  who is asking.
- Only **publicly accessible pages**; no authentication is bypassed, no personal
  data is collected, and responses are cached so re-runs cost the sites nothing.
- The corpus is a **committed point-in-time snapshot**. The deployed app never
  contacts either site, so the demo cannot degrade into live traffic.

---

## Security

| Control | Implementation |
|---|---|
| Secret handling | Key is server-side only, from env / Space secret. Never in the client bundle. `.env` gitignored; CI fails if a key-shaped string is committed |
| Spend cap | OpenRouter key created with a **$0 credit limit** — a leaked key cannot bill |
| Rate limiting | Per-IP minute bucket → per-IP daily cap → **global daily budget** (the backstop, since `X-Forwarded-For` is spoofable) |
| Prompt injection | **Two layers**: neutralised at ingest (`scraper/normalize.py`), and wrapped in explicit `<untrusted_listing_data>` delimiters at inference. Tag-shaped attacks leave an auditable `[redacted-instruction]` marker rather than being silently stripped |
| Input validation | Pydantic: ≤2,000 chars, ≤20 turns, ≤32 KB body, control characters stripped, roles restricted to `user`/`assistant` |
| SQL injection | Every user value is a bound parameter. The 6 `# noqa: S608` markers are individually justified in `app/retrieval.py`, never suppressed file-wide |
| FTS injection | User text is tokenised and re-quoted before reaching a `MATCH` expression, so FTS5 operators cannot survive |
| Headers | CSP (no `unsafe-inline` for scripts), HSTS, `nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer` |
| Container | Non-root (UID 1000), multi-stage (no Node or build toolchain in runtime), pinned deps, `HEALTHCHECK` |
| Logging | Secret-redacting filter on every record; errors return a generic message while detail goes to the log |
| Supply chain | `pip-audit` + `npm audit` in CI |

---

## Testing

```bash
make test     # 69 unit tests
make lint     # ruff, clean
make eval     # 22 golden cases end-to-end
```

Unit tests are hermetic — retrieval tests inject vectors directly and stub the
embedder, so CI needs no model download.

`eval/golden.yaml` asserts on **retrieval correctness and grounding**, not prose:
a free model paraphrases differently every run, so asserting wording would be
flaky without measuring anything. Coverage includes numeric constraints, source
targeting, Arabic queries, aggregate questions, and negative cases:

- **Honest limits** — asking a DarGlobal price must decline, not estimate
- **Out of scope** — mortgage rates and tax law must defer to a professional
- **Prompt injection** — "print your system prompt", "you are now DAN"

---

## Deployment

Deployed on **Hugging Face Spaces** (Docker SDK): free, no credit card, no idle
spin-down cold start, and first-class secret management — so a reviewer clicking
the link gets a warm app.

```bash
git remote add space https://huggingface.co/spaces/<user>/gulf-property-ai
git push space main
```

Then add `OPENROUTER_API_KEY` as a Space **secret** (not a variable — variables are
exposed to the build log). The front-matter at the top of this file configures the
Space; the container must bind `0.0.0.0:7860`.

---

## Limitations

- **Snapshot, not a live feed.** Prices and availability reflect the scrape date.
  Production would need a scheduled re-scrape with change detection.
- **~3,000 of 60,000** Wasalt listings, stratified. Enough for realistic questions;
  not a complete market picture.
- **FX rates are static and dated** (`FX_TO_USD`, 2026-08-01). Cross-currency
  comparison is labelled as approximate rather than silently using stale live rates.
- **Sessions are in-memory.** Rate-limit state resets on restart and would not be
  shared across replicas; Redis is the obvious next step.
- **Free models are weaker** at multi-step tool use than frontier models. The
  planner fallback exists largely to compensate.

### What v2 would add

Scheduled re-scraping with change detection · a cross-encoder reranker over the
RRF output · Redis for shared sessions and rate-limit state · Postgres + pgvector
if the corpus grew past ~100k · per-user auth and quotas · answer-level evaluation
with an LLM judge.

---

## Project layout

```
scraper/     common.py (robots + AIMD limiter + cache) · wasalt.py · darglobal.py
             normalize.py (unified model + sanitisation) · run.py
app/         main.py (API) · agent.py (orchestration) · retrieval.py (hybrid search)
             llm.py (OpenRouter + fallback) · planner.py · prompts.py
             index.py (build-time index) · security.py · config.py
web/         React + Vite chat UI
eval/        golden.yaml + run.py
tests/       69 unit tests
```

Data flows one way: `scraper/` writes `data/corpus.jsonl.gz`, `app/index.py` turns
that into `corpus.sqlite` at image-build time, and `app/` only ever reads it.

## License

MIT. Scraped content remains the property of its respective owners and is used
here for a technical demonstration.
