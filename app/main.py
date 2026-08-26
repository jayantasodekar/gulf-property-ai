"""FastAPI application: chat API + static SPA, in one container."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .agent import answer
from .config import settings
from .llm import client
from .retrieval import Filters, get_retriever
from .security import (
    BodySizeLimitMiddleware,
    ChatRequest,
    RedactingFilter,
    SecurityHeadersMiddleware,
    client_key,
    limiter,
)

logging.basicConfig(
    level=getattr(logging, settings.log_level.upper(), logging.INFO),
    format="%(asctime)s %(levelname)-5s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logging.getLogger().addFilter(RedactingFilter())
for noisy in ("httpx", "httpcore", "curl_cffi"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

log = logging.getLogger("app")


async def _warm_embedder() -> None:
    """Load the embedding model in the background, after startup.

    Loading it eagerly would delay readiness, and on Render a slow first
    health check looks like a failed deploy. Loading it lazily means the first
    real visitor pays ~3s on top of an already-slow cold start. Doing it in the
    background gets both: /healthz answers immediately, and the model is
    resident before anyone asks a question.
    """
    try:
        r = get_retriever()
        await asyncio.to_thread(r._embed, "warm up the embedding model")
        log.info("embedding model warm")
    except Exception as exc:  # noqa: BLE001
        log.warning("embedder warm-up failed (will load on demand): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm the index off the event loop; it does file + model I/O.
    try:
        await asyncio.to_thread(get_retriever)
    except Exception as exc:  # noqa: BLE001
        log.error("retriever failed to load: %s", exc)
    await client.discover()
    warm = asyncio.create_task(_warm_embedder())
    log.info(
        "%s ready on %s:%s (llm=%s)",
        settings.app_name, settings.host, settings.port, client.enabled,
    )
    yield
    warm.cancel()


app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs",
    redoc_url=None,
)

app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(BodySizeLimitMiddleware)
if settings.cors_list:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_list,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )


def sse(event: dict) -> str:
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


@app.post("/api/chat")
async def chat(req: ChatRequest, request: Request):
    """Stream one assistant turn as Server-Sent Events."""
    verdict = limiter.check(client_key(request))
    if not verdict.allowed:
        return JSONResponse(
            {"error": verdict.reason},
            status_code=429,
            headers={"Retry-After": str(verdict.retry_after)},
        )

    history = [t.model_dump() for t in req.history]

    async def gen():
        try:
            async for event in answer(req.message, history):
                yield sse(event)
        except Exception as exc:  # noqa: BLE001
            # Never leak internals to the client; the detail goes to the log.
            log.exception("chat failed: %s", exc)
            yield sse({"type": "error", "message": "Something went wrong. Please try again."})
            yield sse({"type": "done", "meta": {"mode": "error"}})

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/properties")
async def properties(
    q: str = "",
    city: str | None = None,
    source: str | None = None,
    listing_type: str | None = None,
    property_type: str | None = None,
    min_price_usd: float | None = None,
    max_price_usd: float | None = None,
    min_bedrooms: int | None = None,
    limit: int = 12,
):
    """Direct search endpoint - also what powers 'search mode'."""
    r = get_retriever()
    filters = Filters(
        city=city, source=source, listing_type=listing_type,
        property_type=property_type, min_price_usd=min_price_usd,
        max_price_usd=max_price_usd, min_bedrooms=min_bedrooms,
    )
    return {"results": r.search(q, filters, k=max(1, min(limit, 50)))}


@app.get("/api/stats")
async def stats():
    r = get_retriever()
    return {
        "corpus": r.corpus_stats(),
        "llm": client.status(),
        "limits": limiter.snapshot(),
    }


@app.get("/healthz")
async def healthz():
    """Liveness + readiness. Returns 503 if the corpus is missing."""
    try:
        r = get_retriever()
        n = r.corpus_stats()["total"]
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"status": "error", "detail": str(exc)[:200]}, status_code=503)
    if not n:
        return JSONResponse({"status": "empty corpus"}, status_code=503)
    return {
        "status": "ok",
        "properties": n,
        "llm_enabled": client.enabled,
        "embedder_warm": r._embedder is not None,
        "ai_quota_exhausted": client.account_quota_exhausted,
    }


# --- static SPA (mounted last so /api/* wins) ------------------------------
if settings.web_dist.exists():
    app.mount(
        "/assets",
        StaticFiles(directory=str(settings.web_dist / "assets")),
        name="assets",
    )

    @app.get("/{path:path}")
    async def spa(path: str):
        candidate = settings.web_dist / path
        if path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(settings.web_dist / "index.html")

else:  # pragma: no cover - dev convenience only

    @app.get("/")
    async def no_ui():
        return {
            "message": f"{settings.app_name} API is running, but the UI is not built.",
            "hint": "cd web && npm install && npm run build",
            "endpoints": ["/api/chat", "/api/properties", "/api/stats", "/healthz"],
        }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app", host=settings.host, port=settings.port, log_level="info"
    )
