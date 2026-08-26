"""Configuration. Everything comes from the environment -- no secrets in code."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from .paths import CORPUS_PATH, DB_PATH, EMBEDDING_MODEL, ROOT


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # --- OpenRouter -------------------------------------------------------
    openrouter_api_key: str = ""
    openrouter_base_url: str = "https://openrouter.ai/api/v1"

    # Ordered preference list of FREE models that support tool calling.
    # Verified against GET /api/v1/models?supported_parameters=tools.
    # At startup we intersect this with the live catalogue, so a model being
    # retired degrades to the next one instead of breaking the app.
    # Verified by probing the live API, not just the catalogue: the
    # thinkingmachines/inkling* entries advertise :free + tools but reject
    # ordinary API calls with 403 ("only available on agentic harnesses"),
    # so they are excluded rather than wasting a slot on every request.
    model_candidates: str = (
        "dots-studio/dots-3-note-preview:free,"
        "nvidia/nemotron-3.5-lightning:free,"
        "poolside/laguna-s-2.1:free,"
        "liquid/lfm-2.5-2.6b:free"
    )
    request_timeout: float = 75.0
    max_tool_rounds: int = 3

    # --- app --------------------------------------------------------------
    app_name: str = "Gulf Property AI"
    public_url: str = "http://localhost:7860"
    host: str = "0.0.0.0"  # noqa: S104 - required for containerised deploys
    port: int = 7860
    log_level: str = "INFO"

    # --- data -------------------------------------------------------------
    db_path: Path = DB_PATH
    corpus_path: Path = CORPUS_PATH
    embedding_model: str = EMBEDDING_MODEL
    web_dist: Path = ROOT / "web" / "dist"

    # --- security ---------------------------------------------------------
    rate_limit_per_minute: int = 20
    rate_limit_per_day: int = 200
    global_daily_budget: int = 3000
    max_message_chars: int = 2000
    max_history_turns: int = 20
    cors_origins: str = ""  # comma-separated; empty = same-origin only

    @property
    def models(self) -> list[str]:
        return [m.strip() for m in self.model_candidates.split(",") if m.strip()]

    @property
    def cors_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openrouter_api_key.strip())


settings = Settings()
