"""Settings, read from the environment and an optional .env file. Every name starts with DESK_."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=ROOT / ".env", env_prefix="DESK_", extra="ignore")

    # Any server that speaks the OpenAI Chat Completions format.
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o-mini"
    # Some servers reject stream_options. Turn this off for them and token counts go missing.
    llm_stream_usage: bool = True
    # Seconds to wait for the next chunk. A model on a laptop CPU needs far more than an API.
    llm_timeout_s: float = 60.0

    tenant_dir: Path = ROOT / "data" / "tenants" / "demo_coaching"
    data_dir: Path = ROOT / "var"
    # Empty keeps everything in SQLite files under data_dir. A postgresql:// address moves the
    # desk's tables and the conversation checkpoints to PostgreSQL.
    database_url: str = ""
    prices_file: Path = ROOT / "prices.toml"

    # Empty means the embedded Qdrant under data_dir. Set a URL to use a Qdrant server.
    qdrant_url: str = ""
    qdrant_api_key: str = ""
    embedder: str = "fastembed"  # "fastembed" or "hash" (tests and offline runs)
    dense_model: str = "BAAI/bge-small-en-v1.5"
    # Model files are kept here, so they are downloaded once and survive a cleared temp folder.
    embed_cache_dir: Path = ROOT / "var" / "fastembed"

    # True: the code searches with the caller's words before every model call. False: the
    # model decides when to search. See the results in the README before changing it.
    always_search_first: bool = False
    max_model_calls_per_turn: int = 6

    # "local" keeps leads in the desk's own database. "hubspot" needs a private app token.
    crm: str = "local"
    hubspot_token: str = ""
    hubspot_client_secret: str = ""  # signs the webhooks HubSpot sends to the desk
    # The shared secret a website uses to sign leads it posts to /webhooks/lead.
    webhook_secret: str = ""
    # When set, the operator endpoint requires it in the X-Desk-Secret header.
    operator_secret: str = ""
    # An incoming-webhook URL (Slack, Discord) that gets an alert on a handoff or a hot lead.
    notify_url: str = ""
    notify_secret: str = ""
    # The address the outside world uses for this server, for example an ngrok or Cloud Run
    # URL. HubSpot and Twilio sign the full URL they called, so it must be known exactly.
    public_base_url: str = "http://localhost:8000"
    session_cost_cap_usd: float = 0.50

    # Voice. All three models run on the CPU.
    voice_stt_model: str = "base"  # a Whisper size: tiny, base, small. Bigger is slower and more accurate
    voice_language: str = ""  # "en" or "hi" pins the language. Empty lets Whisper detect it
    voice_file: Path = ROOT / "var" / "voices" / "en_US-lessac-medium.onnx"
    voice_silence_ms: int = 700  # how long a pause ends the caller's turn
    twilio_auth_token: str = ""  # checks that phone webhooks really come from Twilio

    # Where to send traces, for example http://localhost:4318/v1/traces. Empty sends none.
    otlp_endpoint: str = ""
    log_json: bool = False


@lru_cache
def get_settings() -> Settings:
    return Settings()
