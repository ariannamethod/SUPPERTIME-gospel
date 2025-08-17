from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os

from dotenv import load_dotenv

# Load variables from a .env file if present
load_dotenv()


@dataclass
class Settings:
    """Application configuration loaded from environment variables."""

    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4.1")
    openai_temperature: float = float(os.getenv("OPENAI_TEMPERATURE", "1.2"))
    telegram_token: str | None = os.getenv("TELEGRAM_TOKEN")
    openai_api_key: str | None = os.getenv("OPENAI_API_KEY")
    db_path: str = os.getenv("ST_DB", "supertime.db")
    summary_every: int = int(os.getenv("SUMMARY_EVERY", "20"))
    assistant_id: str | None = os.getenv("ASSISTANT_ID")
    hero_ctx_cache_dir: Path = Path(os.getenv("HERO_CTX_CACHE_DIR", ".hero_ctx_cache"))


settings = Settings()
