from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import logging

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
    chaos_cleanup_max_age_hours: int = int(os.getenv("CHAOS_CLEANUP_MAX_AGE_HOURS", "24"))

    def validate(self) -> None:
        """Ensure all required settings are present.

        Currently ``telegram_token`` and ``openai_api_key`` are mandatory for
        the application to run. A :class:`RuntimeError` is raised if any of
        these values are missing.
        """

        required = {
            "telegram_token": self.telegram_token,
            "openai_api_key": self.openai_api_key,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )

    def get_log_level(self) -> int:
        """Return the configured logging level.

        Reads ``LOG_LEVEL`` from the environment and falls back to ``INFO``
        when the variable is unset or invalid.
        """

        level_name = os.getenv("LOG_LEVEL", "INFO").upper()
        return getattr(logging, level_name, logging.INFO)


settings = Settings()
