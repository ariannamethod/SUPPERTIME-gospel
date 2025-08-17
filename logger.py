import logging
import os


def _get_log_level() -> int:
    """Return the configured log level.

    Reads the ``LOG_LEVEL`` environment variable and falls back to ``INFO``
    if the variable is unset or invalid.
    """

    level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level_name, logging.INFO)


logging.basicConfig(
    level=_get_log_level(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger("suppertimegospel")
