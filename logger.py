import logging

from config import settings


logging.basicConfig(
    level=settings.get_log_level(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

