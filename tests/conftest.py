import asyncio
import pytest

from db import db_init
from theatre import hero_manager


@pytest.fixture(scope="session", autouse=True)
def init_runtime():
    asyncio.run(db_init())
    hero_manager.reload()
