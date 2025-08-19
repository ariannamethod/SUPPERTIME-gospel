import asyncio
import pytest

from db import db_init
from theatre import reload_heroes


@pytest.fixture(scope="session", autouse=True)
def init_runtime():
    asyncio.run(db_init())
    reload_heroes()
