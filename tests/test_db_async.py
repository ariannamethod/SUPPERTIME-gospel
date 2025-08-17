import asyncio
import pytest

import monolith


@pytest.mark.asyncio
async def test_parallel_db_access():
    async def worker(cid, n):
        await monolith.db_set(cid, accepted=1, chapter=n, dialogue_n=n, last_summary="")
        state = await monolith.db_get(cid)
        assert state["dialogue_n"] == n
        assert state["chapter"] == n
        assert state["accepted"] is True

    await asyncio.gather(*(worker(1000 + i, i) for i in range(10)))


@pytest.mark.asyncio
async def test_parallel_get_same_chat():
    chat_id = 9999
    results = await asyncio.gather(*(monolith.db_get(chat_id) for _ in range(5)))
    assert all(r["chat_id"] == chat_id for r in results)
