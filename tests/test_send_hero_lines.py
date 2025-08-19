import os
import pytest
import random
import asyncio
from unittest.mock import AsyncMock, MagicMock

# Prevent network calls during import
os.environ.setdefault("ASSISTANT_ID", "test")

from monolith import send_hero_lines


def test_send_chat_action_multiple_calls(monkeypatch):
    chat = MagicMock()
    chat.id = 1
    typing_msg = MagicMock()
    typing_msg.delete = AsyncMock()
    final_msg = MagicMock()
    chat.send_message = AsyncMock(side_effect=[typing_msg, final_msg])

    context = MagicMock()
    context.bot = MagicMock()
    context.bot.send_chat_action = AsyncMock()

    monkeypatch.setattr(random, "uniform", lambda a, b: 3)

    async def fast_sleep(_):
        pass
    monkeypatch.setattr(asyncio, "sleep", fast_sleep)

    asyncio.run(send_hero_lines(chat, "*Judas*\nhello", context))

    assert context.bot.send_chat_action.await_count > 1
