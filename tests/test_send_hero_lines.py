import os
import pytest
import random
import asyncio
from unittest.mock import AsyncMock, MagicMock
from telegram.constants import ParseMode

# Prevent network calls during import
os.environ.setdefault("ASSISTANT_ID", "test")

from bridge import send_hero_lines


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

    asyncio.run(send_hero_lines(chat, "*Judas*\nhello", context, participants=["Judas"]))

    assert context.bot.send_chat_action.await_count > 1


def test_full_unrecognized_fallback(monkeypatch, caplog):
    chat = MagicMock(id=1, send_message=AsyncMock())
    context = MagicMock(bot=MagicMock(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    text = "plain text"
    with caplog.at_level("WARNING"):
        asyncio.run(send_hero_lines(chat, text, context, participants=["Judas"]))
    chat.send_message.assert_awaited_once_with(
        f"**Narrator**\n{text}",
        parse_mode=ParseMode.MARKDOWN,
        reply_to_message_id=None,
    )
    assert "Expected 1 lines" in caplog.text


def test_partial_unrecognized_fallback(monkeypatch, caplog):
    chat = MagicMock(id=1, send_message=AsyncMock())
    context = MagicMock(bot=MagicMock(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    text = "**Judas**: hi\nno name here"
    with caplog.at_level("WARNING"):
        asyncio.run(
            send_hero_lines(
                chat,
                text,
                context,
                participants=["Judas", "Peter"],
            )
        )
    chat.send_message.assert_awaited_once_with(
        f"**Narrator**\n{text}",
        parse_mode=ParseMode.MARKDOWN,
        reply_to_message_id=None,
    )
    assert "Expected 2 lines" in caplog.text


def test_quote_line_unrecognized(monkeypatch, caplog):
    chat = MagicMock(id=1, send_message=AsyncMock())
    context = MagicMock(bot=MagicMock(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    text = '"Judas" hi'
    with caplog.at_level("WARNING"):
        asyncio.run(send_hero_lines(chat, text, context, participants=["Judas"]))
    chat.send_message.assert_awaited_once_with(
        f"**Narrator**\n{text}",
        parse_mode=ParseMode.MARKDOWN,
        reply_to_message_id=None,
    )
    assert "Expected 1 lines" in caplog.text


def test_list_marker_unrecognized(monkeypatch, caplog):
    chat = MagicMock(id=1, send_message=AsyncMock())
    context = MagicMock(bot=MagicMock(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    text = "- Judas hi"
    with caplog.at_level("WARNING"):
        asyncio.run(send_hero_lines(chat, text, context, participants=["Judas"]))
    chat.send_message.assert_awaited_once_with(
        f"**Narrator**\n{text}",
        parse_mode=ParseMode.MARKDOWN,
        reply_to_message_id=None,
    )
    assert "Expected 1 lines" in caplog.text
