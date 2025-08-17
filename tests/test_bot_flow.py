import os
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

# Ensure environment variables to avoid network calls
os.environ.setdefault("ASSISTANT_ID", "test")
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import monolith


def make_message(chat, text=None):
    msg = SimpleNamespace(chat=chat, text=text)
    msg.reply_text = AsyncMock()
    return msg


def make_callback_query(chat_id, chat, data):
    q = SimpleNamespace(
        data=data,
        answer=AsyncMock(),
        edit_message_text=AsyncMock(),
        message=SimpleNamespace(chat_id=chat_id, chat=chat, delete=AsyncMock()),
    )
    return q


def test_full_user_flow(monkeypatch):
    chat_id = 12345
    chat = SimpleNamespace(id=chat_id)

    # Reset DB state
    monolith.db_set(chat_id, accepted=0, chapter=None, dialogue_n=0, last_summary="")

    # Patch network and heavy functions
    monkeypatch.setattr(monolith, "ensure_thread", lambda cid: "thread-1")
    monkeypatch.setattr(monolith, "load_chapter_context_all", AsyncMock())
    monkeypatch.setattr(monolith, "thread_add_message", lambda *a, **k: None)
    monkeypatch.setattr(monolith, "run_and_wait", AsyncMock())
    monkeypatch.setattr(monolith, "thread_last_text", lambda tid: "**Judas**: hi")
    monkeypatch.setattr(monolith, "send_hero_lines", AsyncMock())
    monkeypatch.setattr(monolith, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
    monkeypatch.setattr(monolith, "MARKOV", SimpleNamespace(glitch=lambda: ""))
    fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
    monkeypatch.setattr(monolith, "client", fake_client)

    context = SimpleNamespace()

    # /start
    update = SimpleNamespace(message=make_message(chat), effective_chat=chat)
    asyncio.run(monolith.start(update, context))
    update.message.reply_text.assert_awaited()

    # user presses OK
    update_ok = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ok"), effective_chat=chat)
    asyncio.run(monolith.on_click(update_ok, context))
    update_ok.callback_query.edit_message_text.assert_awaited()
    state = monolith.db_get(chat_id)
    assert state["accepted"] is True

    # user selects chapter 1
    chapter_text = monolith.CHAPTERS[1]
    mock_load = monolith.load_chapter_context_all
    update_ch = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ch_1"), effective_chat=chat)
    asyncio.run(monolith.on_click(update_ch, context))
    assert mock_load.awaited
    called_text = mock_load.await_args.args[0]
    assert called_text == chapter_text
    state = monolith.db_get(chat_id)
    assert state["chapter"] == 1

    # user sends a message
    user_msg = SimpleNamespace(chat=chat, text="hello")
    user_msg.reply_text = AsyncMock()
    chat.send_message = AsyncMock()
    update_text = SimpleNamespace(message=user_msg, effective_chat=chat)
    asyncio.run(monolith.on_text(update_text, context))
    state = monolith.db_get(chat_id)
    assert state["dialogue_n"] == 1

    # repeated /start -> OK -> chapters
    update2 = SimpleNamespace(message=make_message(chat), effective_chat=chat)
    asyncio.run(monolith.start(update2, context))
    update2.message.reply_text.assert_awaited()
    update_ok2 = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ok"), effective_chat=chat)
    asyncio.run(monolith.on_click(update_ok2, context))
    update_ok2.callback_query.edit_message_text.assert_awaited()


def test_unknown_chapter_callback(monkeypatch):
    chat_id = 4242
    chat = SimpleNamespace(id=chat_id, send_message=AsyncMock())

    monolith.db_set(chat_id, accepted=1, chapter=3, dialogue_n=2, last_summary="old")
    state_before = monolith.db_get(chat_id)

    monkeypatch.setattr(monolith, "ensure_thread", lambda cid: "thread-1")

    update = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ch_bad"))
    context = SimpleNamespace()
    update.effective_chat = chat
    asyncio.run(monolith.on_click(update, context))

    chat.send_message.assert_awaited_once_with("Unknown chapter")
    state_after = monolith.db_get(chat_id)
    assert state_after == state_before


def test_menu_shows_chapters(monkeypatch):
    chat_id = 777
    chat = SimpleNamespace(id=chat_id)
    msg = make_message(chat)
    update = SimpleNamespace(message=msg)
    context = SimpleNamespace()

    asyncio.run(monolith.menu_cmd(update, context))
    msg.reply_text.assert_awaited()
    args, kwargs = msg.reply_text.call_args
    assert args[0].strip() == ""
    assert isinstance(kwargs.get("reply_markup"), monolith.InlineKeyboardMarkup)
