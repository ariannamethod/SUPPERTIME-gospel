import os
import asyncio
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from telegram.constants import ParseMode

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
    asyncio.run(monolith.db_set(chat_id, accepted=0, chapter=None, dialogue_n=0, last_summary=""))

    # Patch network and heavy functions
    monkeypatch.setattr(monolith, "ensure_thread", AsyncMock(return_value="thread-1"))
    monkeypatch.setattr(monolith, "load_chapter_context_all", AsyncMock())
    monkeypatch.setattr(monolith, "thread_add_message", lambda *a, **k: None)
    monkeypatch.setattr(monolith, "run_and_wait", AsyncMock())
    monkeypatch.setattr(monolith, "thread_last_text", lambda tid: "**Judas**: hi")
    monkeypatch.setattr(monolith, "send_hero_lines", AsyncMock())
    monkeypatch.setattr(monolith, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
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
    state = asyncio.run(monolith.db_get(chat_id))
    assert state["accepted"] is True

    # user selects chapter 1
    chapter_text = monolith.CHAPTERS[1]
    mock_load = monolith.load_chapter_context_all
    update_ch = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ch_1"), effective_chat=chat)
    asyncio.run(monolith.on_click(update_ch, context))
    assert mock_load.awaited
    called_text = mock_load.await_args.args[0]
    assert called_text == chapter_text
    state = asyncio.run(monolith.db_get(chat_id))
    assert state["chapter"] == 1

    # user sends a message
    user_msg = SimpleNamespace(chat=chat, text="hello", message_id=1)
    user_msg.reply_text = AsyncMock()
    chat.send_message = AsyncMock()
    update_text = SimpleNamespace(message=user_msg, effective_chat=chat)
    asyncio.run(monolith.on_text(update_text, context))
    send_args = monolith.send_hero_lines.await_args_list[-1]
    assert send_args.kwargs["reply_to_message_id"] == 1
    state = asyncio.run(monolith.db_get(chat_id))
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

    asyncio.run(monolith.db_get(chat_id))  # ensure row exists
    asyncio.run(monolith.db_set(chat_id, accepted=1, chapter=3, dialogue_n=2, last_summary="old"))
    state_before = asyncio.run(monolith.db_get(chat_id))

    monkeypatch.setattr(monolith, "ensure_thread", AsyncMock(return_value="thread-1"))

    update = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ch_bad"), effective_chat=chat)
    context = SimpleNamespace()
    asyncio.run(monolith.on_click(update, context))

    chat.send_message.assert_awaited_once_with("Unknown chapter")
    state_after = asyncio.run(monolith.db_get(chat_id))
    assert state_after == state_before


def test_menu_shows_chapters(monkeypatch):
    chat_id = 777
    chat = SimpleNamespace(id=chat_id)
    msg = make_message(chat)
    update = SimpleNamespace(message=msg, effective_chat=chat)
    context = SimpleNamespace()

    asyncio.run(monolith.menu_cmd(update, context))
    msg.reply_text.assert_awaited()
    args, kwargs = msg.reply_text.call_args
    assert args[0] == "YOU CHOOSE:"
    assert isinstance(kwargs.get("reply_markup"), monolith.InlineKeyboardMarkup)


def test_menu_and_start_cancel_idle(monkeypatch):
    chat_id = 1010
    chat = SimpleNamespace(id=chat_id)
    context = SimpleNamespace()

    async def run():
        msg = make_message(chat)
        update_menu = SimpleNamespace(message=msg, effective_chat=chat)
        idle = asyncio.create_task(asyncio.sleep(3600))
        monolith.IDLE_TASKS[chat_id] = idle
        await monolith.menu_cmd(update_menu, context)
        await asyncio.sleep(0)
        assert idle.cancelled()
        assert chat_id not in monolith.IDLE_TASKS
        msg.reply_text.assert_awaited()

        msg2 = make_message(chat)
        monkeypatch.setattr(monolith, "ensure_thread", AsyncMock())
        update_start = SimpleNamespace(message=msg2, effective_chat=chat)
        idle2 = asyncio.create_task(asyncio.sleep(3600))
        monolith.IDLE_TASKS[chat_id] = idle2
        await monolith.start(update_start, context)
        await asyncio.sleep(0)
        assert idle2.cancelled()
        assert chat_id not in monolith.IDLE_TASKS
        msg2.reply_text.assert_awaited()

    asyncio.run(run())


def test_idle_loop_cancelled_after_menu(monkeypatch):
    chat_id = 3030
    chat = SimpleNamespace(id=chat_id)
    context = SimpleNamespace()

    async def run():
        await monolith.db_get(chat_id)
        await monolith.db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary="")
        monolith.LAST_ACTIVITY[chat_id] = time.time() - monolith.INACTIVITY_TIMEOUT - 1
        monkeypatch.setattr(monolith, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(monolith, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(monolith, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(monolith, "run_and_wait", AsyncMock())
        monkeypatch.setattr(monolith, "thread_last_text", lambda tid: "**Judas**: hi")
        monkeypatch.setattr(monolith, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode"), silence={}))
        send_mock = AsyncMock()
        monkeypatch.setattr(monolith, "send_hero_lines", send_mock)
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(monolith, "client", fake_client)

        context.bot = SimpleNamespace(get_chat=AsyncMock(return_value=chat))

        await monolith.silence_watchdog(context)

        idle = monolith.IDLE_TASKS.get(chat_id)
        assert idle is not None and not idle.done()

        msg = make_message(chat)
        update_menu = SimpleNamespace(message=msg, effective_chat=chat)
        await monolith.menu_cmd(update_menu, context)
        await asyncio.sleep(0)

        assert idle.cancelled()
        assert chat_id not in monolith.IDLE_TASKS

    asyncio.run(run())


def test_on_text_sends_pre_message(monkeypatch):
    chat_id = 999
    chat = SimpleNamespace(id=chat_id)

    asyncio.run(monolith.db_get(chat_id))
    asyncio.run(monolith.db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary=""))

    monkeypatch.setattr(monolith, "ensure_thread", AsyncMock(return_value="thread-1"))
    monkeypatch.setattr(monolith, "load_chapter_context_all", AsyncMock())
    monkeypatch.setattr(monolith, "thread_add_message", lambda *a, **k: None)
    monkeypatch.setattr(monolith, "run_and_wait", AsyncMock())
    monkeypatch.setattr(monolith, "thread_last_text", lambda tid: "**Judas**: hi")
    monkeypatch.setattr(monolith, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
    fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
    monkeypatch.setattr(monolith, "client", fake_client)

    send_mock = AsyncMock()
    monkeypatch.setattr(monolith, "send_hero_lines", send_mock)

    user_msg = SimpleNamespace(chat=chat, text="hi", message_id=5)
    user_msg.reply_text = AsyncMock()
    chat.send_message = AsyncMock()
    update = SimpleNamespace(message=user_msg, effective_chat=chat)
    context = SimpleNamespace()

    monolith.LAST_ACTIVITY[chat_id] = time.time() - monolith.INACTIVITY_TIMEOUT - 1

    asyncio.run(monolith.on_text(update, context))

    assert send_mock.await_count == 2
    first_text = send_mock.await_args_list[0].args[1]
    assert "опять ты" in first_text
    for call in send_mock.await_args_list:
        assert call.kwargs["reply_to_message_id"] == 5


def test_reply_prioritizes_hero(monkeypatch):
    chat_id = 2021
    chat = SimpleNamespace(id=chat_id)

    asyncio.run(monolith.db_get(chat_id))
    asyncio.run(monolith.db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary=""))

    monkeypatch.setattr(monolith, "ensure_thread", AsyncMock(return_value="thread-1"))
    monkeypatch.setattr(monolith, "load_chapter_context_all", AsyncMock())
    monkeypatch.setattr(monolith, "thread_add_message", lambda *a, **k: None)
    monkeypatch.setattr(monolith, "run_and_wait", AsyncMock())
    monkeypatch.setattr(monolith, "thread_last_text", lambda tid: "**Judas**: ok")
    monkeypatch.setattr(monolith, "send_hero_lines", AsyncMock())
    monkeypatch.setattr(monolith, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Peter"], "mode")))
    fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
    monkeypatch.setattr(monolith, "client", fake_client)

    captured = {}

    def fake_build_scene_prompt(ch, ch_text, responders, user_text, summary):
        captured["responders"] = list(responders)
        return "prompt"

    monkeypatch.setattr(monolith, "build_scene_prompt", fake_build_scene_prompt)

    reply_msg = SimpleNamespace(text="**Judas**\nhello")
    user_msg = SimpleNamespace(chat=chat, text="answer", reply_to_message=reply_msg, message_id=9)
    user_msg.reply_text = AsyncMock()
    chat.send_message = AsyncMock()
    update = SimpleNamespace(message=user_msg, effective_chat=chat)
    context = SimpleNamespace()

    asyncio.run(monolith.on_text(update, context))

    assert captured["responders"][0] == "Judas"
    send_call = monolith.send_hero_lines.await_args
    assert send_call.kwargs["reply_to_message_id"] == 9


def test_send_hero_lines_reply_to(monkeypatch):
    chat = SimpleNamespace(id=1)
    typing_msg = SimpleNamespace(delete=AsyncMock())
    final_msg = SimpleNamespace()
    chat.send_message = AsyncMock(side_effect=[typing_msg, final_msg])
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    asyncio.run(monolith.send_hero_lines(chat, "**Judas**: hi", context, reply_to_message_id=77))

    calls = chat.send_message.await_args_list
    assert len(calls) == 2
    assert calls[1].kwargs["reply_to_message_id"] == 77


def test_send_hero_lines_fallback(monkeypatch):
    chat = SimpleNamespace(id=1, send_message=AsyncMock())
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    asyncio.run(monolith.send_hero_lines(chat, "plain text", context))

    chat.send_message.assert_awaited_once_with(
        "plain text", parse_mode=ParseMode.MARKDOWN, reply_to_message_id=None
    )


def test_chapter_callback_send_error(monkeypatch):
    chat_id = 555
    chat = SimpleNamespace(id=chat_id, send_message=AsyncMock())

    asyncio.run(monolith.db_set(chat_id, accepted=1, chapter=None, dialogue_n=0, last_summary=""))

    monkeypatch.setattr(monolith, "ensure_thread", AsyncMock(return_value="thread-1"))
    monkeypatch.setattr(monolith, "load_chapter_context_all", AsyncMock())
    monkeypatch.setattr(monolith, "thread_add_message", lambda *a, **k: None)
    monkeypatch.setattr(monolith, "run_and_wait", AsyncMock())
    monkeypatch.setattr(monolith, "thread_last_text", lambda tid: "**Judas**: hi")
    monkeypatch.setattr(monolith, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
    fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
    monkeypatch.setattr(monolith, "client", fake_client)

    send_mock = AsyncMock(side_effect=RuntimeError("boom"))
    monkeypatch.setattr(monolith, "send_hero_lines", send_mock)

    q = make_callback_query(chat_id, chat, "ch_1")
    update = SimpleNamespace(callback_query=q, effective_chat=chat)
    context = SimpleNamespace()

    asyncio.run(monolith.on_click(update, context))

    q.message.delete.assert_not_awaited()
    chat.send_message.assert_awaited_once_with("Failed to load chapter")
