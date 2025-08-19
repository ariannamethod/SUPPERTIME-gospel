import os
import asyncio
import time
import random
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

from telegram.constants import ParseMode

# Ensure environment variables to avoid network calls
os.environ.setdefault("ASSISTANT_ID", "test")
os.environ.setdefault("TELEGRAM_TOKEN", "test-token")
os.environ.setdefault("OPENAI_API_KEY", "test-key")

import bridge
from db import db_get, db_set


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


async def run_on_click(update, context, monkeypatch):
    """Run on_click and wait for spawned task to finish."""
    orig_create_task = asyncio.create_task
    holder = {}

    def capture(coro):
        task = orig_create_task(coro)
        holder["task"] = task
        return task

    monkeypatch.setattr(asyncio, "create_task", capture)
    await bridge.on_click(update, context)
    task = holder.get("task")
    if task:
        await task


def test_full_user_flow(monkeypatch):
    async def run():
        chat_id = 12345
        chat = SimpleNamespace(id=chat_id)

        # Reset DB state
        await db_set(chat_id, accepted=0, chapter=None, dialogue_n=0, last_summary="")

        # Patch network and heavy functions
        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(bridge, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "run_and_wait", AsyncMock())
        monkeypatch.setattr(bridge, "thread_last_text", lambda tid: "**Judas**: hi")
        monkeypatch.setattr(bridge, "send_hero_lines", AsyncMock())
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)

        context = SimpleNamespace()

        # /start
        update = SimpleNamespace(message=make_message(chat), effective_chat=chat)
        await bridge.start(update, context)
        update.message.reply_text.assert_awaited()

        # user presses OK
        update_ok = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ok"), effective_chat=chat)
        await bridge.on_click(update_ok, context)
        update_ok.callback_query.edit_message_text.assert_awaited()
        state = await db_get(chat_id)
        assert state["accepted"] is True

        # user selects chapter 1
        chapter_text = bridge.CHAPTERS[1]
        mock_load = bridge.load_chapter_context_all
        update_ch = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ch_1"), effective_chat=chat)
        await run_on_click(update_ch, context, monkeypatch)
        assert mock_load.awaited
        called_text = mock_load.await_args.args[0]
        assert called_text == chapter_text
        state = await db_get(chat_id)
        assert state["chapter"] == 1

        # user sends a message
        user_msg = SimpleNamespace(chat=chat, text="hello", message_id=1)
        user_msg.reply_text = AsyncMock()
        chat.send_message = AsyncMock()
        update_text = SimpleNamespace(message=user_msg, effective_chat=chat)
        await bridge.on_text(update_text, context)
        send_args = bridge.send_hero_lines.await_args_list[-1]
        assert send_args.kwargs["reply_to_message_id"] == 1
        state = await db_get(chat_id)
        assert state["dialogue_n"] == 1

        # repeated /start -> OK -> chapters
        update2 = SimpleNamespace(message=make_message(chat), effective_chat=chat)
        await bridge.start(update2, context)
        update2.message.reply_text.assert_awaited()
        update_ok2 = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ok"), effective_chat=chat)
        await bridge.on_click(update_ok2, context)
        update_ok2.callback_query.edit_message_text.assert_awaited()

    asyncio.run(run())


def test_unknown_chapter_callback(monkeypatch):
    async def run():
        chat_id = 4242
        chat = SimpleNamespace(id=chat_id, send_message=AsyncMock())

        await db_get(chat_id)  # ensure row exists
        await db_set(chat_id, accepted=1, chapter=3, dialogue_n=2, last_summary="old")
        state_before = await db_get(chat_id)

        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))

        update = SimpleNamespace(callback_query=make_callback_query(chat_id, chat, "ch_bad"), effective_chat=chat)
        context = SimpleNamespace()
        await bridge.on_click(update, context)

        chat.send_message.assert_awaited_once_with("Unknown chapter")
        state_after = await db_get(chat_id)
        assert state_after == state_before

    asyncio.run(run())


def test_single_callback_loads_chapter(monkeypatch):
    async def run():
        chat_id = 9090
        chat = SimpleNamespace(id=chat_id)
        await db_set(chat_id, accepted=1, chapter=None, dialogue_n=0, last_summary="")

        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        mock_load = AsyncMock()
        monkeypatch.setattr(bridge, "load_chapter_context_all", mock_load)
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "run_and_wait", AsyncMock())
        monkeypatch.setattr(bridge, "request_scene", AsyncMock(return_value="**Judas**: hi"))
        monkeypatch.setattr(bridge, "send_hero_lines", AsyncMock())
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)

        update = SimpleNamespace(
            callback_query=make_callback_query(chat_id, chat, "ch_1"), effective_chat=chat
        )
        context = SimpleNamespace()
        await run_on_click(update, context, monkeypatch)

        assert mock_load.await_count == 1
        state = await db_get(chat_id)
        assert state["chapter"] == 1

    asyncio.run(run())


def test_menu_shows_chapters(monkeypatch):
    chat_id = 777
    chat = SimpleNamespace(id=chat_id)
    msg = make_message(chat)
    update = SimpleNamespace(message=msg, effective_chat=chat)
    context = SimpleNamespace()

    asyncio.run(bridge.menu_cmd(update, context))
    msg.reply_text.assert_awaited()
    args, kwargs = msg.reply_text.call_args
    assert args[0] == "YOU CHOOSE:"
    assert isinstance(kwargs.get("reply_markup"), bridge.InlineKeyboardMarkup)


def test_menu_and_start_cancel_idle(monkeypatch):
    chat_id = 1010
    chat = SimpleNamespace(id=chat_id)
    context = SimpleNamespace()

    async def run():
        msg = make_message(chat)
        update_menu = SimpleNamespace(message=msg, effective_chat=chat)
        bridge.IDLE.last_activity[chat_id] = time.time()
        idle = bridge.IDLE.start(chat_id, asyncio.sleep(3600))
        await bridge.menu_cmd(update_menu, context)
        await asyncio.sleep(0)
        assert idle.cancelled()
        assert chat_id not in bridge.IDLE.idle_tasks
        assert chat_id not in bridge.IDLE.last_activity
        msg.reply_text.assert_awaited()

        msg2 = make_message(chat)
        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock())
        update_start = SimpleNamespace(message=msg2, effective_chat=chat)
        bridge.IDLE.last_activity[chat_id] = time.time()
        idle2 = bridge.IDLE.start(chat_id, asyncio.sleep(3600))
        await bridge.start(update_start, context)
        await asyncio.sleep(0)
        assert idle2.cancelled()
        assert chat_id not in bridge.IDLE.idle_tasks
        assert chat_id not in bridge.IDLE.last_activity
        msg2.reply_text.assert_awaited()

    asyncio.run(run())


def test_no_response_when_menu_called_early(monkeypatch):
    async def run():
        chat_id = 5151
        chat = SimpleNamespace(id=chat_id, send_message=AsyncMock())
        await db_get(chat_id)
        await db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary="")

        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(bridge, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)
        send_mock = AsyncMock()
        monkeypatch.setattr(bridge, "send_hero_lines", send_mock)
        monkeypatch.setattr(bridge, "request_scene", AsyncMock(return_value="**Judas**: hi"))

        started = asyncio.Event()
        finished = asyncio.Event()

        async def fake_run_and_wait(*a, **k):
            started.set()
            await finished.wait()

        monkeypatch.setattr(bridge, "run_and_wait", fake_run_and_wait)

        user_msg = SimpleNamespace(chat=chat, text="hi", message_id=1)
        user_msg.reply_text = AsyncMock()
        update_text = SimpleNamespace(message=user_msg, effective_chat=chat)
        context = SimpleNamespace()

        task = asyncio.create_task(bridge.on_text(update_text, context))
        await started.wait()
        msg_menu = make_message(chat)
        update_menu = SimpleNamespace(message=msg_menu, effective_chat=chat)
        await bridge.menu_cmd(update_menu, context)
        finished.set()
        await task
        assert send_mock.await_count == 0

    asyncio.run(run())


def test_send_hero_lines_delay_and_separate(monkeypatch):
    chat = SimpleNamespace(id=1)
    typing_msgs = [SimpleNamespace(delete=AsyncMock()), SimpleNamespace(delete=AsyncMock())]
    final_msgs = [SimpleNamespace(), SimpleNamespace()]
    chat.send_message = AsyncMock(
        side_effect=[typing_msgs[0], final_msgs[0], typing_msgs[1], final_msgs[1]]
    )
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    sleep_mock = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep_mock)
    uniform_mock = MagicMock(return_value=3)
    monkeypatch.setattr(random, "uniform", uniform_mock)

    asyncio.run(
        bridge.send_hero_lines(
            chat,
            "**Judas**: hi\n**Peter**: bye",
            context,
            participants=["Judas", "Peter"],
        )
    )

    assert chat.send_message.await_count == 4
    final_texts = [c.args[0] for c in chat.send_message.await_args_list[1::2]]
    assert final_texts == ["**Judas**\nhi", "**Peter**\nbye"]
    assert [c.args[0] for c in sleep_mock.await_args_list] == [1, 1, 1, 1, 1, 1]
    assert uniform_mock.call_count == 2


def test_idle_loop_cancelled_after_menu(monkeypatch):
    chat_id = 3030
    chat = SimpleNamespace(id=chat_id)
    context = SimpleNamespace()

    async def run():
        await db_get(chat_id)
        await db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary="")
        bridge.IDLE.last_activity[chat_id] = time.time() - bridge.INACTIVITY_TIMEOUT - 1
        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(bridge, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "run_and_wait", AsyncMock())
        monkeypatch.setattr(bridge, "thread_last_text", lambda tid: "**Judas**: hi")
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode"), silence={}))
        send_mock = AsyncMock()
        monkeypatch.setattr(bridge, "send_hero_lines", send_mock)
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)

        context.bot = SimpleNamespace(get_chat=AsyncMock(return_value=chat))

        await bridge.silence_watchdog(context)

        idle = bridge.IDLE.idle_tasks.get(chat_id)
        assert idle is not None and not idle.done()

        msg = make_message(chat)
        update_menu = SimpleNamespace(message=msg, effective_chat=chat)
        await bridge.menu_cmd(update_menu, context)
        await asyncio.sleep(0)

        assert idle.cancelled()
        assert chat_id not in bridge.IDLE.idle_tasks
        assert chat_id not in bridge.IDLE.last_activity

    asyncio.run(run())


def test_on_text_sends_pre_message(monkeypatch):
    async def run():
        chat_id = 999
        chat = SimpleNamespace(id=chat_id)

        await db_get(chat_id)
        await db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary="")

        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(bridge, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "run_and_wait", AsyncMock())
        monkeypatch.setattr(bridge, "thread_last_text", lambda tid: "**Judas**: hi")
        monkeypatch.setattr(bridge, "request_scene", AsyncMock(return_value="**Judas**: hi"))
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)

        send_mock = AsyncMock()
        monkeypatch.setattr(bridge, "send_hero_lines", send_mock)

        user_msg = SimpleNamespace(chat=chat, text="hi", message_id=5)
        user_msg.reply_text = AsyncMock()
        chat.send_message = AsyncMock()
        update = SimpleNamespace(message=user_msg, effective_chat=chat)
        context = SimpleNamespace()

        bridge.IDLE.last_activity[chat_id] = time.time() - bridge.INACTIVITY_TIMEOUT - 1

        await bridge.on_text(update, context)

        assert send_mock.await_count == 2
        first_text = send_mock.await_args_list[0].args[1]
        assert "опять ты" in first_text
        for call in send_mock.await_args_list:
            assert call.kwargs["reply_to_message_id"] == 5

    asyncio.run(run())


def test_reply_prioritizes_hero(monkeypatch):
    async def run():
        chat_id = 2021
        chat = SimpleNamespace(id=chat_id)

        await db_get(chat_id)
        await db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary="")

        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(bridge, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "run_and_wait", AsyncMock())
        monkeypatch.setattr(bridge, "thread_last_text", lambda tid: "**Judas**: ok")
        monkeypatch.setattr(bridge, "send_hero_lines", AsyncMock())
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Peter"], "mode")))
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)

        captured = {}

        def fake_build_scene_prompt(ch, ch_text, responders, user_text, summary):
            captured["responders"] = list(responders)
            return "prompt"

        monkeypatch.setattr(bridge, "build_scene_prompt", fake_build_scene_prompt)

        reply_msg = SimpleNamespace(text="**Judas**\nhello")
        user_msg = SimpleNamespace(chat=chat, text="answer", reply_to_message=reply_msg, message_id=9)
        user_msg.reply_text = AsyncMock()
        chat.send_message = AsyncMock()
        update = SimpleNamespace(message=user_msg, effective_chat=chat)
        context = SimpleNamespace()

        await bridge.on_text(update, context)

        assert captured["responders"][0] == "Judas"
        send_call = bridge.send_hero_lines.await_args
        assert send_call.kwargs["reply_to_message_id"] == 9

    asyncio.run(run())


def test_send_hero_lines_reply_to(monkeypatch):
    chat = SimpleNamespace(id=1)
    typing_msg = SimpleNamespace(delete=AsyncMock())
    final_msg = SimpleNamespace()
    chat.send_message = AsyncMock(side_effect=[typing_msg, final_msg])
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    asyncio.run(
        bridge.send_hero_lines(
            chat,
            "**Judas**: hi",
            context,
            reply_to_message_id=77,
            participants=["Judas"],
        )
    )

    calls = chat.send_message.await_args_list
    assert len(calls) == 2
    assert calls[1].kwargs["reply_to_message_id"] == 77
    typing_msg.delete.assert_awaited_once()


def test_send_hero_lines_fallback(monkeypatch):
    chat = SimpleNamespace(id=1, send_message=AsyncMock())
    context = SimpleNamespace(bot=SimpleNamespace(send_chat_action=AsyncMock()))
    monkeypatch.setattr(asyncio, "sleep", AsyncMock())

    asyncio.run(
        bridge.send_hero_lines(
            chat,
            "plain text",
            context,
            participants=["Judas"],
        )
    )

    chat.send_message.assert_awaited_once_with(
        "**Narrator**\nplain text",
        parse_mode=ParseMode.MARKDOWN,
        reply_to_message_id=None,
    )


def test_chapter_callback_send_error(monkeypatch):
    async def run():
        chat_id = 555
        chat = SimpleNamespace(id=chat_id, send_message=AsyncMock())

        await db_get(chat_id)  # ensure row exists
        await db_set(chat_id, accepted=1, chapter=None, dialogue_n=0, last_summary="")

        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(bridge, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "run_and_wait", AsyncMock())
        monkeypatch.setattr(bridge, "thread_last_text", lambda tid: "**Judas**: hi")
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)

        send_mock = AsyncMock(side_effect=RuntimeError("boom"))
        monkeypatch.setattr(bridge, "send_hero_lines", send_mock)

        q = make_callback_query(chat_id, chat, "ch_1")
        update = SimpleNamespace(callback_query=q, effective_chat=chat)
        context = SimpleNamespace()

        await run_on_click(update, context, monkeypatch)

        q.message.delete.assert_not_awaited()
        chat.send_message.assert_awaited_once_with("Failed to load chapter")

    asyncio.run(run())


def test_no_send_when_chapter_changes_during_wait(monkeypatch):
    chat_id = 424242
    chat = SimpleNamespace(id=chat_id, send_message=AsyncMock())

    async def run():
        await db_get(chat_id)
        await db_set(chat_id, accepted=1, chapter=1, dialogue_n=0, last_summary="")

        monkeypatch.setattr(bridge, "ensure_thread", AsyncMock(return_value="thread-1"))
        monkeypatch.setattr(bridge, "load_chapter_context_all", AsyncMock())
        monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
        monkeypatch.setattr(bridge, "CHAOS", SimpleNamespace(pick=lambda *a, **k: (["Judas"], "mode")))
        fake_client = SimpleNamespace(beta=SimpleNamespace(threads=SimpleNamespace(messages=SimpleNamespace(create=MagicMock()))))
        monkeypatch.setattr(bridge, "client", fake_client)

        send_mock = AsyncMock()
        monkeypatch.setattr(bridge, "send_hero_lines", send_mock)
        monkeypatch.setattr(bridge, "request_scene", AsyncMock(return_value="**Judas**: hi"))

        idle = bridge.IDLE.start(chat_id, asyncio.sleep(3600))

        async def fake_run_and_wait(thread_id, extra_instructions=None, timeout_s=120):
            await db_set(chat_id, chapter=2)

        monkeypatch.setattr(bridge, "run_and_wait", fake_run_and_wait)

        user_msg = SimpleNamespace(chat=chat, text="hi", message_id=7)
        user_msg.reply_text = AsyncMock()
        update = SimpleNamespace(message=user_msg, effective_chat=chat)
        context = SimpleNamespace()

        await bridge.on_text(update, context)
        await asyncio.sleep(0)

        assert send_mock.await_count == 0
        assert idle.cancelled()
        assert chat_id not in bridge.IDLE.idle_tasks
        assert chat_id not in bridge.IDLE.last_activity

    asyncio.run(run())


def test_idle_tracker_cleanup():
    async def run():
        t1 = bridge.IDLE.start(1, asyncio.sleep(3600))
        bridge.IDLE.last_activity[1] = time.time()
        t2 = bridge.IDLE.start(2, asyncio.sleep(3600))
        bridge.IDLE.last_activity[2] = time.time()
        await asyncio.sleep(0)
        bridge.IDLE.cleanup()
        await asyncio.sleep(0)
        assert t1.cancelled() and t2.cancelled()
        assert bridge.IDLE.idle_tasks == {}
        assert bridge.IDLE.last_activity == {}

    asyncio.run(run())
