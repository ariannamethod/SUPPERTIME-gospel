import os
import pytest

# Prevent network calls during import
os.environ.setdefault("ASSISTANT_ID", "test")

from monolith import guess_participants, parse_prompt_sections, parse_lines


def test_guess_participants_header():
    chapter = "Participants: Judas, Mary, Jan"
    assert guess_participants(chapter) == ["Judas", "Mary", "Jan"]


def test_guess_participants_regex_detection():
    chapter = "Mary spoke with the Teacher while Peter listened."
    assert guess_participants(chapter) == ["Yeshua", "Peter", "Mary"]


def test_guess_participants_default():
    chapter = "No known names here."
    assert guess_participants(chapter) == [
        "Judas", "Yeshua", "Peter", "Mary", "Jan", "Thomas"
    ]


def test_parse_prompt_sections_basic():
    text = (
        "NAME: Yeshua\n"
        "VOICE: gentle\n"
        "BEHAVIOR:\n"
        "kind and patient\n"
        "INIT: Hello\n"
        "REPLY:\n"
        "first line\n"
        "second line\n"
    )
    sections = parse_prompt_sections(text)
    assert sections["NAME"] == "Yeshua"
    assert sections["VOICE"] == "gentle"
    assert sections["BEHAVIOR"] == "kind and patient"
    assert sections["INIT"] == "Hello"
    assert sections["REPLY"] == "first line\nsecond line"


def test_parse_lines_block_format():
    text = (
        "*Judas*\n"
        "betrayal whispers\n"
        "*Peter*\n"
        "steadfast response"
    )
    assert list(parse_lines(text)) == [
        ("Judas", "betrayal whispers"),
        ("Peter", "steadfast response"),
    ]


def test_main_checks_env(monkeypatch):
    import monolith

    monkeypatch.setattr(monolith, "TELEGRAM_TOKEN", None)
    monkeypatch.setattr(monolith.settings, "openai_api_key", None)
    with pytest.raises(RuntimeError):
        monolith.main()


def test_ensure_thread_uses_to_thread(monkeypatch):
    import monolith
    import asyncio

    chat_id = 99999
    monolith.db_get(chat_id)

    called = {}

    def fake_create(*args, **kwargs):
        class Obj:
            id = "th123"
        return Obj()

    async def fake_to_thread(func, *args, **kwargs):
        called["func"] = func
        return func(*args, **kwargs)

    monkeypatch.setattr(monolith.client.beta.threads, "create", fake_create)
    monkeypatch.setattr(monolith.asyncio, "to_thread", fake_to_thread)

    tid = asyncio.run(monolith.ensure_thread(chat_id))
    assert tid == "th123"
    assert called["func"] is monolith.client.beta.threads.create


def test_thread_add_message_uses_to_thread(monkeypatch):
    import monolith
    import asyncio

    recorded = {}

    def fake_create(*args, **kwargs):
        recorded["kwargs"] = kwargs

    async def fake_to_thread(func, *args, **kwargs):
        recorded["func"] = func
        return func(*args, **kwargs)

    monkeypatch.setattr(monolith.client.beta.threads.messages, "create", fake_create)
    monkeypatch.setattr(monolith.asyncio, "to_thread", fake_to_thread)

    asyncio.run(monolith.thread_add_message("tid", "user", "hi"))
    assert recorded["func"] is monolith.client.beta.threads.messages.create
    assert recorded["kwargs"] == {"thread_id": "tid", "role": "user", "content": "hi"}
