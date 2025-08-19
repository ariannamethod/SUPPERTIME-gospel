import asyncio
from unittest.mock import AsyncMock

import httpx

import bridge
import theatre


async def _fast_sleep(_):
    """Helper to skip actual sleeping in tests."""
    pass


def test_request_scene_handles_retry_after(monkeypatch):
    monkeypatch.setattr(bridge.asyncio, "sleep", _fast_sleep)

    calls = {"n": 0}

    def fake_last_text(thread_id):
        if calls["n"] == 0:
            calls["n"] += 1
            req = httpx.Request("GET", "https://example.com")
            resp = httpx.Response(429, headers={"retry-after": "1"}, request=req)
            raise bridge.OpenAIRetryAfter("rate", response=resp, body=None)
        return "**Judas**: hi"

    monkeypatch.setattr(bridge, "thread_last_text", fake_last_text)
    monkeypatch.setattr(bridge, "thread_add_message", lambda *a, **k: None)
    monkeypatch.setattr(bridge, "run_and_wait", AsyncMock())

    text = asyncio.run(bridge.request_scene("thread", ["Judas"]))
    assert text == "**Judas**: hi"


def test_load_chapter_context_handles_api_errors(monkeypatch, tmp_path):
    monkeypatch.setattr(theatre.asyncio, "sleep", _fast_sleep)
    monkeypatch.setattr(theatre.hero_manager, "cache_dir", tmp_path)
    theatre.hero_manager.ctx_cache.clear()

    calls = {"n": 0}

    def fake_create(*a, **k):
        calls["n"] += 1
        if calls["n"] == 1:
            req = httpx.Request("GET", "https://example.com")
            raise theatre.APIConnectionError(request=req)
        if calls["n"] == 2:
            req = httpx.Request("GET", "https://example.com")
            resp = httpx.Response(429, headers={"retry-after": "1"}, request=req)
            raise theatre.OpenAIRetryAfter("rate", response=resp, body=None)

        class Resp:
            output_text = "ctx"

        return Resp()

    monkeypatch.setattr(theatre.client.responses, "create", fake_create)

    hero = theatre.Hero("Tester", {"INIT": "hi"}, "raw")
    asyncio.run(hero.load_chapter_context("md", "hash"))
    assert hero.ctx == "ctx"

