import asyncio
import theatre


def test_load_chapter_context_uses_memory_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(theatre.hero_manager, "cache_dir", tmp_path)
    theatre.hero_manager.ctx_cache.clear()
    theatre.hero_manager.ctx_cache[("Tester", "hash")] = "ctx"

    hero = theatre.Hero("Tester", {"INIT": "hi"}, "raw")

    def fake_create(*a, **k):
        raise AssertionError("API should not be called on cache hit")

    monkeypatch.setattr(theatre.client.responses, "create", fake_create)

    asyncio.run(hero.load_chapter_context("md", "hash"))
    assert hero.ctx == "ctx"


def test_load_chapter_context_reads_cache_file(monkeypatch, tmp_path):
    monkeypatch.setattr(theatre.hero_manager, "cache_dir", tmp_path)
    theatre.hero_manager.ctx_cache.clear()
    cache_file = tmp_path / "Tester_hash.txt"
    cache_file.write_text("filectx", encoding="utf-8")

    hero = theatre.Hero("Tester", {"INIT": "hi"}, "raw")

    def fake_create(*a, **k):
        raise AssertionError("API should not be called when file cache exists")

    monkeypatch.setattr(theatre.client.responses, "create", fake_create)

    asyncio.run(hero.load_chapter_context("md", "hash"))
    assert hero.ctx == "filectx"
    assert theatre.hero_manager.ctx_cache[("Tester", "hash")] == "filectx"


def test_load_chapter_context_calls_api_when_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(theatre.hero_manager, "cache_dir", tmp_path)
    theatre.hero_manager.ctx_cache.clear()

    class Resp:
        output_text = "apictx"

    calls = {"n": 0}

    def fake_create(*a, **k):
        calls["n"] += 1
        return Resp()

    monkeypatch.setattr(theatre.client.responses, "create", fake_create)

    hero = theatre.Hero("Tester", {"INIT": "hi"}, "raw")
    asyncio.run(hero.load_chapter_context("md", "hash"))
    assert hero.ctx == "apictx"
    assert calls["n"] == 1
    assert (tmp_path / "Tester_hash.txt").read_text(encoding="utf-8") == "apictx"
    assert theatre.hero_manager.ctx_cache[("Tester", "hash")] == "apictx"
