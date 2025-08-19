import time
from types import SimpleNamespace

import monolith


def test_cleanup_removes_old_entries(monkeypatch):
    cd = monolith.ChaosDirector()
    cd.silence["old"] = 1
    cd.last_activity["old"] = time.time() - 2 * 3600
    cd.silence["recent"] = 2
    cd.last_activity["recent"] = time.time()
    cd.cleanup(max_age_hours=1)
    assert "old" not in cd.silence
    assert "old" not in cd.last_activity
    assert "recent" in cd.silence
    assert "recent" in cd.last_activity


def test_periodic_cleanup_calls_chaos_cleanup(monkeypatch):
    called = SimpleNamespace(flag=False)

    async def fake_cleanup_threads():
        return None

    monkeypatch.setattr(monolith, "cleanup_threads", fake_cleanup_threads)
    monkeypatch.setattr(monolith, "cleanup_hero_cache", lambda: None)
    monkeypatch.setattr(monolith.CHAOS, "cleanup", lambda: setattr(called, "flag", True))

    import asyncio
    asyncio.run(monolith.periodic_cleanup(SimpleNamespace()))

    assert called.flag
