import time
from types import SimpleNamespace

import bridge
from theatre import ChaosDirector


def test_cleanup_removes_old_entries(monkeypatch):
    cd = ChaosDirector()
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
    called = SimpleNamespace(flag=False, arg=None)

    async def fake_cleanup_threads():
        return None

    monkeypatch.setattr(bridge, "cleanup_threads", fake_cleanup_threads)
    monkeypatch.setattr(bridge, "cleanup_hero_cache", lambda: None)
    monkeypatch.setattr(bridge.settings, "chaos_cleanup_max_age_hours", 5)

    def fake_cleanup(max_age_hours):
        called.flag = True
        called.arg = max_age_hours

    monkeypatch.setattr(bridge.CHAOS, "cleanup", fake_cleanup)

    import asyncio
    asyncio.run(bridge.periodic_cleanup(SimpleNamespace()))

    assert called.flag
    assert called.arg == 5
