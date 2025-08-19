import asyncio
import logging
import sqlite3

import bridge
import db


def test_db_get_returns_default_on_error(monkeypatch, caplog):
    def fake_connect(*a, **k):  # pragma: no cover - stub
        raise db.aiosqlite.Error("boom")

    monkeypatch.setattr(db.aiosqlite, "connect", fake_connect)
    with caplog.at_level(logging.ERROR):
        state = asyncio.run(db.db_get(123))
    assert state == {
        "chat_id": 123,
        "thread_id": None,
        "accepted": False,
        "chapter": None,
        "dialogue_n": 0,
        "last_summary": "",
    }
    assert "DB get failed for chat_id 123" in caplog.text


def test_cleanup_threads_logs_sqlite_error(monkeypatch, caplog):
    rows = [{"chat_id": 1, "thread_id": "tid"}]

    class DummyCursor:
        def fetchall(self):
            return rows

    class DummyConn:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            pass

        def execute(self, *a, **k):
            return DummyCursor()

    import sqlite3 as sqlite3_module

    monkeypatch.setattr(sqlite3_module, "connect", lambda path: DummyConn())

    class DummyThreads:
        def delete(self, tid):
            pass

    class DummyClient:
        class Beta:
            threads = DummyThreads()

        beta = Beta()

    monkeypatch.setattr(bridge, "client", DummyClient())

    async def fake_db_set(*a, **k):  # pragma: no cover - stub
        raise sqlite3.Error("fail")

    monkeypatch.setattr(bridge, "db_set", fake_db_set)

    with caplog.at_level(logging.ERROR):
        asyncio.run(bridge.cleanup_threads())
    assert "Failed to delete thread tid" in caplog.text
