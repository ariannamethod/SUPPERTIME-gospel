import asyncio
import sqlite3

from logger import logger
from config import settings

DB_PATH = settings.db_path
SUMMARY_EVERY = settings.summary_every


def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


async def db_init():
    def _init():
        with get_connection() as conn:
            conn.execute(
                """
        CREATE TABLE IF NOT EXISTS chats (
            chat_id     INTEGER PRIMARY KEY,
            thread_id   TEXT,
            accepted    INTEGER DEFAULT 0,
            chapter     INTEGER,
            dialogue_n  INTEGER DEFAULT 0,
            last_summary TEXT
        )"""
            )
            try:
                conn.execute("ALTER TABLE chats ADD COLUMN last_summary TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chats_thread_id ON chats(thread_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_chats_chapter ON chats(chapter)"
            )
            conn.commit()

    await asyncio.to_thread(_init)


async def db_get(chat_id):
    def _get():
        try:
            with get_connection() as conn:
                cur = conn.execute(
                    "SELECT chat_id, thread_id, accepted, chapter, dialogue_n, last_summary FROM chats WHERE chat_id=?",
                    (chat_id,),
                )
                row = cur.fetchone()
                if row:
                    return {
                        "chat_id": row["chat_id"],
                        "thread_id": row["thread_id"],
                        "accepted": bool(row["accepted"]),
                        "chapter": row["chapter"],
                        "dialogue_n": row["dialogue_n"],
                        "last_summary": row["last_summary"],
                    }
                conn.execute("INSERT OR IGNORE INTO chats(chat_id) VALUES(?)", (chat_id,))
                conn.commit()
        except sqlite3.Error as e:
            logger.exception("DB get failed for chat_id %s: %s", chat_id, e)
        return {
            "chat_id": chat_id,
            "thread_id": None,
            "accepted": False,
            "chapter": None,
            "dialogue_n": 0,
            "last_summary": "",
        }

    return await asyncio.to_thread(_get)


async def db_set(chat_id, **fields):
    keys = ", ".join([f"{k}=?" for k in fields.keys()])
    vals = list(fields.values()) + [chat_id]

    def _set():
        try:
            with get_connection() as conn:
                conn.execute(f"UPDATE chats SET {keys} WHERE chat_id=?", vals)
                conn.commit()
        except sqlite3.Error as e:
            logger.exception("DB set failed for chat_id %s: %s", chat_id, e)

    await asyncio.to_thread(_set)


asyncio.run(db_init())
