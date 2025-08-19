import asyncio
import aiosqlite
import logging
import logger

from config import settings

logger = logging.getLogger(__name__)

DB_PATH = settings.db_path
SUMMARY_EVERY = settings.summary_every


async def db_init():
    async with aiosqlite.connect(DB_PATH) as conn:
        await conn.execute(
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
            await conn.execute("ALTER TABLE chats ADD COLUMN last_summary TEXT")
        except aiosqlite.OperationalError:
            pass
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chats_thread_id ON chats(thread_id)"
        )
        await conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_chats_chapter ON chats(chapter)"
        )
        await conn.commit()


async def db_get(chat_id):
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            conn.row_factory = aiosqlite.Row
            async with conn.execute(
                "SELECT chat_id, thread_id, accepted, chapter, dialogue_n, last_summary FROM chats WHERE chat_id=?",
                (chat_id,),
            ) as cur:
                row = await cur.fetchone()
            if row:
                return {
                    "chat_id": row["chat_id"],
                    "thread_id": row["thread_id"],
                    "accepted": bool(row["accepted"]),
                    "chapter": row["chapter"],
                    "dialogue_n": row["dialogue_n"],
                    "last_summary": row["last_summary"],
                }
            await conn.execute(
                "INSERT OR IGNORE INTO chats(chat_id) VALUES(?)", (chat_id,)
            )
            await conn.commit()
    except aiosqlite.Error as e:
        logger.exception("DB get failed for chat_id %s: %s", chat_id, e)
    return {
        "chat_id": chat_id,
        "thread_id": None,
        "accepted": False,
        "chapter": None,
        "dialogue_n": 0,
        "last_summary": "",
    }


async def db_set(chat_id, **fields):
    keys = ", ".join([f"{k}=?" for k in fields.keys()])
    vals = list(fields.values()) + [chat_id]
    try:
        async with aiosqlite.connect(DB_PATH) as conn:
            await conn.execute(f"UPDATE chats SET {keys} WHERE chat_id=?", vals)
            await conn.commit()
    except aiosqlite.Error as e:
        logger.exception("DB set failed for chat_id %s: %s", chat_id, e)


