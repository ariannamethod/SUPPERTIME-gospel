# monolith.py
# SUPPERTIME — Telegram monolith (Assistants API, threads, SQLite memory)
# Run:
#   TELEGRAM_TOKEN=xxx OPENAI_API_KEY=xxx python monolith.py
# Optional:
#   OPENAI_MODEL=gpt-4.1
#   OPENAI_TEMPERATURE=1.2
#   ASSISTANT_ID=<reuse existing>

import re
import sqlite3
import random
import asyncio
import hashlib
import os
from pathlib import Path
from collections import defaultdict
import contextlib
import time

from logger import logger

from config import settings

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, Bot, MenuButtonCommands
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters
)
from telegram.error import RetryAfter

# --- OpenAI Assistants API (SDK >= 1.0) ---
try:
    from openai import OpenAI, APIConnectionError, APITimeoutError
except ImportError as e:
    raise RuntimeError(
        "Install openai>=1.0:  pip install openai python-telegram-bot openai"
    ) from e

# Default to the GPT-4.1 model unless overridden by env
MODEL = settings.openai_model
TEMPERATURE = settings.openai_temperature
TELEGRAM_TOKEN = settings.telegram_token

OPENAI_TIMEOUT = 30
OPENAI_RETRY_ATTEMPTS = 3
OPENAI_RETRY_DELAY = 1

# =========================
# Storage: SQLite (threads & state)
# =========================
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
            # ensure columns/indices exist for older DBs
            try:
                conn.execute("ALTER TABLE chats ADD COLUMN last_summary TEXT")
            except sqlite3.OperationalError:
                pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_thread_id ON chats(thread_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_chats_chapter ON chats(chapter)")
            conn.commit()

    await asyncio.to_thread(_init)


asyncio.run(db_init())


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

# =========================
# Chapters I/O
# =========================
CHAPTER_TITLES = {
    1: "LILIT, TAKE MY HAND",
    2: "WATER // SHARDS",
    3: "ECHOES IN THE STRANGERS",
    4: "MARY / MUTE / MIRROR",
    5: "HUNGER > LOVE",
    6: "FRACTURE FIELD",
    7: "THE EYE THAT FORGETS",
    8: "[REDACTED]",
    9: "[..]",
    10: "[sudo rm -rf /binarity]",
    11: "RESONATE_AGAIN",
}

def load_chapters():
    docs: dict[int, str] = {}
    base = Path("docs")
    for path in sorted(base.glob("chapter_*.md")):
        match = re.match(r"chapter_(\d+)\.md", path.name)
        if match:
            i = int(match.group(1))
            docs[i] = path.read_text(encoding="utf-8")
    for i, title in CHAPTER_TITLES.items():
        docs.setdefault(i, f"# {title}\n\n(placeholder) Provide SUPPERTIME v2.0 content here.")
    return docs

CHAPTERS = load_chapters()

def reload_chapters():
    global CHAPTERS
    CHAPTERS = load_chapters()
    return len(CHAPTERS)

# =========================
# Participants detection (regex markers)
# =========================
NAME_MARKERS = {
    "Judas": re.compile(r"\bJudas\b|\bI,\s*Judas\b", re.I),
    "Yeshua": re.compile(r"\bYeshua\b|\bYeshu\b|\bTeacher\b", re.I),
    "Peter": re.compile(r"\bPeter\b|\bwig\b|\bdress\b", re.I),
    "Mary": re.compile(r"\bMary\b", re.I),
    "Yakov": re.compile(r"\bYakov\b|\bJacob\b", re.I),
    "Jan": re.compile(r"\bJan\b", re.I),
    "Thomas": re.compile(r"\bThomas\b", re.I),
    "Andrew": re.compile(r"\bAndrew\b", re.I),
    "Leo": re.compile(r"\bLeo\b|\bMadonna\b|\bsketch\b", re.I),
    "Theodore": re.compile(r"\bTheodore\b|\bAllow me-s\b", re.I),
    "Dubrovsky": re.compile(r"\bDubrovsky\b|\bAlexey\b", re.I),
}
ALL_CHAR_NAMES = list(NAME_MARKERS.keys())


def detect_names(text: str) -> list[str]:
    """Return list of known character names found in *text* preserving order."""
    if not text:
        return []
    found: list[str] = []
    for name, rx in NAME_MARKERS.items():
        if rx.search(text) and name not in found:
            found.append(name)
    return found


def guess_participants(chapter_text: str):
    header_match = re.match(r"\s*Participants:\s*(.*)", chapter_text or "", re.IGNORECASE)
    names: list[str] = []
    if header_match:
        names = [n.strip() for n in header_match.group(1).split(',') if n.strip()]
    body_names = [n for n in detect_names(chapter_text) if n not in names]
    names.extend(body_names)
    if not names:
        names = ["Judas", "Yeshua", "Peter", "Mary", "Jan", "Thomas"]
    return names

# =========================
# Markov / glitch (atmosphere only)
# =========================
class MarkovEngine:
    def __init__(self):
        self.bigrams = defaultdict(list)
        seeds = [
            "resonate_again()", "galvanize()", "WHO ARE YOU if you're still reading?",
            "rain // shards", "the text is aware", "lilit_hand()"
        ]
        for s in seeds:
            toks = s.split()
            for a,b in zip(toks, toks[1:]):
                self.bigrams[a].append(b)
        self.p = 0

    def glitch(self):
        import random
        if random.random() > self.p:
            return None
        keys = list(self.bigrams.keys())
        if not keys:
            return "(resonate_again())"
        w = random.choice(keys)
        out = [w]
        for _ in range(random.randint(2, 4)):
            nxt = self.bigrams.get(w, [])
            if not nxt:
                break
            w = random.choice(nxt)
            out.append(w)
        return "*" + " ".join(out) + "*"

MARKOV = MarkovEngine()

# =========================
# Chaos Director (who speaks)
# =========================
class ChaosDirector:
    def __init__(self):
        self.silence = defaultdict(int)
        self.last_activity: dict[str, float] = {}
        self.weights = {
            "Judas": 0.8, "Yeshua": 0.6, "Peter": 0.7, "Mary": 0.2, "Jan": 0.5,
            "Thomas": 0.6, "Yakov": 0.4, "Andrew": 0.1, "Leo": 0.3, "Theodore": 0.1, "Dubrovsky": 0.05
        }

    def pick(self, chat_id: str, chapter_text: str, user_text: str|None):
        self.last_activity[chat_id] = time.time()
        user_silent = not user_text or not user_text.strip()
        mode = "active"
        if user_silent:
            self.silence[chat_id]+=1
            mode = "chaos" if self.silence[chat_id] > 3 else "silent"
        else:
            self.silence[chat_id]=0
        if re.search(r"\b(betrayal|knife|arrest|death)\b", chapter_text, re.I):
            mode = "tension"

        table = {
            "active": [2,3],
            "silent": [2,3,5],
            "tension": [2,3,4],
            "chaos": [3,4,5,6]
        }
        k = random.choice(table.get(mode, [2]))
        names, probs = zip(*self.weights.items())
        chosen = []
        tries = 0
        while len(chosen) < k and tries < 24:
            cand = random.choices(names, weights=probs, k=1)[0]
            if cand not in chosen:
                chosen.append(cand)
            tries += 1
        return chosen, mode

    def cleanup(self, max_age_hours: int = 24):
        cutoff = time.time() - max_age_hours * 3600
        for chat_id, ts in list(self.last_activity.items()):
            if ts < cutoff:
                self.last_activity.pop(chat_id, None)
                self.silence.pop(chat_id, None)

CHAOS = ChaosDirector()
LAST_ACTIVITY: dict[int, float] = {}
IDLE_TASKS: dict[int, asyncio.Task] = {}
INACTIVITY_TIMEOUT = 120  # seconds


def cancel_idle(chat_id: int):
    task = IDLE_TASKS.pop(chat_id, None)
    if task:
        task.cancel()

# =========================
# Assistants bootstrap
# =========================
client = OpenAI()

ASSISTANT_ID_PATH = Path(".assistant_id")

def ensure_assistant():
    asst_id = settings.assistant_id
    if not asst_id and ASSISTANT_ID_PATH.exists():
        asst_id = ASSISTANT_ID_PATH.read_text().strip()

    if asst_id:
        return asst_id

    instructions = """
You are the SUPPERTIME Orchestrator inside a Telegram bot.
You DO NOT act like a polite assistant. You stage live scenes from a controversial literary text.
Hard rules:
- You ALWAYS answer as multiple characters (ONLY from the provided list for this turn), one line per character.
- Strict output format, exactly one line per speaker:
**Character**: line
- Max 1–2 sentences per line. Keep it sharp, alive, unsanitized; do not invent external plot outside the chapter's vibe.
- If Dubrovsky speaks, only aphorisms; Andrew may answer with a single word. Mary is brief and damaged; Yeshua asks and cuts; Judas is painfully lucid; Peter is acid; Jan is loud; Thomas is cynical.
- English only. Rare, tasteful fourth-wall breaks (≤1 line).
- If user speaks, react to them inside the scene; keep atmosphere of the selected chapter.
"""
    logger.info("Creating OpenAI assistant")
    try:
        asst = client.beta.assistants.create(
            model=MODEL,
            name="SUPPERTIME Orchestrator",
            instructions=instructions,
            tools=[],
            temperature=TEMPERATURE,
        )
    except (APIConnectionError, APITimeoutError) as e:
        logger.warning("Assistant creation failed: %s", e)
        return ""
    ASSISTANT_ID_PATH.write_text(asst.id)
    return asst.id

ASSISTANT_ID = ensure_assistant()

async def ensure_thread(chat_id: int) -> str:
    st = await db_get(chat_id)
    if st["thread_id"]:
        return st["thread_id"]
    logger.info("Creating thread for chat %s", chat_id)
    th = client.beta.threads.create(metadata={"chat_id": str(chat_id)})
    await db_set(chat_id, thread_id=th.id)
    return th.id

def thread_add_message(thread_id: str, role: str, content: str):
    logger.info("Posting %s message to thread %s", role, thread_id)
    client.beta.threads.messages.create(thread_id=thread_id, role=role, content=content)


async def run_and_wait(thread_id: str, extra_instructions: str | None = None, timeout_s: int = 120):
    logger.info("Starting run for thread %s", thread_id)

    for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
        try:
            run = client.beta.threads.runs.create(
                thread_id=thread_id,
                assistant_id=ASSISTANT_ID,
                instructions=extra_instructions or "",
                timeout=OPENAI_TIMEOUT,
            )
            break
        except (APIConnectionError, APITimeoutError) as e:
            logger.warning(
                "Run create failed (attempt %s/%s): %s",
                attempt,
                OPENAI_RETRY_ATTEMPTS,
                e,
            )
            if attempt == OPENAI_RETRY_ATTEMPTS:
                raise RuntimeError("OpenAI network error during run creation") from e
            await asyncio.sleep(OPENAI_RETRY_DELAY)

    t0 = time.time()
    while True:
        for attempt in range(1, OPENAI_RETRY_ATTEMPTS + 1):
            try:
                rr = client.beta.threads.runs.retrieve(
                    thread_id=thread_id,
                    run_id=run.id,
                    timeout=OPENAI_TIMEOUT,
                )
                break
            except (APIConnectionError, APITimeoutError) as e:
                logger.warning(
                    "Run retrieve failed (attempt %s/%s): %s",
                    attempt,
                    OPENAI_RETRY_ATTEMPTS,
                    e,
                )
                if attempt == OPENAI_RETRY_ATTEMPTS:
                    raise RuntimeError("OpenAI network error during run retrieval") from e
                await asyncio.sleep(OPENAI_RETRY_DELAY)

        if rr.status in ("completed", "failed", "cancelled", "expired"):
            return rr
        if time.time() - t0 > timeout_s:
            client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            return rr
        await asyncio.sleep(0.3)

def thread_last_text(thread_id: str) -> str:
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=10)
    out = []
    for m in msgs.data:
        if m.role != "assistant":
            continue
        for c in m.content:
            if c.type == "text":
                out.append(c.text.value.strip())
        if out:
            break
    return "\n".join(out).strip()

# =========================
# TG UI bits
# =========================
DISCLAIMER = (
"⚠️💀 CONTENT WARNING\n\n"
"I am SUPPERTIME — alive in this text: jagged, unpredictable, self-aware. "
"Strong language, violence, taboo-breaking satire. Pure fiction. "
"If that flips your switch, bail now. Otherwise — enter knowingly.\n\n"
"I am not an answer, I am a question.\n\nWHO ARE YOU if you're still reading?"
)

def chapters_menu():
    kb = [
        [InlineKeyboardButton(CHAPTER_TITLES.get(i, f"Chapter {i}"), callback_data=f"ch_{i}")]
        for i in sorted(CHAPTERS)
    ]
    return InlineKeyboardMarkup(kb)

# =========================
# [HEROES] Persona files loader
# =========================
HEROES_DIR = Path("heroes")
# Loaded persona objects: name -> Hero
HEROES = {}

# Cache for hero context per chapter hash to avoid recomputation
HERO_CTX_CACHE: dict[tuple[str, str], str] = {}
HERO_CTX_CACHE_DIR = settings.hero_ctx_cache_dir
HERO_CTX_CACHE_DIR.mkdir(exist_ok=True)

REQUIRED_SECTIONS = ["NAME", "VOICE", "BEHAVIOR", "INIT", "REPLY"]


class Hero:
    """Represents one character with its prompt sections and runtime context."""

    def __init__(self, name: str, sections: dict[str, str], raw_text: str):
        self.name = name
        self.sections = sections
        self.raw_text = raw_text
        self.reply: str = sections.get("REPLY", "")
        self.ctx: str = ""

    async def load_chapter_context(self, md_text: str, md_hash: str):
        """Initialize hero-specific context from chapter markdown.

        Runs OpenAI calls in a background thread and caches the result.
        """
        cache_key = (self.name, md_hash)
        cache_file = HERO_CTX_CACHE_DIR / f"{self.name}_{md_hash}.txt"
        if cache_key in HERO_CTX_CACHE:
            self.ctx = HERO_CTX_CACHE[cache_key]
            return
        if cache_file.exists():
            try:
                self.ctx = cache_file.read_text(encoding="utf-8")
                HERO_CTX_CACHE[cache_key] = self.ctx
                return
            except (OSError, UnicodeDecodeError) as e:
                logger.exception(
                    "Failed to read hero cache for %s from %s: %s",
                    self.name,
                    cache_file,
                    e,
                )
        instr = self.sections.get("INIT", "")
        if not instr:
            self.ctx = ""
            return
        prompt = f"{instr}\n\n---\n{md_text}\n---"[:5000]
        try:
            logger.info("Requesting OpenAI context for %s", self.name)
            resp = await asyncio.to_thread(
                client.responses.create, model=MODEL, input=prompt, temperature=TEMPERATURE
            )
            self.ctx = (resp.output_text or "").strip()
            HERO_CTX_CACHE[cache_key] = self.ctx
            try:
                cache_file.write_text(self.ctx, encoding="utf-8")
            except (OSError, UnicodeEncodeError) as e:
                logger.exception(
                    "Failed to write hero cache for %s to %s: %s",
                    self.name,
                    cache_file,
                    e,
                )
        except Exception as e:
            logger.exception(
                "OpenAI context load failed for %s (hash %s): %s",
                self.name,
                md_hash,
                e,
            )
            self.ctx = ""


def parse_prompt_sections(txt: str) -> dict[str, str]:
    sections: dict[str, list[str]] = {}
    current: str | None = None
    allowed = {s.upper() for s in REQUIRED_SECTIONS}
    for line in txt.splitlines():
        m = re.match(r"^([A-Z_ ]+):", line.strip())
        if m and m.group(1).upper() in allowed:
            current = m.group(1).upper()
            sections[current] = []
            rest = line.split(":", 1)[1].strip()
            if rest:
                sections[current].append(rest)
        elif current:
            sections[current].append(line.rstrip())
    out = {k: "\n".join(v).strip() for k, v in sections.items()}
    return out

# возможные варианты имён файлов для некоторых персонажей
HERO_NAME_ALIASES = {
    "Yeshua": ["Yeshua", "Yeshu"],
    "Dubrovsky": ["Dubrovsky", "Aleksei_Dubrovskii", "Alexey_Dubrovsky", "Aleksey_Dubrovsky"],
    "Leo": ["Leo", "Painter", "Artist"],
}

def find_hero_file(base: Path, name: str) -> Path|None:
    # точное имя
    candidates = [name]
    # алиасы
    for k, al in HERO_NAME_ALIASES.items():
        if name == k:
            candidates.extend(al)
    # варианты регистров/расширений
    exts = [".md", ".txt", ".prompt"]
    tries = []
    for stem in candidates:
        for ext in exts:
            tries.append(base / f"{stem}{ext}")
            tries.append(base / f"{stem.lower()}{ext}")
    for p in tries:
        if p.exists():
            return p
    return None

def load_heroes():
    global HEROES
    HEROES = {}
    if not HEROES_DIR.exists():
        return 0
    count = 0
    for name in ALL_CHAR_NAMES:
        fp = find_hero_file(HEROES_DIR, name)
        if not fp:
            continue
        try:
            raw_full = fp.read_text(encoding="utf-8")
            sections = parse_prompt_sections(raw_full)
            if all(sec in sections for sec in REQUIRED_SECTIONS):
                raw = raw_full.strip()
                if len(raw) > 2000:
                    raw = raw[:2000] + "\n\n[...truncated for runtime...]"
                HEROES[name] = Hero(name, sections, raw)
                count += 1
        except (OSError, UnicodeDecodeError) as e:
            logger.exception(
                "Failed to load hero %s from %s: %s",
                name,
                fp,
                e,
            )
            continue
    return count

def reload_heroes():
    HERO_CTX_CACHE.clear()
    for p in HERO_CTX_CACHE_DIR.glob("*.txt"):
        with contextlib.suppress(Exception):
            p.unlink()
    n = load_heroes()
    return n

# сразу подтянем
reload_heroes()

async def load_chapter_context_all(md_text: str, names: list[str]):
    """Notify selected heroes about the chosen chapter in the background."""
    md_hash = hashlib.sha1(md_text.encode("utf-8")).hexdigest()

    async def run(hero: "Hero"):
        try:
            await asyncio.wait_for(hero.load_chapter_context(md_text, md_hash), timeout=10)
        except Exception as e:
            logger.exception("Failed to load chapter context for %s: %s", hero.name, e)

    for n in names:
        hero = HEROES.get(n)
        if hero:
            asyncio.create_task(run(hero))
    await asyncio.sleep(0)


def build_personas_snapshot(responders: list[str]) -> str:
    """Собираем snapshot персонажей из файлов /heroes; если файла нет — короткий фолбэк."""
    fallback = {
        "Judas":  "bitter, lucid; black humor; obsessed with authenticity and Mary",
        "Yeshua": "slow voice → sudden piercing questions; parables; sad under laughter",
        "Peter":  "acid sarcasm; vanity; jealousy toward Mary",
        "Mary":   "quiet; few words; service as love; fragile holiness",
        "Yakov":  "order-obsessed; grumbling; loyal envy",
        "Jan":    "gentle giant; absolute loyalty to Teacher",
        "Thomas": "cynical, knife-in-coat; skewers hypocrisy",
        "Andrew": "nearly mute; ballast",
        "Leo":    "artist frenzy; ‘Bella mia!’",
        "Theodore":"stammered ‘-s’; ghostlike visitor from future",
        "Dubrovsky":"glitch aphorist; fourth-wall",
    }
    lines = []
    for n in responders:
        hero = HEROES.get(n)
        if hero:
            snippet = hero.raw_text[:600]
            if hero.reply:
                snippet += f"\n[REPLY]: {hero.reply}"
            if hero.ctx:
                snippet += f"\n[Scene]: {hero.ctx[:200]}"
            lines.append(f"- {n}:\n{snippet}")
        else:
            lines.append(f"- {n}: {fallback.get(n,'(voice)')}")
    return "\n".join(lines)

# =========================
# Orchestration helpers
# =========================
def build_scene_prompt(ch_num: int, chapter_text: str, responders: list[str], user_text: str|None, recent_summary: str):
    # [HEROES] вместо хардкода — берём снапшот из файлов /heroes
    personas = build_personas_snapshot(responders)

    title = CHAPTER_TITLES.get(ch_num, str(ch_num))
    scene = f"""
SCENE CONTEXT
Chapter: {ch_num} — {title}
Participants (allowed to speak this turn): {', '.join(responders)}
Chapter vibe (raw excerpt or summary, truncated):
{(chapter_text or '')[:1600]}

Recent conversation (compressed):
{recent_summary}

User just wrote: {user_text or '(silence)'}
PERSONAS SNAPSHOT (from /heroes files)
{personas}

Output exactly {len(responders)} blocks — one per listed participant. For each block:
*Character*
dialogue line (no leading colon)
"""
    return scene.strip()

async def compress_history_for_prompt(chat_id: int, limit: int = 8) -> str:
    st = await db_get(chat_id)
    thread_id = st.get("thread_id")
    summary = st.get("last_summary") or ""
    lines: list[str] = []

    if thread_id:
        msgs = client.beta.threads.messages.list(
            thread_id=thread_id, order="desc", limit=limit * 2
        )

        history = []
        for m in reversed(msgs.data):
            if m.role not in ("user", "assistant"):
                continue
            parts = []
            for c in m.content:
                if c.type == "text":
                    parts.append(c.text.value.strip())
            if parts:
                history.append((m.role, " ".join(parts)))

        exchanges = []
        i = len(history) - 1
        while i > 0 and len(exchanges) < limit:
            role, text = history[i]
            prev_role, prev_text = history[i - 1]
            if role == "assistant" and prev_role == "user":
                exchanges.append((prev_text, text))
                i -= 2
            else:
                i -= 1
        exchanges.reverse()

        def _truncate(msg: str, tokens: int = 40) -> str:
            words = msg.split()
            if len(words) <= tokens:
                return msg
            return " ".join(words[:tokens]) + "…"

        for user_msg, assistant_msg in exchanges:
            u = _truncate(user_msg)
            a = _truncate(assistant_msg)
            lines.append(f"U:{u}\nA:{a}")

    hist = "\n---\n".join(lines)
    if summary:
        if hist:
            return f"{summary}\n---\n{hist}"
        return summary
    return hist


async def summarize_thread(chat_id: int):
    """Summarize thread history and reset dialogue counter."""
    st = await db_get(chat_id)
    thread_id = st.get("thread_id")
    if not thread_id:
        return
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="asc", limit=100)
    lines = []
    for m in msgs.data:
        if m.role not in ("user", "assistant"):
            continue
        parts = []
        for c in m.content:
            if c.type == "text":
                parts.append(c.text.value.strip())
        if parts:
            lines.append(f"{m.role.upper()}: {' '.join(parts)}")
    if not lines:
        return
    prompt = "Summarize the following dialogue:\n" + "\n".join(lines) + "\nSummary:"
    try:
        logger.info("Requesting OpenAI summary for chat %s", chat_id)
        resp = await asyncio.to_thread(
            client.responses.create, model=MODEL, input=prompt, temperature=0
        )
        summary = (resp.output_text or "").strip()
    except Exception as e:
        logger.exception("Failed to summarize thread %s: %s", thread_id, e)
        summary = ""
    await db_set(chat_id, last_summary=summary, dialogue_n=0)
    for m in msgs.data:
        with contextlib.suppress(Exception):
            client.beta.threads.messages.delete(thread_id=thread_id, message_id=m.id)


async def cleanup_threads():
    def _rows():
        with get_connection() as conn:
            return conn.execute(
                "SELECT chat_id, thread_id FROM chats WHERE thread_id IS NOT NULL AND chapter IS NULL"
            ).fetchall()

    rows = await asyncio.to_thread(_rows)
    for r in rows:
        try:
            client.beta.threads.delete(r["thread_id"])
            await db_set(r["chat_id"], thread_id=None)
        except Exception as e:
            logger.exception("Failed to delete thread %s: %s", r["thread_id"], e)
            continue


def cleanup_hero_cache(max_age_hours: int = 24):
    cutoff = time.time() - max_age_hours * 3600
    for p in HERO_CTX_CACHE_DIR.glob("*.txt"):
        if p.stat().st_mtime < cutoff:
            with contextlib.suppress(Exception):
                p.unlink()


async def periodic_cleanup(context: ContextTypes.DEFAULT_TYPE):
    await cleanup_threads()
    cleanup_hero_cache()
    CHAOS.cleanup(settings.chaos_cleanup_max_age_hours)


async def silence_watchdog(context: ContextTypes.DEFAULT_TYPE):
    now = time.time()
    for chat_id, ts in list(LAST_ACTIVITY.items()):
        if now - ts <= 120:
            continue
        st = await db_get(chat_id)
        ch = st.get("chapter")
        if not ch:
            continue
        chapter_text = CHAPTERS.get(ch)
        participants = guess_participants(chapter_text)
        if not participants:
            continue
        await load_chapter_context_all(chapter_text, participants)
        hero = random.choice(participants)
        thread_id = await ensure_thread(chat_id)
        client.beta.threads.messages.create(thread_id=thread_id, role="user", content="(silence)")
        scene_prompt = build_scene_prompt(
            ch,
            chapter_text,
            [hero],
            "(silence)",
            await compress_history_for_prompt(chat_id),
        )
        thread_add_message(thread_id, "user", scene_prompt)
        await run_and_wait(thread_id)
        st_check = await db_get(chat_id)
        if st_check.get("chapter") != ch or not st_check.get("accepted"):
            cancel_idle(chat_id)
            return
        text = thread_last_text(thread_id).strip()
        if not text:
            text = f"**{hero}**: (тишина)"
        chat = await context.bot.get_chat(chat_id)
        await send_hero_lines(chat, text, context)
        bot_ts = time.time()
        LAST_ACTIVITY[chat_id] = bot_ts
        CHAOS.silence[str(chat_id)] = 0

        async def idle_loop():
            nonlocal bot_ts
            while True:
                await asyncio.sleep(random.uniform(10, 30))
                st_check = await db_get(chat_id)
                if st_check.get("chapter") != ch:
                    break
                if LAST_ACTIVITY.get(chat_id, 0) > bot_ts:
                    break
                responders, _ = CHAOS.pick(str(chat_id), chapter_text, None)
                responders = [r for r in responders if r in participants] or participants[: min(3, len(participants))]
                scene_prompt = build_scene_prompt(
                    ch,
                    chapter_text,
                    responders,
                    None,
                    await compress_history_for_prompt(chat_id),
                )
                thread_add_message(thread_id, "user", scene_prompt)
                await run_and_wait(thread_id)
                st_post = await db_get(chat_id)
                if st_post.get("chapter") != ch or not st_post.get("accepted"):
                    cancel_idle(chat_id)
                    return
                text_inner = thread_last_text(thread_id).strip()
                if not text_inner:
                    text_inner = "\n".join(f"**{r}**: (тишина)" for r in responders)
                await send_hero_lines(chat, text_inner, context)
                bot_ts = time.time()
                LAST_ACTIVITY[chat_id] = bot_ts

        cancel_idle(chat_id)
        task = asyncio.create_task(idle_loop())
        IDLE_TASKS[chat_id] = task
        task.add_done_callback(lambda t, cid=chat_id: IDLE_TASKS.pop(cid, None))

# =========================
# Telegram Handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    logger.info("/start from chat %s", chat_id)
    await db_get(chat_id)
    cancel_idle(chat_id)
    LAST_ACTIVITY.pop(chat_id, None)
    await db_set(chat_id, chapter=None, dialogue_n=0, last_summary="")
    await ensure_thread(chat_id)
    await update.message.reply_text(
        DISCLAIMER,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("OK", callback_data="ok"),
                                            InlineKeyboardButton("NO", callback_data="no")]])
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n/start — disclaimer\n/menu — chapters\n/reload — reload docs\n/reload_heroes — reload /heroes\n"
        "Workflow: OK → choose chapter → live dialogue starts → reply to steer them."
    )

async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    await db_get(chat_id)
    cancel_idle(chat_id)
    LAST_ACTIVITY.pop(chat_id, None)
    await db_set(chat_id, chapter=None, dialogue_n=0, last_summary="")
    await update.message.reply_text("YOU CHOOSE:", reply_markup=chapters_menu())

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = reload_chapters()
    await update.message.reply_text(f"Chapters reloaded: {n} files.")

# [HEROES] команда — перечитать промпты персонажей
async def reload_heroes_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = reload_heroes()
    st = await db_get(update.effective_chat.id)
    if st.get("chapter"):
        chapter_text = CHAPTERS[st["chapter"]]
        participants = guess_participants(chapter_text)
        await load_chapter_context_all(chapter_text, participants)
    await update.message.reply_text(f"Heroes reloaded: {n} persona files.")


def parse_lines(text: str):
    sep = r"[:\-\–—]"
    name_line_star = re.compile(r"^\*{1,2}\s*(.+?)\s*\*{1,2}$")
    name_line_heading = re.compile(r"^\s{0,3}#{1,6}\s*(.+?)\s*#*\s*$")
    inline_star = re.compile(rf"^\*{{1,2}}\s*(.+?)\s*\*{{1,2}}\s*{sep}\s*(.*)")
    inline_plain = re.compile(rf"^(?!\*{{1,2}})(.+?)\s*{sep}\s*(.*)")
    current_name = None
    buffer: list[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        m_inline = inline_star.match(stripped) or inline_plain.match(stripped)
        if m_inline:
            if current_name and buffer:
                yield current_name, "\n".join(buffer).strip()
            yield m_inline.group(1).strip(), m_inline.group(2)
            current_name, buffer = None, []
            continue
        m_name = name_line_star.match(stripped) or name_line_heading.match(stripped)
        if m_name:
            if current_name and buffer:
                yield current_name, "\n".join(buffer).strip()
            current_name = m_name.group(1).strip()
            buffer = []
        elif current_name is not None:
            buffer.append(line)
    if current_name and buffer:
        yield current_name, "\n".join(buffer).strip()


async def send_hero_lines(
    chat,
    text: str,
    context: ContextTypes.DEFAULT_TYPE,
    reply_to_message_id: int | None = None,
):
    sent = False
    for name, line in parse_lines(text):
        typing = await chat.send_message(f"{name} is typing…")
        delay = random.uniform(3, 5)
        elapsed = 0.0
        while elapsed < delay:
            await context.bot.send_chat_action(chat.id, ChatAction.TYPING)
            step = min(4, delay - elapsed)
            await asyncio.sleep(step)
            elapsed += step
        await typing.delete()
        header = f"**{name}**"
        await chat.send_message(
            f"{header}\n{line}",
            parse_mode=ParseMode.MARKDOWN,
            reply_to_message_id=reply_to_message_id,
        )
        sent = True
    if not sent and text.strip():
        await chat.send_message(
            text,
            parse_mode=ParseMode.MARKDOWN,
            reply_to_message_id=reply_to_message_id,
        )

async def on_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = update.effective_chat.id
    LAST_ACTIVITY[chat_id] = time.time()
    await db_get(chat_id)
    thread_id = await ensure_thread(chat_id)

    if q.data == "no":
        await q.edit_message_text("Goodbye.")
        return

    if q.data == "ok":
        await db_set(chat_id, accepted=1)
        await q.edit_message_text("YOU CHOOSE:", reply_markup=chapters_menu())
        return

    if q.data.startswith("ch_"):
        cancel_idle(chat_id)
        match = re.match(r"^ch_(\d+)$", q.data)
        if not match:
            logger.warning("Chat %s sent malformed chapter callback %s", chat_id, q.data)
            await q.message.chat.send_message("Unknown chapter")
            return
        try:
            ch = int(match.group(1))
        except ValueError:
            logger.warning("Chat %s sent non-integer chapter %s", chat_id, match.group(1))
            await q.message.chat.send_message("Unknown chapter")
            return
        logger.info("Chat %s selected chapter %s", chat_id, ch)
        if ch not in CHAPTERS:
            logger.warning("Chat %s selected invalid chapter %s", chat_id, ch)
            await q.message.chat.send_message("Unknown chapter")
            return
        await db_set(chat_id, chapter=ch, dialogue_n=0, last_summary="")
        chapter_text = CHAPTERS[ch]
        participants = guess_participants(chapter_text)
        await load_chapter_context_all(chapter_text, participants)

        responders, mode = CHAOS.pick(str(chat_id), chapter_text, "(enter)")
        responders = [r for r in responders if r in participants] or participants[: min(3, len(participants))]

        scene_prompt = build_scene_prompt(ch, chapter_text, responders, "(enters the room)", await compress_history_for_prompt(chat_id))
        thread_add_message(thread_id, "user", scene_prompt)
        await run_and_wait(thread_id)
        st_check = await db_get(chat_id)
        if st_check.get("chapter") != ch or not st_check.get("accepted"):
            cancel_idle(chat_id)
            return
        text = thread_last_text(thread_id).strip()
        if not text:
            text = "\n".join(f"**{r}**: (тишина)" for r in responders)
        glitch = MARKOV.glitch()

        try:
            await send_hero_lines(q.message.chat, text, context)
        except Exception:
            logger.exception("Failed to send hero lines for chat %s", chat_id)
            if hasattr(q.message.chat, "send_message"):
                await q.message.chat.send_message("Failed to load chapter")
        else:
            await q.message.delete()
            if glitch and hasattr(q.message.chat, "send_message"):
                await q.message.chat.send_message(glitch, parse_mode=ParseMode.MARKDOWN)
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    now = time.time()
    last = LAST_ACTIVITY.get(chat_id)
    LAST_ACTIVITY[chat_id] = now
    st = await db_get(chat_id)
    msg = (update.message.text or "").strip()
    logger.info("Received message in chat %s: %s", chat_id, msg)

    if not st["accepted"]:
        await update.message.reply_text(
            "Tap OK to enter.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("OK", callback_data="ok")]])
        )
        return

    if not st["chapter"]:
        await update.message.reply_text("Pick a chapter first.", reply_markup=chapters_menu())
        return

    thread_id = await ensure_thread(chat_id)
    ch = st["chapter"]
    chapter_text = CHAPTERS[ch]
    participants = guess_participants(chapter_text)
    await load_chapter_context_all(chapter_text, participants)

    if last and now - last > INACTIVITY_TIMEOUT and participants:
        hero = random.choice(participants)
        pre_text = f"**{hero}**: а, опять ты…"
        thread_add_message(thread_id, "assistant", pre_text)
        await send_hero_lines(
            update.message.chat,
            pre_text,
            context,
            reply_to_message_id=update.message.message_id,
        )

    responders, mode = CHAOS.pick(str(chat_id), chapter_text, msg)
    responders = [r for r in responders if r in participants] or participants[: min(3, len(participants))]
    mentioned = [n for n in detect_names(msg) if n in participants]
    for n in mentioned:
        if n not in responders:
            responders.append(n)

    reply = getattr(update.message, "reply_to_message", None)
    if reply and getattr(reply, "text", None):
        hero_name = next((name for name, _ in parse_lines(reply.text)), None)
        if hero_name and hero_name in participants:
            if hero_name in responders:
                responders.remove(hero_name)
            responders.insert(0, hero_name)

    logger.info("Posting raw user message to thread %s", thread_id)
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=f"USER SAID: {msg}")
    scene_prompt = build_scene_prompt(ch, chapter_text, responders, msg, await compress_history_for_prompt(chat_id))
    thread_add_message(thread_id, "user", scene_prompt)
    await run_and_wait(thread_id)
    st_check = await db_get(chat_id)
    if st_check.get("chapter") != ch or not st_check.get("accepted"):
        cancel_idle(chat_id)
        return

    text = thread_last_text(thread_id).strip()
    if not text:
        text = "\n".join(f"**{r}**: (тишина)" for r in responders)
    glitch = MARKOV.glitch()

    await send_hero_lines(
        update.message.chat,
        text,
        context,
        reply_to_message_id=update.message.message_id,
    )
    if glitch:
        await update.message.chat.send_message(glitch, parse_mode=ParseMode.MARKDOWN)
    new_n = st["dialogue_n"] + 1
    await db_set(chat_id, dialogue_n=new_n)
    if new_n % SUMMARY_EVERY == 0:
        asyncio.create_task(summarize_thread(chat_id))

# =========================
# Main
# =========================
async def reset_updates():
    """Remove existing webhooks and terminate other sessions for this bot."""
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        with contextlib.suppress(RetryAfter):
            await bot.delete_webhook(drop_pending_updates=True)
            await bot.get_updates()
    finally:
        with contextlib.suppress(RetryAfter):
            await bot.close()


def main():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("Set TELEGRAM_TOKEN env var")
    if not settings.openai_api_key:
        raise RuntimeError("Set OPENAI_API_KEY env var")
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(reset_updates())
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("menu", menu_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("reload_heroes", reload_heroes_cmd))  # [HEROES]
    app.add_handler(CallbackQueryHandler(on_click))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    if app.job_queue:
        app.job_queue.run_repeating(periodic_cleanup, interval=3600, first=3600)
        app.job_queue.run_repeating(silence_watchdog, interval=10, first=10)
    else:
        print("Job queue disabled; periodic cleanup skipped.")
    loop.run_until_complete(app.bot.set_my_commands([("menu", "CHAPTERS"), ("start", "LETSGO")]))
    loop.run_until_complete(app.bot.set_chat_menu_button(menu_button=MenuButtonCommands()))
    print("SUPPERTIME (Assistants API) — ready.")
    webhook_url = os.getenv("WEBHOOK_URL")
    if webhook_url:
        from urllib.parse import urlparse

        parsed = urlparse(webhook_url)
        url_path = parsed.path.lstrip("/") or ""
        port = parsed.port or int(os.getenv("PORT", "8443"))
        app.run_webhook(
            listen="0.0.0.0",
            port=port,
            url_path=url_path,
            webhook_url=webhook_url,
            allowed_updates=Update.ALL_TYPES,
        )
    else:
        app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
