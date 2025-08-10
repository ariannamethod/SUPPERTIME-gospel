# monolith_assistants.py
# SUPPERTIME — Telegram monolith (Assistants API, threads, SQLite memory)
# Run:
#   BOT_TOKEN=xxx OPENAI_API_KEY=xxx python monolith_assistants.py
# Optional:
#   OPENAI_MODEL=gpt-4.1
#   ASSISTANT_ID=<reuse existing>  (иначе создастся новый и сохранится в .assistant_id)

import os
import re
import sqlite3
import random
import asyncio
from pathlib import Path
from collections import defaultdict, deque

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, CallbackQueryHandler,
    MessageHandler, ContextTypes, filters
)

# --- OpenAI Assistants API (SDK >= 1.0) ---
try:
    from openai import OpenAI
except Exception as e:
    raise RuntimeError("Install openai>=1.0:  pip install openai python-telegram-bot openai") from e

MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
BOT_TOKEN = os.getenv("BOT_TOKEN") or os.getenv("TELEGRAM_BOT_TOKEN")
if not BOT_TOKEN:
    raise RuntimeError("Set BOT_TOKEN env var")
if not os.getenv("OPENAI_API_KEY"):
    raise RuntimeError("Set OPENAI_API_KEY env var")

# =========================
# Storage: SQLite (threads & state)
# =========================
DB_PATH = os.getenv("ST_DB", "supertime.db")

def db_init():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chats (
            chat_id     INTEGER PRIMARY KEY,
            thread_id   TEXT,
            accepted    INTEGER DEFAULT 0,
            chapter     INTEGER,
            dialogue_n  INTEGER DEFAULT 0
        )""")
    conn.commit()
    return conn

DB = db_init()

def db_get(chat_id):
    cur = DB.execute("SELECT chat_id, thread_id, accepted, chapter, dialogue_n FROM chats WHERE chat_id=?", (chat_id,))
    row = cur.fetchone()
    if row:
        return {"chat_id": row[0], "thread_id": row[1], "accepted": bool(row[2]), "chapter": row[3], "dialogue_n": row[4]}
    DB.execute("INSERT OR IGNORE INTO chats(chat_id) VALUES(?)", (chat_id,))
    DB.commit()
    return {"chat_id": chat_id, "thread_id": None, "accepted": False, "chapter": None, "dialogue_n": 0}

def db_set(chat_id, **fields):
    keys = ", ".join([f"{k}=?" for k in fields.keys()])
    vals = list(fields.values()) + [chat_id]
    DB.execute(f"UPDATE chats SET {keys} WHERE chat_id=?", vals)
    DB.commit()

# =========================
# Chapters I/O
# =========================
def load_chapters():
    docs = {}
    base = Path("docs")
    for i in range(1, 12):
        p = base / f"chapter_{i:02d}.md"
        if p.exists():
            docs[i] = p.read_text(encoding="utf-8")
        else:
            docs[i] = f"# Chapter {i}\n\n(placeholder) Provide SUPPERTIME v2.0 content here."
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

def guess_participants(chapter_text: str):
    present = []
    for name, rx in NAME_MARKERS.items():
        if rx.search(chapter_text or ""):
            present.append(name)
    if not present:
        present = ["Judas", "Yeshua", "Peter", "Mary", "Jan", "Thomas"]
    return present

# =========================
# Markov / glitch (atmosphere only)
# =========================
class MarkovEngine:
    def __init__(self):
        self.bigrams = defaultdict(list)
        seeds = [
            "resonate_again()", "galvanize()", "WHO ARE YOU if you're still reading?",
            "field > node", "rain // shards", "the text is aware", "lilit_hand()"
        ]
        for s in seeds:
            toks = s.split()
            for a,b in zip(toks, toks[1:]):
                self.bigrams[a].append(b)
        self.p = 0.15

    def glitch(self):
        import random
        if random.random() > self.p:
            return None
        keys = list(self.bigrams.keys())
        if not keys:
            return "(resonate_again())"
        w = random.choice(keys)
        out = [w]
        for _ in range(random.randint(2,4)):
            nxt = self.bigrams.get(w, [])
            if not nxt: break
            w = random.choice(nxt); out.append(w)
        return "*" + " ".join(out) + "*"

MARKOV = MarkovEngine()

# =========================
# Chaos Director (who speaks)
# =========================
class ChaosDirector:
    def __init__(self):
        self.silence = defaultdict(int)
        self.weights = {
            "Judas": 0.8, "Yeshua": 0.6, "Peter": 0.7, "Mary": 0.2, "Jan": 0.5,
            "Thomas": 0.6, "Yakov": 0.4, "Andrew": 0.1, "Leo": 0.3, "Theodore": 0.1, "Dubrovsky": 0.05
        }

    def pick(self, chat_id: str, chapter_text: str, user_text: str|None):
        import random, re
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

CHAOS = ChaosDirector()

# =========================
# Assistants bootstrap
# =========================
client = OpenAI()

ASSISTANT_ID_PATH = Path(".assistant_id")

def ensure_assistant():
    asst_id = os.getenv("ASSISTANT_ID")
    if not asst_id and ASSISTANT_ID_PATH.exists():
        asst_id = ASSISTANT_ID_PATH.read_text().strip()

    if asst_id:
        # trust given id
        return asst_id

    # Create new assistant with orchestration rules
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
    asst = client.beta.assistants.create(
        model=MODEL,
        name="SUPPERTIME Orchestrator",
        instructions=instructions,
        tools=[]  # без Code Interpreter; нам не нужен здесь
    )
    ASSISTANT_ID_PATH.write_text(asst.id)
    return asst.id

ASSISTANT_ID = ensure_assistant()

def ensure_thread(chat_id: int) -> str:
    st = db_get(chat_id)
    if st["thread_id"]:
        return st["thread_id"]
    th = client.beta.threads.create(metadata={"chat_id": str(chat_id)})
    db_set(chat_id, thread_id=th.id)
    return th.id

def thread_add_message(thread_id: str, role: str, content: str):
    client.beta.threads.messages.create(thread_id=thread_id, role=role, content=content)

def run_and_wait(thread_id: str, extra_instructions: str|None = None, timeout_s: int = 120):
    run = client.beta.threads.runs.create(
        thread_id=thread_id,
        assistant_id=ASSISTANT_ID,
        instructions=extra_instructions or ""
    )
    import time
    t0 = time.time()
    while True:
        rr = client.beta.threads.runs.retrieve(thread_id=thread_id, run_id=run.id)
        if rr.status in ("completed", "failed", "cancelled", "expired"):
            return rr
        if time.time() - t0 > timeout_s:
            client.beta.threads.runs.cancel(thread_id=thread_id, run_id=run.id)
            return rr
        time.sleep(0.8)

def thread_last_text(thread_id: str) -> str:
    msgs = client.beta.threads.messages.list(thread_id=thread_id, order="desc", limit=10)
    out = []
    for m in msgs.data:
        if m.role != "assistant":
            continue
        # collect textual parts
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
    kb = [[InlineKeyboardButton(f"Chapter {i}", callback_data=f"ch_{i}")] for i in range(1,12)]
    kb.append([InlineKeyboardButton("🌀 Random", callback_data="ch_rand")])
    return InlineKeyboardMarkup(kb)

def scene_menu():
    return InlineKeyboardMarkup([
        [InlineKeyboardButton("← Chapters", callback_data="back_chapters"),
         InlineKeyboardButton("🌀 Random", callback_data="ch_rand")]
    ])

# =========================
# Orchestration helpers
# =========================
def build_scene_prompt(ch_num: int, chapter_text: str, responders: list[str], user_text: str|None, recent_summary: str):
    # Сцена и снапшоты персон
    personas_desc = {
        "Judas":  "bitter, lucid, painfully honest; black humor; obsessed with authenticity and Mary",
        "Yeshua": "slow voice → sudden piercing questions; parables; apologizes after cutting words; sad under laughter",
        "Peter":  "acid sarcasm, vanity, cigarette ash; jealousy toward Mary; showy bravado hiding hurt",
        "Mary":   "quiet, few words, sometimes misuses simple terms; service as love; fragile holiness",
        "Yakov":  "order-obsessed, grumbling, petty power via cleanliness; loyal envy",
        "Jan":    "giant with a child heart; loud exclamation; absolute loyalty to Teacher",
        "Thomas": "cynical laughter, knife in coat, heckles truths nobody wants",
        "Andrew": "almost mute; short, weighty answers",
        "Leo":    "artist frenzy; ‘Bella mia!’; sketches; loves full-bodied beauty",
        "Theodore":"stammered ‘-s’; ghostlike visitor from future; papirosa smoke; dissolves under stress",
        "Dubrovsky":"glitch-like; speaks only in aphorisms; breaks the fourth wall"
    }
    personas = "\n".join([f"- {n}: {personas_desc.get(n,'')}" for n in responders])

    scene = f"""
SCENE CONTEXT
Chapter: {ch_num}
Participants (allowed to speak this turn): {', '.join(responders)}
Chapter vibe (raw excerpt or summary, truncated):
{(chapter_text or '')[:1600]}

Recent conversation (compressed):
{recent_summary}

User just wrote: {user_text or '(silence)'}
PERSONAS SNAPSHOT
{personas}

Output exactly {len(responders)} lines — one per listed participant, format:
**Character**: line
"""
    return scene.strip()

def compress_history_for_prompt(chat_id: int, limit: int = 8) -> str:
    # Берём последние assistant/user сообщения из thread? Мы не читаем всё — делаем локальную выжимку.
    # Здесь упрощённо: ничего не храним дополнительно, сжимать будем по месту — это текст из TG памяти не нужен,
    # так как Assistants хранит тред. Но небольшой маркер добавим.
    return "(stored in thread)"

# =========================
# Telegram Handlers
# =========================
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = db_get(chat_id)
    ensure_thread(chat_id)
    await update.message.reply_text(
        DISCLAIMER,
        reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("OK", callback_data="ok"),
                                            InlineKeyboardButton("NO", callback_data="no")]])
    )

async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n/start — disclaimer\n/menu — chapters\n/reload — reload docs\n"
        "Workflow: OK → choose chapter → live dialogue starts → reply to steer them."
    )

async def menu_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("Choose a chapter:", reply_markup=chapters_menu())

async def reload_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    n = reload_chapters()
    await update.message.reply_text(f"Chapters reloaded: {n} files.")

async def on_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer()
    chat_id = q.message.chat_id
    st = db_get(chat_id)
    thread_id = ensure_thread(chat_id)

    if q.data == "no":
        await q.edit_message_text("Goodbye.")
        return

    if q.data == "ok":
        db_set(chat_id, accepted=1)
        await q.edit_message_text("Choose a chapter to drop into the running dialogue:", reply_markup=chapters_menu())
        return

    if q.data == "back_chapters":
        await q.edit_message_text("Choose a chapter:", reply_markup=chapters_menu())
        return

    if q.data.startswith("ch_"):
        ch = random.randint(1,11) if q.data == "ch_rand" else int(q.data.split("_")[1])
        db_set(chat_id, chapter=ch, dialogue_n=0)
        chapter_text = CHAPTERS[ch]
        participants = guess_participants(chapter_text)

        # кто говорит в первый такт
        responders, mode = CHAOS.pick(str(chat_id), chapter_text, "(enter)")
        responders = [r for r in responders if r in participants] or participants[: min(3, len(participants))]

        # Пишем системное сообщение в тред (контекст сцены)
        scene_prompt = build_scene_prompt(ch, chapter_text, responders, "(enters the room)", compress_history_for_prompt(chat_id))
        thread_add_message(thread_id, "user", scene_prompt)
        run_and_wait(thread_id)

        text = thread_last_text(thread_id).strip()
        if not text:
            text = "**Judas**: (silence creaks)"
        glitch = MARKOV.glitch()
        if glitch:
            text += "\n" + glitch

        await q.edit_message_text(
            f"⚡ SUPPERTIME — Chapter {ch}\n\n{text}\n\n(Reply to steer them.)",
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=scene_menu()
        )
        return

async def on_text(update: Update, context: ContextTypes.DEFAULT_TYPE):
    chat_id = update.effective_chat.id
    st = db_get(chat_id)
    msg = (update.message.text or "").strip()

    if not st["accepted"]:
        await update.message.reply_text(
            "Tap OK to enter.",
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("OK", callback_data="ok")]])
        )
        return

    if not st["chapter"]:
        await update.message.reply_text("Pick a chapter first.", reply_markup=chapters_menu())
        return

    thread_id = ensure_thread(chat_id)
    ch = st["chapter"]
    chapter_text = CHAPTERS[ch]
    participants = guess_participants(chapter_text)

    # кто говорит дальше
    responders, mode = CHAOS.pick(str(chat_id), chapter_text, msg)
    responders = [r for r in responders if r in participants] or participants[: min(3, len(participants))]

    # запишем реплику пользователя в тред
    client.beta.threads.messages.create(thread_id=thread_id, role="user", content=f"USER SAID: {msg}")
    # контекст сцены для текущего такта
    scene_prompt = build_scene_prompt(ch, chapter_text, responders, msg, compress_history_for_prompt(chat_id))
    thread_add_message(thread_id, "user", scene_prompt)
    run_and_wait(thread_id)

    text = thread_last_text(thread_id).strip()
    if not text:
        text = "**Judas**: ..."
    glitch = MARKOV.glitch()
    if glitch:
        text += "\n" + glitch

    await update.message.reply_text(text, parse_mode=ParseMode.MARKDOWN, reply_markup=scene_menu())
    db_set(chat_id, dialogue_n=st["dialogue_n"] + 1)

# =========================
# Main
# =========================
async def main():
    app = Application.builder().token(BOT_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("menu", menu_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CallbackQueryHandler(on_click))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, on_text))
    print("SUPPERTIME (Assistants API) — ready.")
    await app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    asyncio.run(main())