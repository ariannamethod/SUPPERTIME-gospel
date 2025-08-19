import re
import asyncio
import random
import contextlib
import time
import os
from pathlib import Path
import logging
import logger

from config import settings

from telegram import (
    Update,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Bot,
    MenuButtonCommands,
)
from telegram.constants import ParseMode, ChatAction
from telegram.ext import (
    Application,
    CommandHandler,
    CallbackQueryHandler,
    MessageHandler,
    ContextTypes,
    filters,
)
from telegram.error import RetryAfter

from openai import APIConnectionError, APITimeoutError, RateLimitError as OpenAIRetryAfter

from theatre import (
    client,
    load_chapter_context_all,
    build_scene_prompt,
    parse_lines,
    is_valid_scene,
    CHAOS,
    MARKOV,
    CHAPTER_TITLES,
    cleanup_hero_cache,
)
from db import db_get, db_set, db_init, SUMMARY_EVERY

MODEL = settings.openai_model
TEMPERATURE = settings.openai_temperature
TELEGRAM_TOKEN = settings.telegram_token

logger = logging.getLogger(__name__)

OPENAI_TIMEOUT = 30
OPENAI_RETRY_ATTEMPTS = 3
OPENAI_RETRY_DELAY = 1

# =========================
# Chapters I/O
# =========================

def load_chapters():
    docs: dict[int, str] = {}
    base = Path("docs")
    for path in sorted(base.glob("chapter_*.md")):
        match = re.match(r"chapter_(\d+)\.md", path.name)
        if match:
            i = int(match.group(1))
            docs[i] = path.read_text(encoding="utf-8")
    for i, title in CHAPTER_TITLES.items():
        docs.setdefault(
            i,
            f"# {title}\n\n(placeholder) Provide SUPPERTIME v2.0 content here.",
        )
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
# Assistants bootstrap
# =========================
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


async def request_scene(thread_id: str, participants: list[str], retries: int = 2) -> str:
    """Request a scene from OpenAI with retries and exponential backoff."""
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            text = thread_last_text(thread_id).strip()
            if is_valid_scene(text, participants):
                return text

            msg = (
                f"Respond with exactly {len(participants)} lines in the format '**Name**: dialogue' using only these names: "
                f"{', '.join(participants)}."
            )
            thread_add_message(thread_id, "user", msg)
            await run_and_wait(thread_id)
            text = thread_last_text(thread_id).strip()
            if is_valid_scene(text, participants):
                return text
        except APIConnectionError as e:
            logger.warning(
                "Scene request connection error (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                e,
            )
            await asyncio.sleep(delay)
            delay *= 2
        except APITimeoutError as e:
            logger.warning(
                "Scene request timeout (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                e,
            )
            await asyncio.sleep(delay)
            delay *= 2
        except OpenAIRetryAfter as e:
            wait = getattr(e, "retry_after", delay)
            logger.warning(
                "Scene request rate limited (attempt %d/%d): %s",
                attempt + 1,
                retries + 1,
                e,
            )
            await asyncio.sleep(wait)
            delay = max(delay * 2, wait * 2)
    logger.error("Failed to produce scene for %s after %d attempts", participants, retries + 1)
    return "**Narrator**: The scene falls silent."


# =========================
# Telegram helpers
# =========================
async def send_hero_lines(
    chat,
    text: str,
    context: ContextTypes.DEFAULT_TYPE,
    reply_to_message_id: int | None = None,
    participants: list[str] | None = None,
):
    try:
        lines = list(parse_lines(text))
    except ValueError:
        lines = []
    if participants is not None:
        names = [n for n, _ in lines]
        if len(lines) != len(participants) or any(n not in participants for n in names):
            logger.warning(
                "Expected %d lines for participants %s, got %d (%s); using Narrator fallback",
                len(participants),
                participants,
                len(lines),
                names,
            )
            await chat.send_message(
                f"**Narrator**\n{text}",
                parse_mode=ParseMode.MARKDOWN,
                reply_to_message_id=reply_to_message_id,
            )
            return
    sent = False
    for name, line in lines:
        typing = await chat.send_message(f"{name} is typing…")
        delay = random.uniform(3, 5)
        elapsed = 0.0
        while elapsed < delay:
            await context.bot.send_chat_action(chat.id, ChatAction.TYPING)
            step = min(1, delay - elapsed)
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


# =========================
# Cleanup helpers
# =========================
async def cleanup_threads():
    import sqlite3

    def _rows():
        with sqlite3.connect(settings.db_path) as conn:
            return conn.execute("SELECT chat_id, thread_id FROM chats WHERE thread_id IS NOT NULL AND chapter IS NULL").fetchall()

    rows = await asyncio.to_thread(_rows)
    for r in rows:
        try:
            client.beta.threads.delete(r["thread_id"])
            await db_set(r["chat_id"], thread_id=None)
        except Exception as e:
            logger.exception("Failed to delete thread %s: %s", r["thread_id"], e)
            continue


def cancel_idle(chat_id: int):
    task = IDLE_TASKS.pop(chat_id, None)
    if task:
        task.cancel()


async def periodic_cleanup(context: ContextTypes.DEFAULT_TYPE):
    await cleanup_threads()
    cleanup_hero_cache()
    CHAOS.cleanup(settings.chaos_cleanup_max_age_hours)


LAST_ACTIVITY: dict[int, float] = {}
IDLE_TASKS: dict[int, asyncio.Task] = {}
INACTIVITY_TIMEOUT = 120


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
        text = await request_scene(thread_id, [hero])
        st_check = await db_get(chat_id)
        if st_check.get("chapter") != ch or not st_check.get("accepted"):
            cancel_idle(chat_id)
            return
        if not text:
            text = f"**{hero}**: (тишина)"
        chat = await context.bot.get_chat(chat_id)
        await send_hero_lines(chat, text, context, participants=[hero])
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
                text_inner = await request_scene(thread_id, responders)
                st_post = await db_get(chat_id)
                if st_post.get("chapter") != ch or not st_post.get("accepted"):
                    cancel_idle(chat_id)
                    return
                if not text_inner:
                    text_inner = "\n".join(f"**{r}**: (тишина)" for r in responders)
                await send_hero_lines(chat, text_inner, context, participants=responders)
                bot_ts = time.time()
                LAST_ACTIVITY[chat_id] = bot_ts

        cancel_idle(chat_id)
        task = asyncio.create_task(idle_loop())
        IDLE_TASKS[chat_id] = task
        task.add_done_callback(lambda t, cid=chat_id: IDLE_TASKS.pop(cid, None))


# =========================
# Conversation helpers
# =========================
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
        reply_markup=InlineKeyboardMarkup(
            [[InlineKeyboardButton("OK", callback_data="ok"), InlineKeyboardButton("NO", callback_data="no")]]
        ),
    )


async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n/start — disclaimer\n/menu — chapters\n/reload — reload docs\n/reload_heroes — reload /heroes\n" "Workflow: OK → choose chapter → live dialogue starts → reply to steer them.",
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


async def reload_heroes_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE):
    from theatre import hero_manager  # local import to avoid circular

    n = hero_manager.reload()
    st = await db_get(update.effective_chat.id)
    if st.get("chapter"):
        chapter_text = CHAPTERS[st["chapter"]]
        participants = guess_participants(chapter_text)
        await load_chapter_context_all(chapter_text, participants)
    await update.message.reply_text(f"Heroes reloaded: {n} persona files.")


async def on_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    q = update.callback_query
    await q.answer(cache_time=1)
    chat_id = update.effective_chat.id
    LAST_ACTIVITY[chat_id] = time.time()
    await db_get(chat_id)

    if q.data == "no":
        await q.edit_message_text("Goodbye.")
        return
    if q.data == "ok":
        await db_set(chat_id, accepted=1, chapter=None, dialogue_n=0, last_summary="")
        await q.edit_message_text("YOU CHOOSE:", reply_markup=chapters_menu())
        return
    if q.data.startswith("ch_"):
        try:
            ch = int(q.data[3:])
        except ValueError:
            await q.message.chat.send_message("Unknown chapter")
            return
        if ch not in CHAPTERS:
            await q.message.chat.send_message("Unknown chapter")
            return

        async def handle_chapter():
            thread_id = await ensure_thread(chat_id)
            await db_set(chat_id, chapter=ch, dialogue_n=0, last_summary="")
            chapter_text = CHAPTERS[ch]
            participants = guess_participants(chapter_text)
            await load_chapter_context_all(chapter_text, participants)
            responders, _ = CHAOS.pick(str(chat_id), chapter_text, "(enter)")
            responders = [r for r in responders if r in participants] or participants[: min(3, len(participants))]
            scene_prompt = build_scene_prompt(
                ch, chapter_text, responders, "(enters the room)", await compress_history_for_prompt(chat_id)
            )
            thread_add_message(thread_id, "user", scene_prompt)
            await run_and_wait(thread_id)
            text = await request_scene(thread_id, responders)
            st_check = await db_get(chat_id)
            if st_check.get("chapter") != ch or not st_check.get("accepted"):
                cancel_idle(chat_id)
                return
            if not text:
                text = "\n".join(f"**{r}**: (тишина)" for r in responders)
            glitch = MARKOV.glitch()
            try:
                await send_hero_lines(q.message.chat, text, context, participants=responders)
            except Exception:
                logger.exception("Failed to send hero lines for chat %s", chat_id)
                await q.message.chat.send_message("Failed to load chapter")
                return
            await q.message.delete()
            if glitch:
                await q.message.chat.send_message(glitch, parse_mode=ParseMode.MARKDOWN)

        asyncio.create_task(handle_chapter())
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
            reply_markup=InlineKeyboardMarkup([[InlineKeyboardButton("OK", callback_data="ok")]]),
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
            participants=[hero],
        )

    responders, _ = CHAOS.pick(str(chat_id), chapter_text, msg)
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
    scene_prompt = build_scene_prompt(
        ch, chapter_text, responders, msg, await compress_history_for_prompt(chat_id)
    )
    thread_add_message(thread_id, "user", scene_prompt)
    await run_and_wait(thread_id)
    text = await request_scene(thread_id, responders)
    st_check = await db_get(chat_id)
    if st_check.get("chapter") != ch or not st_check.get("accepted"):
        cancel_idle(chat_id)
        return

    if not text:
        text = "\n".join(f"**{r}**: (тишина)" for r in responders)
    glitch = MARKOV.glitch()

    await send_hero_lines(
        update.message.chat,
        text,
        context,
        reply_to_message_id=update.message.message_id,
        participants=responders,
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
    loop.run_until_complete(db_init())
    from theatre import hero_manager

    hero_manager.reload()
    loop.run_until_complete(reset_updates())
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_cmd))
    app.add_handler(CommandHandler("menu", menu_cmd))
    app.add_handler(CommandHandler("reload", reload_cmd))
    app.add_handler(CommandHandler("reload_heroes", reload_heroes_cmd))
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
