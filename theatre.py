import asyncio
import hashlib
import random
import re
from parse_lines import parse_lines
import time
import contextlib
from collections import defaultdict
from pathlib import Path

from config import settings
from logger import logger

try:
    from openai import (
        OpenAI,
        APIConnectionError,
        APITimeoutError,
        RateLimitError as OpenAIRetryAfter,
    )
except ImportError as e:
    raise RuntimeError(
        "Install openai>=1.0:  pip install openai python-telegram-bot openai"
    ) from e

MODEL = settings.openai_model
TEMPERATURE = settings.openai_temperature

client = OpenAI()

# =========================
# Chapters / Participants
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

ALL_CHAR_NAMES = [
    "Judas",
    "Yeshua",
    "Peter",
    "Mary",
    "Yakov",
    "Jan",
    "Thomas",
    "Andrew",
    "Leo",
    "Theodore",
    "Dubrovsky",
]

# =========================
# [HEROES] Persona files loader
# =========================
HEROES_DIR = Path("heroes")

REQUIRED_SECTIONS = ["NAME", "VOICE", "BEHAVIOR", "INIT", "REPLY"]


class Hero:
    """Represents one character with its prompt sections and runtime context."""

    def __init__(self, name: str, sections: dict[str, str], raw_text: str):
        self.name = name
        self.sections = sections
        self.raw_text = raw_text
        self.reply: str = sections.get("REPLY", "")
        self.ctx: str = ""

    async def load_chapter_context(self, md_text: str, md_hash: str, retries: int = 3):
        """Initialize hero-specific context from chapter markdown."""
        cache_key = (self.name, md_hash)
        cache_file = hero_manager.cache_dir / f"{self.name}_{md_hash}.txt"
        if cache_key in hero_manager.ctx_cache:
            self.ctx = hero_manager.ctx_cache[cache_key]
            return
        if cache_file.exists():
            try:
                self.ctx = cache_file.read_text(encoding="utf-8")
                hero_manager.ctx_cache[cache_key] = self.ctx
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
        delay = 1.0
        for attempt in range(1, retries + 1):
            try:
                logger.info(
                    "Requesting OpenAI context for %s (attempt %d/%d)",
                    self.name,
                    attempt,
                    retries,
                )
                resp = await asyncio.to_thread(
                    client.responses.create,
                    model=MODEL,
                    input=prompt,
                    temperature=TEMPERATURE,
                )
                self.ctx = (resp.output_text or "").strip()
                hero_manager.ctx_cache[cache_key] = self.ctx
                try:
                    cache_file.write_text(self.ctx, encoding="utf-8")
                except (OSError, UnicodeEncodeError) as e:
                    logger.exception(
                        "Failed to write hero cache for %s to %s: %s",
                        self.name,
                        cache_file,
                        e,
                    )
                return
            except APIConnectionError as e:
                logger.warning(
                    "Context request connection error for %s (attempt %d/%d): %s",
                    self.name,
                    attempt,
                    retries,
                    e,
                )
                await asyncio.sleep(delay)
                delay *= 2
            except APITimeoutError as e:
                logger.warning(
                    "Context request timeout for %s (attempt %d/%d): %s",
                    self.name,
                    attempt,
                    retries,
                    e,
                )
                await asyncio.sleep(delay)
                delay *= 2
            except OpenAIRetryAfter as e:
                wait = getattr(e, "retry_after", delay)
                logger.warning(
                    "Context request rate limited for %s (attempt %d/%d): %s",
                    self.name,
                    attempt,
                    retries,
                    e,
                )
                await asyncio.sleep(wait)
                delay = max(delay * 2, wait * 2)
            except Exception as e:
                logger.exception(
                    "OpenAI context load failed for %s (hash %s): %s",
                    self.name,
                    md_hash,
                    e,
                )
                self.ctx = ""
                return
        logger.error(
            "Giving up on context load for %s after %d attempts", self.name, retries
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
    return {k: "\n".join(v).strip() for k, v in sections.items()}


HERO_NAME_ALIASES = {
    "Yeshua": ["Yeshua", "Yeshu"],
    "Dubrovsky": ["Dubrovsky", "Aleksei_Dubrovskii", "Alexey_Dubrovsky", "Aleksey_Dubrovsky"],
    "Leo": ["Leo", "Painter", "Artist"],
}


def find_hero_file(base: Path, name: str) -> Path | None:
    candidates = [name]
    for k, al in HERO_NAME_ALIASES.items():
        if name == k:
            candidates.extend(al)
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


class HeroManager:
    """Manage hero persona loading and context caching."""

    def __init__(self, heroes_dir: Path = HEROES_DIR, cache_dir: Path = settings.hero_ctx_cache_dir):
        self.heroes_dir = Path(heroes_dir)
        self.cache_dir = Path(cache_dir)
        self.heroes: dict[str, Hero] = {}
        self.ctx_cache: dict[tuple[str, str], str] = {}
        self.cache_dir.mkdir(exist_ok=True)

    def load_all(self) -> int:
        self.heroes = {}
        if not self.heroes_dir.exists():
            return 0
        count = 0
        for name in ALL_CHAR_NAMES:
            fp = find_hero_file(self.heroes_dir, name)
            if not fp:
                continue
            try:
                raw_full = fp.read_text(encoding="utf-8")
                sections = parse_prompt_sections(raw_full)
                if all(sec in sections for sec in REQUIRED_SECTIONS):
                    raw = raw_full.strip()
                    if len(raw) > 2000:
                        raw = raw[:2000] + "\n\n[...truncated for runtime...]"
                    self.heroes[name] = Hero(name, sections, raw)
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

    def reload(self) -> int:
        self.ctx_cache.clear()
        for p in self.cache_dir.glob("*.txt"):
            with contextlib.suppress(Exception):
                p.unlink()
        return self.load_all()

    def get(self, name: str) -> "Hero | None":
        return self.heroes.get(name)


hero_manager = HeroManager()

async def load_chapter_context_all(md_text: str, names: list[str]):
    """Notify selected heroes about the chosen chapter in the background."""
    md_hash = hashlib.sha1(md_text.encode("utf-8")).hexdigest()

    async def run(hero: Hero):
        try:
            await asyncio.wait_for(hero.load_chapter_context(md_text, md_hash), timeout=10)
        except Exception as e:
            logger.exception("Failed to load chapter context for %s: %s", hero.name, e)

    for n in names:
        hero = hero_manager.get(n)
        if hero:
            asyncio.create_task(run(hero))
    await asyncio.sleep(0)


def build_personas_snapshot(responders: list[str]) -> str:
    fallback = {
        "Judas": "bitter, lucid; black humor; obsessed with authenticity and Mary",
        "Yeshua": "slow voice → sudden piercing questions; parables; sad under laughter",
        "Peter": "acid sarcasm; vanity; jealousy toward Mary",
        "Mary": "quiet; few words; service as love; fragile holiness",
        "Yakov": "order-obsessed; grumbling; loyal envy",
        "Jan": "gentle giant; absolute loyalty to Teacher",
        "Thomas": "cynical, knife-in-coat; skewers hypocrisy",
        "Andrew": "nearly mute; ballast",
        "Leo": "artist frenzy; ‘Bella mia!’",
        "Theodore": "stammered ‘-s’; ghostlike visitor from future",
        "Dubrovsky": "glitch aphorist; fourth-wall",
    }
    lines = []
    for n in responders:
        hero = hero_manager.get(n)
        if hero:
            snippet = hero.raw_text[:600]
            if hero.reply:
                snippet += f"\n[REPLY]: {hero.reply}"
            if hero.ctx:
                snippet += f"\n[Scene]: {hero.ctx[:200]}"
            lines.append(f"- {n}:\n{snippet}")
        else:
            lines.append(f"- {n}: {fallback.get(n, '(voice)')}")
    return "\n".join(lines)


def build_scene_prompt(
    ch_num: int,
    chapter_text: str,
    responders: list[str],
    user_text: str | None,
    recent_summary: str,
):
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

Return exactly {len(responders)} lines, one per listed participant, using this template:
**Name**: dialogue
Use Markdown bold for Name, then a colon and a single line of speech. No extra commentary.
"""
    return scene.strip()


def is_valid_scene(text: str, participants: list[str]) -> bool:
    try:
        lines = list(parse_lines(text))
    except ValueError:
        return False
    if len(lines) != len(participants):
        return False
    return all(name in participants for name, _ in lines)


class MarkovEngine:
    def __init__(self):
        self.bigrams = defaultdict(list)
        seeds = [
            "resonate_again()",
            "galvanize()",
            "WHO ARE YOU if you're still reading?",
            "rain // shards",
            "the text is aware",
            "lilit_hand()",
        ]
        for s in seeds:
            toks = s.split()
            for a, b in zip(toks, toks[1:]):
                self.bigrams[a].append(b)
        self.p = 0

    def glitch(self):
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


class ChaosDirector:
    def __init__(self):
        self.silence = defaultdict(int)
        self.last_activity: dict[str, float] = {}
        self.weights = {
            "Judas": 0.8,
            "Yeshua": 0.6,
            "Peter": 0.7,
            "Mary": 0.2,
            "Jan": 0.5,
            "Thomas": 0.6,
            "Yakov": 0.4,
            "Andrew": 0.1,
            "Leo": 0.3,
            "Theodore": 0.1,
            "Dubrovsky": 0.05,
        }

    def pick(self, chat_id: str, chapter_text: str, user_text: str | None):
        self.last_activity[chat_id] = time.time()
        user_silent = not user_text or not user_text.strip()
        mode = "active"
        if user_silent:
            self.silence[chat_id] += 1
            mode = "chaos" if self.silence[chat_id] > 3 else "silent"
        else:
            self.silence[chat_id] = 0
        if re.search(r"\b(betrayal|knife|arrest|death)\b", chapter_text, re.I):
            mode = "tension"

        table = {
            "active": [2, 3],
            "silent": [2, 3, 5],
            "tension": [2, 3, 4],
            "chaos": [3, 4, 5, 6],
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


def cleanup_hero_cache(max_age_hours: int = 24):
    cutoff = time.time() - max_age_hours * 3600
    for p in hero_manager.cache_dir.glob("*.txt"):
        if p.stat().st_mtime < cutoff:
            with contextlib.suppress(Exception):
                p.unlink()
