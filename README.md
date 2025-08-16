# Suppertime Gospel

Suppertime Gospel is a Telegram bot that stages interactive gospel scenes using OpenAI's Assistants API.  Chapters of the narrative live in `docs/` and each character's persona lives in `heroes/`.  The bot lets you drop into any chapter and guide the conversation.

## Environment Variables
Set the following variables before running the bot:

- `TELEGRAM_TOKEN` – Telegram bot token (required)
- `OPENAI_API_KEY` – OpenAI API key (required)
- `OPENAI_MODEL` – OpenAI model name (optional, defaults to `gpt-4.1-mini`)
- `ST_DB` – path to the SQLite database (optional)

## Running Locally
1. Install dependencies: `pip install -r requirements.txt`
2. Export the required environment variables
3. Launch the bot: `python monolith.py`

## Quick Start
```bash
pip install -r requirements.txt
export TELEGRAM_TOKEN="123:ABC"
export OPENAI_API_KEY="sk-yourkey"
python monolith.py
```

## Editing Chapters
Chapter files are Markdown documents in the `docs/` directory named `chapter_XX.md` (two-digit numbers).  Edit an existing file or add a new one, then send `/reload` to the bot in Telegram to pick up changes.

## Editing Hero Prompts
Hero persona prompts are stored in the `heroes/` directory as `.prompt` files.  Each file should contain the sections `NAME`, `VOICE`, `BEHAVIOR`, `INIT`, and `REPLY`.  After modifying or adding files, send `/reload_heroes` to the bot to reload them.
