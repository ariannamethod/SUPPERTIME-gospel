import os
import pytest

# Prevent network calls during import
os.environ.setdefault("ASSISTANT_ID", "test")

from monolith import guess_participants, parse_prompt_sections


def test_guess_participants_header():
    chapter = "Participants: Judas, Mary, Jan"
    assert guess_participants(chapter) == ["Judas", "Mary", "Jan"]


def test_guess_participants_regex_detection():
    chapter = "Mary spoke with the Teacher while Peter listened."
    assert guess_participants(chapter) == ["Yeshua", "Peter", "Mary"]


def test_guess_participants_default():
    chapter = "No known names here."
    assert guess_participants(chapter) == [
        "Judas", "Yeshua", "Peter", "Mary", "Jan", "Thomas"
    ]


def test_parse_prompt_sections_basic():
    text = (
        "NAME: Yeshua\n"
        "VOICE: gentle\n"
        "BEHAVIOR:\n"
        "kind and patient\n"
        "INIT: Hello\n"
        "REPLY:\n"
        "first line\n"
        "second line\n"
    )
    sections = parse_prompt_sections(text)
    assert sections["NAME"] == "Yeshua"
    assert sections["VOICE"] == "gentle"
    assert sections["BEHAVIOR"] == "kind and patient"
    assert sections["INIT"] == "Hello"
    assert sections["REPLY"] == "first line\nsecond line"
