import os

import pytest

os.environ.setdefault("ASSISTANT_ID", "test")

from theatre import parse_lines


def test_parse_lines_double_star_inline():
    text = "**Judas**: hi\n**Peter**: bye"
    assert list(parse_lines(text)) == [("Judas", "hi"), ("Peter", "bye")]


def test_parse_lines_multiline_block_double_star():
    text = (
        "**Judas**\n"
        "line one\n"
        "line two\n"
        "**Mary**\n"
        "single line"
    )
    assert list(parse_lines(text)) == [
        ("Judas", "line one\nline two"),
        ("Mary", "single line"),
    ]


def test_parse_lines_inline_mixed_punctuation_and_spaces():
    text = "  **Judas**   - hi  \nPeter —  bye"
    assert list(parse_lines(text)) == [("Judas", "hi"), ("Peter", "bye")]


def test_parse_lines_markdown_heading_block():
    text = ("# Judas\n" "first\n" "## Mary  ##\n" "second")
    assert list(parse_lines(text)) == [("Judas", "first"), ("Mary", "second")]


def test_parse_lines_name_says_variants():
    text = "Judas says hello\nMary says: bye"
    assert list(parse_lines(text)) == [("Judas", "hello"), ("Mary", "bye")]


def test_parse_lines_bullet_markers_and_quotes():
    text = "- 'Judas': hi\n* \"Mary\" — bye\n• Peter says greetings"
    assert list(parse_lines(text)) == [
        ("Judas", "hi"),
        ("Mary", "bye"),
        ("Peter", "greetings"),
    ]


def test_parse_lines_name_with_spaces_and_quoted_answer():
    text = '**Mary Magdalene**: She said "Hello" loudly'
    assert list(parse_lines(text)) == [
        ("Mary Magdalene", 'She said "Hello" loudly')
    ]


def test_parse_lines_list_marker_with_space_in_name():
    text = "- Mary Magdalene says hello"
    assert list(parse_lines(text)) == [("Mary Magdalene", "hello")]


def test_parse_lines_multilingual_names():
    text = "**Мария**: привет\n**José**: hola\n**李华**: 你好"
    assert list(parse_lines(text)) == [
        ("Мария", "привет"),
        ("José", "hola"),
        ("李华", "你好"),
    ]


def test_parse_lines_malformed_unmatched_star():
    text = "*Judas**: hi"
    with pytest.raises(ValueError):
        list(parse_lines(text))


def test_parse_lines_malformed_unexpected_marker():
    text = "- hi there"
    with pytest.raises(ValueError):
        list(parse_lines(text))

