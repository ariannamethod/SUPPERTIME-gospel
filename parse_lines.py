"""Utilities for parsing speaker lines in theatre scripts."""

import re
from typing import Iterator, List, Tuple

# Building blocks for regex patterns
SEP = r"[:\-\–—]"  # allowed separators between name and text
BULLET = r"(?:[-*+•]\s+)?"  # optional bullet markers at line start
STAR = r"\*{1,2}"  # either * or ** surrounding names


def _normalize(name: str) -> str:
    """Trim surrounding punctuation and whitespace from a speaker name."""
    return name.strip(" \"'“”‘’*-—–•")


def _inline_star_pattern() -> re.Pattern[str]:
    """Match lines like ``**Name**: text`` or ``*Name* - text``."""
    return re.compile(
        rf"^{BULLET}(?P<stars>\*{{1,2}})\s*(?P<name>[^*]+?)\s*(?P=stars)\s*{SEP}\s*(?P<line>.*)"
    )


def _inline_plain_pattern() -> re.Pattern[str]:
    """Match lines like ``Name: text`` without star markers."""
    return re.compile(
        rf"^{BULLET}(?!{STAR})(?P<name>.+?)\s*{SEP}\s*(?P<line>.*)"
    )


def _inline_says_pattern() -> re.Pattern[str]:
    """Match lines where the separator is the word 'says'."""
    return re.compile(
        rf"^{BULLET}(?!{STAR})(?P<name>.+?)\s+says\s*(?:{SEP}\s*)?(?P<line>.*)",
        re.I,
    )


def _name_line_star_pattern() -> re.Pattern[str]:
    """Match block lines consisting solely of a name wrapped in star pairs."""
    return re.compile(rf"^{BULLET}(?P<stars>\*{{1,2}})\s*(?P<name>[^*]+?)\s*(?P=stars)$")


def _name_line_heading_pattern() -> re.Pattern[str]:
    """Match markdown heading lines like ``# Name``."""
    return re.compile(rf"^\s{{0,3}}{BULLET}#{{1,6}}\s*(?P<name>.+?)\s*#*\s*$")


INLINE_PATTERNS: List[re.Pattern[str]] = [
    _inline_star_pattern(),
    _inline_says_pattern(),
    _inline_plain_pattern(),
]

NAME_PATTERNS: List[re.Pattern[str]] = [
    _name_line_star_pattern(),
    _name_line_heading_pattern(),
]


def _match_first(patterns: List[re.Pattern[str]], text: str) -> re.Match[str] | None:
    for pat in patterns:
        m = pat.match(text)
        if m:
            return m
    return None


def _check_unexpected_markers(line: str) -> None:
    """Raise ``ValueError`` if the line contains unmatched or stray markers."""
    if line.count("*") % 2 == 1:
        raise ValueError(f"unmatched '*' in line: {line}")
    if re.search(rf"{SEP}", line) or re.match(r"^[-*+•]", line):
        raise ValueError(f"unexpected marker in line: {line}")


def parse_lines(text: str) -> Iterator[Tuple[str, str]]:
    """Yield ``(name, line)`` pairs from ``text``.

    Supports inline formats (e.g. ``**Name**: hi``) and block formats where
    a name line is followed by one or more content lines. Raises ``ValueError``
    when encountering unmatched markers or malformed lines.
    """
    current_name: str | None = None
    buffer: List[str] = []
    for raw in text.splitlines():
        line = raw.rstrip()
        stripped = line.strip().strip("\"'“”‘’")
        m_inline = _match_first(INLINE_PATTERNS, stripped)
        if m_inline:
            if current_name and buffer:
                yield current_name, "\n".join(buffer).strip()
            yield _normalize(m_inline.group("name")), m_inline.group("line")
            current_name, buffer = None, []
            continue
        m_name = _match_first(NAME_PATTERNS, stripped)
        if m_name:
            if current_name and buffer:
                yield current_name, "\n".join(buffer).strip()
            current_name = _normalize(m_name.group("name"))
            buffer = []
        elif current_name is not None:
            buffer.append(line)
        else:
            if stripped:
                _check_unexpected_markers(stripped)
    if current_name and buffer:
        yield current_name, "\n".join(buffer).strip()
