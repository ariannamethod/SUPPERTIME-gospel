from monolith import parse_lines


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
        ("Judas", "line one"),
        ("Judas", "line two"),
        ("Mary", "single line"),
    ]


def test_parse_lines_skips_blank_lines():
    text = (
        "**Judas**\n"
        "first line\n"
        "\n"
        "second line"
    )
    assert list(parse_lines(text)) == [
        ("Judas", "first line"),
        ("Judas", "second line"),
    ]

