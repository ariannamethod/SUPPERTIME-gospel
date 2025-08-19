import os
import theatre

os.environ.setdefault("ASSISTANT_ID", "test")

def test_valid_scene():
    text = "**Judas**: hi\n**Peter**: ok"
    assert theatre.is_valid_scene(text, ["Judas", "Peter"])

def test_invalid_name():
    text = "**Satan**: hi"
    assert not theatre.is_valid_scene(text, ["Judas"])

def test_invalid_count():
    text = "**Judas**: hi"
    assert not theatre.is_valid_scene(text, ["Judas", "Peter"])
