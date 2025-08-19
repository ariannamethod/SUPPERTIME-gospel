import theatre


def test_markov_glitch_generates_text(monkeypatch):
    engine = theatre.MarkovEngine()
    engine.p = 1.0
    monkeypatch.setattr(theatre.random, "random", lambda: 0.0)
    monkeypatch.setattr(theatre.random, "choice", lambda seq: seq[0])
    monkeypatch.setattr(theatre.random, "randint", lambda a, b: 2)
    assert engine.glitch() == "*WHO ARE YOU*"


def test_markov_glitch_none_when_probability_low():
    engine = theatre.MarkovEngine()
    engine.p = 0.0
    assert engine.glitch() is None
