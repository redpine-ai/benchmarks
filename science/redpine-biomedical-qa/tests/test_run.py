import json

import pytest

import src.run
from src.data import load_questions
from src.run import main, parse_args
from harness.config import load_config


def test_load_questions(tmp_path):
    p = tmp_path / "q.jsonl"
    p.write_text(json.dumps({"id": "a", "track": "cardiology", "question": "Q?", "gold": "G"}) + "\n\n")
    assert load_questions(p) == [{"id": "a", "track": "cardiology", "question": "Q?", "gold": "G"}]


def test_missing_question_set_exits(tmp_path):
    with pytest.raises(SystemExit, match="question set not found"):
        load_questions(tmp_path / "none.jsonl")


def test_parse_args_defaults():
    cfg = load_config(src.run.CONFIG)
    a = parse_args([], cfg)
    assert a.arms == ["websearch", "connect"] and a.provider == "anthropic" and a.workers == 4


def test_main_stops_before_download_without_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(src.run, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(src.run, "load_questions", lambda *a, **k: pytest.fail("loaded data"))
    with pytest.raises(SystemExit, match="ANTHROPIC_API_KEY"):
        main(["--arms", "websearch"])


def test_parse_args_config_default_and_override():
    cfg = load_config(src.run.CONFIG)
    assert parse_args([], cfg).config == str(src.run.CONFIG)
    assert parse_args(["--config", "my/arms.toml"], cfg).config == "my/arms.toml"


def test_main_loads_config_from_flag(monkeypatch, tmp_path):
    seen = {}

    def fake_load(path):
        seen["path"] = path
        raise SystemExit("captured")

    monkeypatch.setattr(src.run, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setattr(src.run, "load_config", fake_load)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit, match="captured"):
        main(["--config", "my/arms.toml"])
    assert seen["path"] == (tmp_path / "my/arms.toml").resolve()
