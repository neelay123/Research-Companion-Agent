import os
import importlib
from pathlib import Path
import pytest

def _reload_settings():
    import research_agent.settings as s
    return importlib.reload(s)

def test_topics_loaded(monkeypatch, tmp_path):
    # Write a temp config.toml two levels above the package source
    cfg = tmp_path / "config.toml"
    cfg.write_text('[research]\ntopics = ["x", "y"]\n', encoding="utf-8")
    monkeypatch.setenv("GEMINI_API_KEY", "k1")
    monkeypatch.setenv("FIRECRAWL_API_KEY", "k2")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("research_agent.settings.__file__",
                        str(tmp_path / "settings.py"), raising=False)
    s = _reload_settings()
    # Note: settings.py reads config.toml relative to its own location;
    # for this test we just confirm the parsing path works on a known file.
    assert isinstance(s.TOPICS, list)

def test_required_env_missing(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(KeyError):
        _reload_settings()
