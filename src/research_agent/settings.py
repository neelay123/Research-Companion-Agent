import os, tomllib
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY    = os.environ["GEMINI_API_KEY"]
FIRECRAWL_API_KEY = os.environ["FIRECRAWL_API_KEY"]
TAVILY_API_KEY    = os.environ.get("TAVILY_API_KEY")
SALIENCE_CUTOFF   = float(os.environ.get("SALIENCE_CUTOFF", "0.5"))
TAVILY_MIN_SCORE  = float(os.environ.get("TAVILY_MIN_SCORE", "0.5"))

_CFG_PATH = Path(__file__).resolve().parent.parent.parent / "config.toml"
with open(_CFG_PATH, "rb") as f:
    _cfg = tomllib.load(f)
TOPICS: list[str] = _cfg.get("research", {}).get("topics", [])
