import asyncio, logging
from collections import defaultdict
from urllib.parse import urlparse
# firecrawl-py 4.x: package imports as `firecrawl`; primary class is `Firecrawl`
# (the legacy `FirecrawlApp` alias exists but exposes no `scrape_url`/`scrape`
# methods on this version — we use `Firecrawl` directly).
from firecrawl import Firecrawl
from tavily import TavilyClient
from .settings import FIRECRAWL_API_KEY, TAVILY_API_KEY, TAVILY_MIN_SCORE

log = logging.getLogger(__name__)
_firecrawl = Firecrawl(api_key=FIRECRAWL_API_KEY)
_tavily = TavilyClient(api_key=TAVILY_API_KEY) if TAVILY_API_KEY else None


async def search_and_rank(queries: list[str], k: int = 8) -> list[str]:
    """Tavily fans out cheap, returns URLs above threshold.

    Tavily v0.7+ `search(query, max_results=k)` returns a dict shaped
    `{"results": [{"url": str, "score": float, "title": ..., "content": ...}, ...]}`.
    """
    if not _tavily:
        raise RuntimeError("TAVILY_API_KEY required for search fan-out")
    urls: dict[str, float] = {}
    for q in queries:
        resp = await asyncio.to_thread(_tavily.search, q, max_results=k)
        for r in resp.get("results", []):
            if r.get("score", 0) >= TAVILY_MIN_SCORE:
                urls[r["url"]] = max(urls.get(r["url"], 0), r["score"])
    return [u for u, _ in sorted(urls.items(), key=lambda kv: -kv[1])[:k]]


async def fetch_markdown(url: str, attempts: int = 3) -> str:
    """Firecrawl with exponential backoff. Raises after final attempt — node lets LangGraph checkpoint+resume.

    Firecrawl v4.x: method is `scrape(url, *, formats=[...])` (not `scrape_url(url, params=...)`),
    and the return value is a Pydantic `firecrawl.v2.types.Document` whose markdown is on the
    `.markdown` attribute (not `.get("markdown")`).
    """
    for i in range(attempts):
        try:
            doc = await asyncio.to_thread(
                _firecrawl.scrape, url, formats=["markdown"],
            )
            return getattr(doc, "markdown", "") or ""
        except Exception as e:
            log.warning("firecrawl %s attempt %d: %s", url, i + 1, e)
            if i == attempts - 1:
                raise
            await asyncio.sleep(2 ** i)
    return ""  # unreachable


# Per-domain stagger so we don't hammer one host.
# Module-global is intentional: Streamlit single-process app shares one event loop.
_domain_locks: dict[str, asyncio.Lock] = defaultdict(asyncio.Lock)


async def fetch_markdown_polite(url: str) -> str:
    host = urlparse(url).netloc.lower()  # case-fold so Arxiv.org and arxiv.org share one lock
    async with _domain_locks[host]:
        return await fetch_markdown(url)
