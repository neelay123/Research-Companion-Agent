import asyncio
import pytest
from research_agent.web import search_and_rank, fetch_markdown_polite

pytestmark = pytest.mark.smoke


def test_tavily_returns_urls():
    async def go():
        urls = await search_and_rank(["mamba state space model"], k=3)
        assert len(urls) >= 1
        assert all(u.startswith("http") for u in urls)
    asyncio.run(go())


def test_firecrawl_returns_markdown():
    async def go():
        md = await fetch_markdown_polite("https://arxiv.org/abs/2312.00752")
        assert len(md) > 200
        assert "Mamba" in md or "state space" in md.lower()
    asyncio.run(go())
