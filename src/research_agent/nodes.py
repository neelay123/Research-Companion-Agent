import asyncio
from google import genai
from google.genai import types
from google.genai import errors as gx
from .settings import GEMINI_API_KEY

_g_client = genai.Client(api_key=GEMINI_API_KEY)

# google-genai 1.x exposes ClientError (4xx) and ServerError (5xx) under
# google.genai.errors instead of google.api_core.exceptions. Transient codes
# we want to retry on: 429 (ResourceExhausted), 500 (Internal),
# 503 (Unavailable), 504 (DeadlineExceeded). We catch the broad classes and
# filter by .code so non-transient 4xx/5xx propagate immediately.
_RETRY_CODES = {429, 500, 503, 504}
_RETRY_EXC = (gx.ClientError, gx.ServerError, asyncio.TimeoutError)

async def _gen(model: str, contents, **cfg) -> object:
    """generate_content with backoff on transient errors."""
    for i in range(4):
        try:
            return await _g_client.aio.models.generate_content(
                model=model, contents=contents,
                config=types.GenerateContentConfig(**cfg),
            )
        except (gx.ClientError, gx.ServerError) as e:
            if getattr(e, "code", None) not in _RETRY_CODES or i == 3:
                raise
            await asyncio.sleep(2 ** i + 1)
        except asyncio.TimeoutError:
            if i == 3: raise
            await asyncio.sleep(2 ** i + 1)

def _parsed_or_raise(resp, schema_name: str):
    """response.parsed is None on parse failure even with response_schema."""
    if resp.parsed is None:
        raise ValueError(f"Gemini failed to produce valid {schema_name}; raw: {resp.text[:500]}")
    return resp.parsed
