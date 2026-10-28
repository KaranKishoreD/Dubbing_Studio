"""
OpenRouter access for every LLM step (translation, director, length fitting,
transcript clean-up, song lyrics).

All calls go through `openrouter_client()`, which caps the model's hidden
"thinking" — it is billed as output and these are short text edits.
"""

import os

from openai import OpenAI

from .json_utils import extract_json_array

TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "anthropic/claude-sonnet-5.5")


def openrouter_client() -> OpenAI:
    """
    OpenRouter client with thinking capped on every call. Claude Sonnet 5.5
    thinks by default there, billed as output ($10/M): a 2-minute dub cost
    ~$0.70 instead of ~$0.11, nearly all hidden reasoning, for line edits
    that need none.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )
    create = client.chat.completions.create

    def no_thinking(**kw):
        # Sonnet 5.5 rejects "off" ("Reasoning is mandatory"), so cap it at
        # the smallest budget Anthropic allows: ≤ ~$0.01 of thinking per call.
        kw.setdefault("extra_body", {}).setdefault("reasoning", {"max_tokens": 1024})
        return create(**kw)

    client.chat.completions.create = no_thinking
    return client


def parse_items(text: str) -> list:
    """The JSON array in a model reply (see json_utils.extract_json_array)."""
    return extract_json_array(text)
