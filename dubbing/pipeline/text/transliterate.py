"""
English words → target-language script, for TTS voices that cannot read
Latin letters (one batched LLM call).
"""

import json
import os

from .languages import LANG_NAMES, script_ok
from .llm import openrouter_client, parse_items

_TRANSLIT_SYSTEM = """\
Rewrite each line so every word in Latin letters is written in {lang} script,
spelled the way a {lang} speaker pronounces it ("office" → the {lang}-script
spelling of office). Change nothing else — same words, same order.
Return ONLY JSON: [{{"id": <int>, "text": "<line>"}}]
"""


def transliterate_latin(items: list[tuple[int, str]], lang_code: str) -> dict[int, str]:
    """{id: text with Latin words in native script} — for voices that cannot
    read Latin (Bodhan). Lines that fail keep their text."""
    if not items or not os.environ.get("OPENROUTER_API_KEY"):
        return {}
    lang = LANG_NAMES.get(lang_code, lang_code)
    try:
        resp = openrouter_client().chat.completions.create(
            model=os.environ.get("TRANSLIT_MODEL", "google/gemini-3.8-flash"),
            messages=[
                {"role": "system", "content": _TRANSLIT_SYSTEM.format(lang=lang)},
                {
                    "role": "user",
                    "content": json.dumps(
                        [{"id": i, "text": t} for i, t in items], ensure_ascii=False
                    ),
                },
            ],
            temperature=0.0,
            timeout=60,
        )
        out = parse_items(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"  ⚠ transliteration skipped: {exc}")
        return {}
    wanted = dict(items)
    return {
        it["id"]: str(it["text"]).strip()
        for it in out
        if isinstance(it, dict)
        and it.get("id") in wanted
        and it.get("text")
        and script_ok(str(it["text"]), lang_code)
    }
