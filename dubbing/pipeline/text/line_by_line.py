"""
Line-by-line translation (the Indic pipeline's first pass).

Bodhan indic-translate translates each line on its own, many in parallel.
Lines with a tone edit ("make it funny") go to the LLM instead, which can
follow a style instruction.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

from ..state import Segment
from .llm import TRANSLATE_MODEL, openrouter_client, parse_items

_TONE_KEYWORDS = {
    "funny",
    "formal",
    "casual",
    "energetic",
    "slow",
    "sad",
    "happy",
    "angry",
    "excited",
    "sarcastic",
    "serious",
    "dramatic",
    "simpler",
    "clearer",
    "punchier",
    "emotional",
    "enthusiastic",
    "tone",
    "style",
    "accent",
    "british",
    "american",
    "mumbai",
    "delhi",
}


_OPENROUTER_SYSTEM = """\
You are dubbing a video transcript into {target_lang}.
The source may contain code-switching (Hinglish, Tanglish, etc.).

Rules:
1. Translate non-English parts to {target_lang}
2. Keep domain technical terms as-is (API, gradient, algorithm, etc.)
3. Apply the style instruction exactly: {instruction}
4. Output must be clean, speakable {target_lang} — no mid-sentence language mixing
5. Match the original sentence length — avoid padding that causes lip-sync drift

Return ONLY valid JSON (no markdown):
[{{"id": <int>, "dubbed_text": "<translated text>"}}]
"""


def bodhan_client() -> OpenAI:
    from ..bodhan_gate import bodhan_client as gated_client

    return gated_client("BODHAN_TRANSLATE_KEY")


def needs_style_llm(instruction: str) -> bool:
    low = instruction.lower()
    return any(kw in low for kw in _TONE_KEYWORDS)


def translate_with_bodhan(
    group: list[Segment], target_lang: str, seg_map: dict[int, Segment]
) -> list[Segment]:
    """Fire all Bodhan translation requests simultaneously; returns the lines that failed."""
    client = bodhan_client()

    def _translate_one(seg: Segment) -> tuple[int, str]:
        from ..bodhan_gate import TRANSLATE

        # Through the shared translate gate: a rate-limit reply is waited
        # out; otherwise the line would fall back to the paid LLM.
        resp = TRANSLATE.call(
            client.chat.completions.create,
            model="indic-translate",
            messages=[{"role": "user", "content": seg.text}],
            extra_body={"target_language_code": target_lang, "target_script": "native"},
        )
        return seg.id, resp.choices[0].message.content.strip()

    failed: list[Segment] = []
    with ThreadPoolExecutor(max_workers=min(len(group), 10)) as pool:
        futures = {pool.submit(_translate_one, seg): seg for seg in group}
        for future in as_completed(futures):
            try:
                seg_id, text = future.result()
            except Exception as exc:
                print(f"  ⚠ Bodhan translation failed for line {futures[future].id}: {exc}")
                failed.append(futures[future])
                continue
            if text:
                seg_map[seg_id].dubbed_text = text
            else:
                failed.append(seg_map[seg_id])
    return failed


def translate_style_edit(
    group: list[Segment],
    target_lang: str,
    instruction: str,
    seg_map: dict[int, Segment],
) -> None:
    """Translate + apply tone via OpenRouter (single call for the whole group)."""
    client = openrouter_client()

    transcript_text = "\n".join(
        f"[S{s.id}] {s.start:.1f}s–{s.end:.1f}s [{s.lang}]: {s.text}" for s in group
    )

    resp = client.chat.completions.create(
        model=TRANSLATE_MODEL,
        messages=[
            {
                "role": "system",
                "content": _OPENROUTER_SYSTEM.format(
                    target_lang=target_lang,
                    instruction=instruction,
                ),
            },
            {"role": "user", "content": transcript_text},
        ],
        temperature=0.4,
    )

    items = parse_items(resp.choices[0].message.content.strip())
    for i, t in enumerate(items):
        # LLM sometimes re-indexes IDs; fall back to group position
        seg_id = t["id"] if t["id"] in seg_map else (group[i].id if i < len(group) else None)
        if seg_id is not None:
            seg_map[seg_id].dubbed_text = t["dubbed_text"]
