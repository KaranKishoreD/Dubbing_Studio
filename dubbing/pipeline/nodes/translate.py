"""
Translation node — all segments translated in parallel via ThreadPoolExecutor.

Routing:
  - Standard segments  → Bodhan indic-translate  (₹0.20/10K output tokens)
  - Tone-edit segments → OpenRouter Gemini Flash  (tone control not in indic-translate)
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from ..state import PipelineState, Segment
from ..cache import get_cache_dir

_TONE_KEYWORDS = {
    "funny", "formal", "casual", "energetic", "slow", "sad", "happy",
    "angry", "excited", "sarcastic", "serious", "dramatic", "simpler",
    "clearer", "punchier", "emotional", "enthusiastic", "tone", "style",
    "accent", "british", "american", "mumbai", "delhi",
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


def _bodhan_client() -> OpenAI:
    return OpenAI(
        base_url="https://api.bodhan.ai/v1",
        api_key=os.environ["BODHAN_TRANSLATE_KEY"],
    )


def _openrouter_client() -> OpenAI:
    return OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )


def _needs_openrouter(instruction: str) -> bool:
    low = instruction.lower()
    return any(kw in low for kw in _TONE_KEYWORDS)


def _strip_json(text: str) -> str:
    if text.startswith("```"):
        return "\n".join(l for l in text.split("\n") if not l.startswith("```")).strip()
    return text


def translate_node(state: PipelineState) -> dict:
    segments = [Segment.from_dict(s) for s in state["segments"]]
    edit_ops = state.get("edit_ops", [])

    edit_map: dict[int, str] = {}
    for op in edit_ops:
        for sid in op["segment_ids"]:
            edit_map[sid] = op["instruction"]

    for seg in segments:
        if seg.id in edit_map:
            seg.edit_instruction = edit_map[seg.id]
            seg.dubbed_text = None

    bodhan_group: list[Segment] = []
    openrouter_groups: dict[str, list[Segment]] = {}

    for seg in segments:
        if seg.dubbed_text is not None:
            continue
        instruction = seg.edit_instruction or "default"
        if instruction == "default" or not _needs_openrouter(instruction):
            bodhan_group.append(seg)
        else:
            openrouter_groups.setdefault(instruction, []).append(seg)

    if not bodhan_group and not openrouter_groups:
        return {"log": state.get("log", []) + ["✓ Translation: all segments cached"]}

    seg_map = {s.id: s for s in segments}
    total = 0

    if bodhan_group:
        _translate_bodhan_parallel(bodhan_group, state["target_lang"], seg_map)
        total += len(bodhan_group)

    for instruction, group in openrouter_groups.items():
        _translate_openrouter(group, state["target_lang"], instruction, seg_map)
        total += len(group)

    cache_dir = get_cache_dir(state["video_id"])
    with open(cache_dir / "translations.json", "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, ensure_ascii=False, indent=2)

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + [
            f"✓ Translated {total} segments in parallel "
            f"(Bodhan: {len(bodhan_group)}, OpenRouter: {sum(len(g) for g in openrouter_groups.values())})"
        ],
    }


def _translate_bodhan_parallel(
    group: list[Segment], target_lang: str, seg_map: dict[int, Segment]
) -> None:
    """Fire all Bodhan translation requests simultaneously."""
    client = _bodhan_client()

    def _translate_one(seg: Segment) -> tuple[int, str]:
        resp = client.chat.completions.create(
            model="indic-translate",
            messages=[{"role": "user", "content": seg.text}],
            extra_body={
                "target_language_code": target_lang,
                "target_script": "native",
            },
        )
        return seg.id, resp.choices[0].message.content.strip()

    with ThreadPoolExecutor(max_workers=min(len(group), 10)) as pool:
        futures = {pool.submit(_translate_one, seg): seg for seg in group}
        for future in as_completed(futures):
            seg_id, text = future.result()
            seg_map[seg_id].dubbed_text = text


def _translate_openrouter(
    group: list[Segment],
    target_lang: str,
    instruction: str,
    seg_map: dict[int, Segment],
) -> None:
    """Translate + apply tone via OpenRouter (single call for the whole group)."""
    client = _openrouter_client()

    transcript_text = "\n".join(
        f"[S{s.id}] {s.start:.1f}s–{s.end:.1f}s [{s.lang}]: {s.text}"
        for s in group
    )

    resp = client.chat.completions.create(
        model="openai/gpt-4o-mini",
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

    raw = _strip_json(resp.choices[0].message.content.strip())
    items = json.loads(raw)
    for i, t in enumerate(items):
        # LLM sometimes re-indexes IDs; fall back to group position
        seg_id = t["id"] if t["id"] in seg_map else (group[i].id if i < len(group) else None)
        if seg_id is not None:
            seg_map[seg_id].dubbed_text = t["dubbed_text"]
