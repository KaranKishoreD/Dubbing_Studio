"""
Whole-dialogue translation with an LLM (the "LLM" pipeline option, and the
fallback for lines Bodhan cannot translate).

The full conversation goes to the model in one request (chunked for long
videos) with each line's speaker, gender and time budget, so verb gender,
register and meaning stay consistent across lines.
"""

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..state import Segment
from .languages import _MIX_TEXT, _SCRIPT_LATIN, _SCRIPT_NATIVE, LANG_MIX, LANG_NAMES, script_ok
from .length_fit import time_budgets
from .llm import TRANSLATE_MODEL, openrouter_client, parse_items

_REGISTER = os.environ.get("TRANSLATE_REGISTER", "spoken").lower()


_CHUNK_LINES = int(os.environ.get("TRANSLATE_CHUNK_LINES", "40"))


_REGISTER_TEXT = {
    "spoken": (
        "natural, everyday SPOKEN {lang} — the way people actually talk, and the "
        "way a good professional {lang} dub sounds. Not formal written or literary "
        "{lang}. Keep this register identical on every line."
    ),
    "formal": (
        "clear, formal {lang} suitable for news or official content. Keep this "
        "register identical on every line."
    ),
    "neutral": (
        "neutral standard {lang}: correct and natural, neither slangy nor bookish. "
        "Keep this register identical on every line."
    ),
}


_DIALOGUE_SYSTEM = """\
You are an expert dubbing translator. You translate a video's dialogue from
English into {lang} for a voice-over that must sound natural and stay in sync.

The input is JSON with:
  "context": the FULL transcript, for understanding only.
  "lines":   the lines to translate. Each has id, speaker, gender, seconds,
             budget_chars, text.

Rules:
1. Register: {register}
2. Gender agreement: each line's "gender" is the SPEAKER'S gender. Use the
   matching verb endings, pronouns and self-references for that speaker.
3. The source comes from automatic speech recognition and contains mis-heard
   words (e.g. "selling you with everything" for "settling in with
   everything", "pair-wise" for "pay rise"). Some lines also carry
   "alt_text": an independent second transcription of the same audio. The
   two recognisers make different mistakes — compare them, use the context,
   and work out what was actually said. Translate THAT. Never translate
   obvious ASR nonsense literally.
4. Translate meaning, not words. Use idioms, fillers and interjections a
   native {lang} speaker would naturally use ("Yeah", "Oh really?", "You're
   kidding" all have natural spoken equivalents).
5. {mix}
   {script_rule}
6. Length is a HARD limit: budget_chars is how many {lang} characters the
   voice can speak in "seconds" at a natural pace (measured, not guessed).
   Never exceed it. {lang} often needs more characters than English for the
   same idea, so when a faithful translation runs long, say it the way a
   native speaker would say it briefly — shorter words, drop filler,
   restructure — instead of translating word for word. Keep the meaning
   that matters; coming in well under the budget is fine.
7. One output line per input line, same id. Do not merge, split, or move
   content between lines — each line plays at its own time.
{extra}
Return ONLY a JSON array, no markdown:
[{{"id": <int>, "text": "<{lang} line>"}}]
"""


def translate_dialogue(
    group: list[Segment],
    all_segments: list[Segment],
    target_lang: str,
    instruction: str | None,
    domain: str | None,
    mix: str = LANG_MIX,
    latin: bool = False,
) -> list[Segment]:
    """
    Translate `group` in context. Writes seg.dubbed_text in place and returns
    the segments that could not be translated (for the Bodhan fallback).
    """
    lang = LANG_NAMES.get(target_lang.split("-")[0], target_lang)
    ordered = sorted(all_segments, key=lambda s: s.start)
    context = "\n".join(
        f"[{s.id}] Speaker {s.speaker} ({s.gender or '?'}): {s.text}" for s in ordered
    )
    extra = ""
    if domain and domain.strip() and domain.lower() not in ("general", "auto"):
        extra += f"8. Content domain: {domain}. Use its natural terminology.\n"
    if instruction:
        extra += f"9. Style instruction for these lines, apply it exactly: {instruction}\n"
    system = _DIALOGUE_SYSTEM.format(
        lang=lang,
        register=_REGISTER_TEXT.get(_REGISTER, _REGISTER_TEXT["spoken"]).format(lang=lang),
        mix=_MIX_TEXT.get(mix, _MIX_TEXT["standard"]).format(lang=lang),
        script_rule=(_SCRIPT_LATIN if latin else _SCRIPT_NATIVE).format(lang=lang),
        extra=extra,
    )

    group = sorted(group, key=lambda s: s.start)
    chunks = [group[i : i + _CHUNK_LINES] for i in range(0, len(group), _CHUNK_LINES)]
    budgets = time_budgets(ordered, target_lang)

    def run(chunk: list[Segment]) -> dict[int, str]:
        lines = [
            {
                "id": s.id,
                "speaker": s.speaker,
                "gender": s.gender or "unknown",
                "seconds": budgets[s.id][0],
                "budget_chars": budgets[s.id][1],
                "text": s.text,
                **({"alt_text": s.asr_alt} if s.asr_alt and s.asr_alt != s.text else {}),
            }
            for s in chunk
        ]
        payload = json.dumps({"context": context, "lines": lines}, ensure_ascii=False)
        resp = openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": payload}],
            temperature=0.3,
            timeout=180,
        )
        items = parse_items(resp.choices[0].message.content.strip())
        wanted = {s.id for s in chunk}
        return {
            it["id"]: str(it["text"]).strip()
            for it in items
            if isinstance(it, dict) and it.get("id") in wanted and it.get("text")
        }

    done: dict[int, str] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(chunks))) as pool:
        futures = {pool.submit(run, c): c for c in chunks}
        for fut in as_completed(futures):
            try:
                done.update(fut.result())
            except Exception as exc:
                print(f"  ⚠ dialogue translation chunk failed: {exc}")

    # One retry, still with full context, for lines the model skipped or wrote
    # partly in the wrong script (it occasionally leaks Latin or even
    # katakana into Tamil, which the TTS voice then mangles).
    code = target_lang.split("-")[0]
    bad = [s for s in group if s.id not in done or not script_ok(done[s.id], code, latin)]
    if bad and len(bad) < len(group):
        try:
            done.update(run(bad))
        except Exception as exc:
            print(f"  ⚠ retry for {len(bad)} line(s) failed: {exc}")
    for s in group:
        if s.id in done and not script_ok(done[s.id], code, latin):
            del done[s.id]  # still wrong script → Bodhan fallback

    for s in group:
        if s.id in done:
            s.dubbed_text = done[s.id]
    return [s for s in group if s.id not in done]
