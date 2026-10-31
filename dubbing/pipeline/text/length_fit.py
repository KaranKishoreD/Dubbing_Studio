"""
Making each translated line fit the time the speaker talks.

  time_budgets()      how many characters each line may hold
  shorten_overlong()  rewrite lines that are clearly too long, before speech
  condense_to_budget  shorten lines the voice really spoke too long (refit)
  expand_to_fill      lengthen lines that end well before the speaker stops
"""

import json
import os

from ..state import Segment
from .languages import LANG_NAMES, script_ok
from .llm import TRANSLATE_MODEL, openrouter_client, parse_items

# Only rewrite a segment if it overruns its slot by more than this factor —
# below it, align_node absorbs the difference without an audible artifact.
# Deliberately loose: this pre-synthesis check uses a rough chars/sec guess,
# so it only catches egregious overruns. synthesize_node re-checks with the
# real clip lengths and shortens precisely.
_OVERRUN_TRIGGER = float(os.environ.get("OVERRUN_TRIGGER", "1.15"))


# A dub shorter than this share of its time budget leaves the speaker's lips
# moving in silence, so it is lengthened.
_MIN_FILL = float(os.environ.get("MIN_FILL", "0.85"))


_BUDGET_BORROW = float(os.environ.get("BUDGET_BORROW_SECS", "0.3"))


def time_budgets(ordered: list[Segment], target_lang: str) -> dict[int, tuple[float, int]]:
    """
    {id: (seconds, budget_chars)} per line. Seconds is the line's real slot —
    up to the next line, borrowing trailing silence — and characters come from
    the voice's measured speaking rate in this language.
    """
    from ..speech.pace import speaker_factors
    from ..steps.align import slot_seconds
    from ..voice.tts_rate import chars_per_sec

    cps = chars_per_sec(target_lang.split("-")[0])
    # A fast talker's line holds more words than a slow lecturer's of the
    # same length; budgeting both at the voice's average made one rushed
    # and the other drawn out.
    tempo = speaker_factors(ordered)
    out = {}
    for i, s in enumerate(ordered):
        # Borrowing silence is a safety margin for the fitter, not part of the
        # target: a line written to fill 3 s of borrowed pause keeps talking
        # long after the speaker's mouth has stopped.
        secs = min(slot_seconds(ordered, i, None), s.duration + _BUDGET_BORROW)
        f = tempo.get(s.speaker if s.speaker is not None else 0, 1.0)
        out[s.id] = (round(secs, 2), max(4, int(secs * cps * f * 0.95)))
    return out


_COMPRESS_SYSTEM = """\
You shorten dubbed subtitle lines so they can be spoken aloud in a fixed time.

For each item you receive an id, the text, and a character budget.
Rewrite the text to fit within its budget while preserving meaning.

Rules:
1. Stay in the SAME language and script as the input. Do not translate.
2. Preserve the meaning of EVERY clause — feelings and whom/what they are
   about, contrast and concession ("though", "still", "but"), negation.
   Drop only filler, hedges and repetition; never a clause.
3. Keep technical terms, names and numbers intact.
4. The result must read as natural, speakable prose — not clipped notes.
5. Keep the same register (spoken vs formal), the same form of address
   (informal stays informal), the same gendered verb endings and pronouns,
   and any everyday English words, exactly as the input line has them.
   Never swap in formal, ceremonial, literary or archaic words to save
   characters — a casual line stays casual even if it ends up a little long.
6. Never exceed the budget. Coming in under it is fine.

Return ONLY valid JSON (no markdown):
[{"id": <int>, "text": "<shortened text>"}]
"""


def shorten_overlong(
    segments: list[Segment], target_lang: str, skip: set[int] | None = None
) -> int:
    """
    Rewrite segments whose text is too long to speak in their slot.
    Returns the number actually shortened. No-ops without an OpenRouter key.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        return 0

    budgets = time_budgets(sorted(segments, key=lambda s: s.start), target_lang)
    jobs: list[tuple[Segment, int]] = []
    for seg in segments:
        if not seg.dubbed_text or seg.id in (skip or ()) or seg.keep_original:
            continue
        budget = budgets[seg.id][1]
        if budget > 0 and len(seg.dubbed_text) > budget * _OVERRUN_TRIGGER:
            jobs.append((seg, budget))

    if not jobs:
        return 0

    new = condense_to_budget([(seg.id, seg.dubbed_text, budget) for seg, budget in jobs])
    seg_map = {seg.id: seg for seg, _ in jobs}
    for sid, text in new.items():
        seg_map[sid].dubbed_text = text
    return len(new)


_EXPAND_SYSTEM = """\
You are a {lang} dubbing writer. Each dubbed line below ends well before the
speaker in the video stops talking, so their lips move in silence. Rewrite
each dub so it naturally fills between min_chars and max_chars.

How:
- Say the original line more completely and the way a native speaker
  really talks: full sentences instead of clipped ones, the connecting
  words and small spoken particles that sound natural in context, details
  the original line itself contains but the dub left out.
- Same meaning, same tone and register, same form of address and gender
  forms as the current dub; same language and script.
Never:
- add facts, opinions or anything the original does not say;
- repeat words or phrases, stack fillers, or pad with "so, so", "you see",
  "basically" and the like;
- make it sound more formal, literary or textbook-like.
If a line cannot be lengthened naturally, return it unchanged.

Return ONLY JSON: [{{"id": <int>, "text": "<dub line>"}}]
"""


def expand_to_fill(items: list[dict], lang_code: str) -> dict[int, str]:
    """
    {id: longer dub} for lines that end well before the speaker stops.
    items: {"id", "original", "dub", "min_chars", "max_chars"}. One batched
    call; a rewrite is kept only if it is longer and within max_chars.
    """
    if not items or not os.environ.get("OPENROUTER_API_KEY"):
        return {}
    lang = LANG_NAMES.get(lang_code.split("-")[0], lang_code)
    try:
        resp = openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content": _EXPAND_SYSTEM.format(lang=lang)},
                {"role": "user", "content": json.dumps(items, ensure_ascii=False)},
            ],
            temperature=0.3,
        )
        out = parse_items(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"  ⚠ lengthening skipped: {exc}")
        return {}
    want = {it["id"]: it for it in items}
    result = {}
    for it in out:
        if not isinstance(it, dict) or it.get("id") not in want:
            continue
        text, w = str(it.get("text") or "").strip(), want[it["id"]]
        if (
            text
            and len(w["dub"]) < len(text) <= w["max_chars"] * 1.1
            and script_ok(text, lang_code.split("-")[0], latin=True)
        ):
            result[it["id"]] = text
    return result


def condense_to_budget(jobs: list[tuple[int, str, int]]) -> dict[int, str]:
    """
    Shorten each (id, text, budget_chars) with one batched LLM call.
    Returns {id: shorter_text} for the ones that genuinely got shorter.
    Empty dict if no OpenRouter key or the call fails.
    """
    if not jobs or not os.environ.get("OPENROUTER_API_KEY"):
        return {}
    payload = [{"id": i, "text": t, "budget_chars": b} for i, t, b in jobs]
    try:
        resp = openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        items = parse_items(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"  ⚠ Length compression skipped: {exc}")
        return {}

    originals = {i: t for i, t, _ in jobs}
    order = [i for i, _, _ in jobs]
    out: dict[int, str] = {}
    for k, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        sid = item.get("id")
        if sid not in originals:
            sid = order[k] if k < len(order) else None
        text = (item.get("text") or "").strip()
        # Only accept a genuine shortening; a longer rewrite defeats the purpose.
        if sid is not None and text and len(text) < len(originals[sid]):
            out[sid] = text
    return out
