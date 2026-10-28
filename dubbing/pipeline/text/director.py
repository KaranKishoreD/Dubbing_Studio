"""
The dubbing director: one LLM pass over the whole dialogue after the
line-by-line translation.

Bodhan translates each line without knowing who speaks to whom. The
director rewrites lines so that gender agreement, spoken register, form of
address, humour and timing are right for the conversation as a whole, and
splits lines that actually contain two speakers.
"""

import json
import os

from ..cache import get_cache_dir
from ..state import Segment
from .languages import LANG_NAMES, script_ok
from .length_fit import _MIN_FILL, _OVERRUN_TRIGGER, time_budgets
from .llm import TRANSLATE_MODEL, openrouter_client, parse_items

_DIRECTOR_SYSTEM = """\
You are a professional {lang} dubbing director and localization specialist.
You get a video's dialogue: per line the speaker, the speaker's gender, the
original line, the current {lang} dub (a line-by-line machine translation
that knows nothing about who speaks), and a character budget that fits the
time the speaker's mouth moves. Rewrite every dub line so the whole
conversation sounds like real people talking in {lang}.

First read the WHOLE dialogue and decide its tone: who these people are
to each other, how casual or formal they are, and how people like them
really talk in {lang} today. Every line — especially greetings, thanks,
goodbyes and short replies — must sound like those people in that
setting. Ceremonial, traditional or textbook phrases are wrong in casual
talk even when they are short; a casual phrase spoken a little faster is
always better than a formal one that fits.

Rules:
1. GENDER (critical). Every first-person verb, adjective and past-tense
   form must agree with the SPEAKER's gender; forms that refer to the
   person addressed must agree with the OTHER speaker's gender.
2. REGISTER. Modern, colloquial, spoken {lang} as people say it today in
   the setting of the video — never literary, archaic, bookish or overly
   Sanskritized/formal. Greetings, exclamations and small talk must be what
   a native speaker would actually say (not a word-for-word calque).
   Everyday English words that {lang} speakers normally use (office words,
   tech terms, "weekend", "meeting") may stay English.
3. ADDRESS CONSISTENCY. Pick ONE form of address between each pair of
   speakers that fits their relationship (peers / colleagues → informal;
   strangers, elders, an audience → respectful) and keep it identical in
   every line. Never mix respectful and informal pronouns for the same
   person.
4. TIMING. The speaker's mouth moves for the whole line, so the dub must
   FILL it: between min_chars and budget_chars, roughly the original's
   syllable count. Too short leaves silent moving lips — then use fuller,
   natural phrasing (complete sentences, the connectives and small words
   people really say), never new facts. Up to ~15% over budget is fine
   (spoken slightly faster); never trade tone or meaning for length.
5. MEANING FIRST. Every clause of the original must be recognisable in
   the dub: who does what, feelings AND what/whom they are about, and how
   the parts relate (cause, contrast, concession). Never drop
   negation, questions, contrast or concession words, or numbers. Keep
   names as they are. If the budget is tight, cut filler words and
   repetition — never a clause; a little over budget beats lost meaning.
6. Same id, one output per input line, in {lang} script.
9. TRANSCREATE, DON'T TRANSLATE. Jokes, sarcasm, slang, punch lines and
   catchphrases must land in {lang} the way they land in the original:
   use the {lang} idiom, slang or comic phrasing a native writer would
   choose for the same effect and timing, never a word-for-word rendering
   that leaves the joke flat or literal. Use the video title (genre: film
   trailer, comedy, lecture…) to judge the tone.
8. SPEAKABLE. A voice reads the dub aloud. Write formulas, symbols,
   variables, powers, sets, units and abbreviations the way the speaker
   would SAY them in {lang} (as words), never as notation a voice cannot
   read.
7. TWO SPEAKERS IN ONE LINE. The speaker labels come from voice analysis,
   which misses quick replies: a line may really be a short exchange
   between two people. Only when the words clearly belong to
   two people (a reply to what was just said, a returned greeting), give
   that line as "turns" instead of "text": consecutive pieces, each with
   the speaker who says it, its exact original words (copied from
   "original", in order, together covering the whole line) and its dub.
   Otherwise never use "turns".

Return ONLY JSON:
[{{"id": <int>, "text": "<dub line>"}}
 or {{"id": <int>, "turns": [{{"speaker": "Speaker N", "original": "<words>", "text": "<dub>"}}, ...]}}]
"""


def video_context(state) -> str:
    """Title and channel of the video, so the director knows the genre."""
    try:
        meta = json.loads(
            (get_cache_dir(state["video_id"]) / "meta.json").read_text(encoding="utf-8")
        )
        return " — ".join(x for x in (meta.get("title"), meta.get("channel")) if x)
    except Exception:
        return ""


def direct_dub(segments: list[Segment], target_lang: str, context: str = "") -> int:
    """
    One batched pass over the whole dialogue: gender agreement, colloquial
    register, consistent address, and length within each line's time
    budget. Returns how many lines changed. Lines whose rewrite comes back
    longer than budget keep the previous text.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        return 0, 0
    ordered = sorted(
        [s for s in segments if s.dubbed_text and not s.keep_original], key=lambda s: s.start
    )
    if not ordered:
        return 0, 0
    budgets = time_budgets(sorted(segments, key=lambda s: s.start), target_lang)
    payload = [
        {
            "id": s.id,
            "speaker": f"Speaker {(s.speaker or 0) + 1}",
            "gender": s.gender or "unknown",
            "original": s.text,
            "dub": s.dubbed_text,
            "min_chars": int(budgets[s.id][1] * _MIN_FILL),
            "budget_chars": budgets[s.id][1],
        }
        for s in ordered
    ]
    lang = LANG_NAMES.get(target_lang.split("-")[0], target_lang)
    try:
        resp = openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content": _DIRECTOR_SYSTEM.format(lang=lang)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"video": context, "lines": payload} if context else payload,
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0.2,
        )
        items = parse_items(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"  ⚠ dubbing-director pass skipped: {exc}")
        return 0, 0
    seg_map = {s.id: s for s in ordered}
    changed = split = 0
    for it in items:
        if not isinstance(it, dict) or it.get("id") not in seg_map:
            continue
        if isinstance(it.get("turns"), list) and len(it["turns"]) >= 2:
            pieces = _split_turns(seg_map[it["id"]], it["turns"], segments)
            if pieces:
                segments.extend(pieces)
                split += 1
                changed += 1
                continue
        s, text = seg_map[it["id"]], str(it.get("text") or "").strip()
        if (
            not text
            or text == s.dubbed_text
            or not script_ok(text, target_lang.split("-")[0], latin=True)
        ):
            continue
        # A rewrite may grow a too-short line, but never past its budget
        # unless it is still shorter than what it replaces.
        if len(text) > budgets[s.id][1] * _OVERRUN_TRIGGER and len(text) >= len(s.dubbed_text):
            continue
        s.dubbed_text = text
        changed += 1
    return changed, split


def _split_turns(seg: Segment, turns: list, segments: list[Segment]) -> list[Segment]:
    """
    Split `seg` into the speaker turns the director found, at Whisper's word
    timestamps. `seg` becomes the first turn; the new segments for the rest
    are returned. Nothing changes unless every turn maps onto the words.
    """
    import re

    norm = lambda x: re.sub(r"[^a-z0-9\u0900-\u0dff]", "", x.lower())
    words = [w for w in (seg.words or []) if norm(w.get("word", ""))]
    speakers = {s.speaker for s in segments if s.speaker is not None}
    gender = {}
    for s in segments:
        if s.speaker is not None and s.gender:
            gender.setdefault(s.speaker, s.gender)
    parsed = []
    for t in turns:
        if not isinstance(t, dict):
            return []
        m = re.search(r"(\d+)", str(t.get("speaker", "")))
        spk = int(m.group(1)) - 1 if m else None
        toks = [norm(x) for x in str(t.get("original", "")).split() if norm(x)]
        text = str(t.get("text") or "").strip()
        if spk not in speakers or not toks or not text:
            return []
        parsed.append((spk, toks, text))
    # Align the turns' words with Whisper's timed words. The line's text and
    # its timings can differ slightly ("Yeah" attached to the line before),
    # so match fuzzily and take each turn's first matched word as its start.
    import difflib

    flat = [(n, x) for n, (_, toks, _) in enumerate(parsed) for x in toks]
    wt = [norm(w["word"]) for w in words]
    pos: dict[int, int] = {}
    for blk in difflib.SequenceMatcher(
        None, [x for _, x in flat], wt, autojunk=False
    ).get_matching_blocks():
        for i in range(blk.size):
            pos[blk.a + i] = blk.b + i
    starts, i = [], 0
    for n, (_, toks, _) in enumerate(parsed):
        idx = [pos[j] for j in range(i, i + len(toks)) if j in pos]
        if len(idx) < max(1, len(toks) // 2):
            return []
        starts.append(0 if n == 0 else idx[0])
        i += len(toks)
    if any(b <= a for a, b in zip(starts, starts[1:])):
        return []
    bounds = starts + [len(words)]
    spans = [(bounds[n], bounds[n + 1], spk, text) for n, (spk, _, text) in enumerate(parsed)]
    if len({sp for *_, sp, _ in spans}) < 2:
        return []
    next_id = max(s.id for s in segments) + 1
    out = []
    for n, (a, b, spk, text) in enumerate(spans):
        ws = words[a:b]
        piece = seg if n == 0 else Segment.from_dict(seg.to_dict())
        if n:
            piece.id, next_id = next_id, next_id + 1
            out.append(piece)
        piece.start, piece.end = round(ws[0]["start"], 3), round(ws[-1]["end"], 3)
        piece.words = ws
        piece.text = "".join(w["word"] for w in ws).strip()
        piece.speaker, piece.gender = spk, gender.get(spk, piece.gender)
        piece.dubbed_text = text
        piece.voice = None
    return out
