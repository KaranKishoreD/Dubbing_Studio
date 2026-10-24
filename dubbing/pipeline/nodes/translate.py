"""
Translation node.

Default (TRANSLATOR=llm): context-aware DIALOGUE translation. The whole
conversation goes to one strong LLM in a single request (chunked for long
videos), with each line's speaker, gender and time budget. Translating lines
one at a time, blind, is what produced:
  - gender errors — Tamil/Hindi verbs agree with the speaker, so a woman's
    "not too bad" came out as "I am a bad man";
  - register flipping between spoken and formal written forms line to line;
  - literal translation of speech-recognition errors ("selling you" for
    "settling in") instead of the obviously intended meaning;
  - dropped clauses with no sign anything was lost.

Fallback (no OpenRouter key, a failed call, or TRANSLATOR=bodhan): Bodhan
indic-translate, one line at a time.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from ..state import PipelineState, Segment
from ..cache import get_cache_dir

_TRANSLATOR = os.environ.get("TRANSLATOR", "llm").lower()
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "anthropic/claude-sonnet-5.5")
_REGISTER = os.environ.get("TRANSLATE_REGISTER", "spoken").lower()
_CHUNK_LINES = int(os.environ.get("TRANSLATE_CHUNK_LINES", "40"))

LANG_NAMES = {
    "hi": "Hindi", "ta": "Tamil", "te": "Telugu", "kn": "Kannada", "ml": "Malayalam",
    "bn": "Bengali", "mr": "Marathi", "gu": "Gujarati", "pa": "Punjabi", "or": "Odia",
    "as": "Assamese", "ur": "Urdu", "ne": "Nepali", "sa": "Sanskrit", "mai": "Maithili",
    "kok": "Konkani", "doi": "Dogri", "ks": "Kashmiri", "sd": "Sindhi", "mni": "Manipuri",
    "brx": "Bodo", "sat": "Santali", "en": "English",
}

_REGISTER_TEXT = {
    "spoken": ("natural, everyday SPOKEN {lang} — the way people actually talk, and the "
               "way a good professional {lang} dub sounds. Not formal written or literary "
               "{lang}. Keep this register identical on every line."),
    "formal": ("clear, formal {lang} suitable for news or official content. Keep this "
               "register identical on every line."),
    "neutral": ("neutral standard {lang}: correct and natural, neither slangy nor bookish. "
                "Keep this register identical on every line."),
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

# Script for English words that stay English. Bodhan mispronounces Latin
# text, so for it they are transliterated; Sarvam Bulbul reads code-mixed
# text, so there they stay in Latin script and sound like English.
_SCRIPT_NATIVE = ('BUT write every word in {lang} script — transliterate loanwords and names\n'
                  '   (e.g. "job" → the {lang}-script spelling of job). No Latin letters at all:\n'
                  '   the text-to-speech voice reads {lang} script and mispronounces Latin.')
_SCRIPT_LATIN = ('Words that stay English are written in normal English spelling '
                 '(Latin letters), e.g. "office", "weekend", "actually" — the voice '
                 'pronounces them as English. Everything else is in {lang} script. '
                 'Never write {lang} words in Latin letters.')
SARVAM_LANGS = {"hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "en"}


def latin_allowed(state) -> bool:
    """English words may stay in Latin script only when Sarvam voices them."""
    code = state["target_lang"].split("-")[0]
    return ((state.get("tts_backend") or "").lower() == "sarvam"
            and code in SARVAM_LANGS and bool(os.environ.get("SARVAM_API_KEY")))


# How much English stays in the dub. Real spoken Tamil/Hindi/… is full of
# English; translating every word into its pure form sounds like a news
# bulletin, not a person.
LANG_MIX = os.environ.get("LANG_MIX", "standard").lower()
_MIX_TEXT = {
    # The original rule, unchanged — the default.
    "standard": ("Keep names, brands, places, numbers and common English loanwords "
                 "that {lang} speakers actually say in English (job, office, weekend, "
                 "team...),"),
    "pure": ("PURE {lang}. Translate everything; keep English only for names, "
             "brands and technical terms that have no {lang} equivalent."),
    "natural": ("NATURAL SPOKEN {lang}, the way a modern {lang} speaker really talks. "
                "Keep the English words such speakers keep in everyday conversation — "
                "common loanwords (office, meeting, weekend, phone, plan, problem, "
                "sorry, okay, actually, basically, seriously), technology and brand "
                "terms, and English the source speaker uses as a set phrase. Never "
                "replace a common English word with a rare pure-{lang} coinage nobody "
                "says. Grammar, verbs and connecting words stay {lang}. If a speaker "
                "already mixes {lang} and English, keep their level of mixing."),
    "heavy": ("CODE-MIXED speech ({lang} + English, like urban young speakers or "
              "vloggers): keep many English words and short English phrases; {lang} "
              "supplies the grammar, verb endings and connectors."),
}

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


def _parse_items(text: str) -> list:
    """Shared robust JSON-array extraction — models wrap or annotate their JSON."""
    from ..edit_resolver import extract_json_array
    return extract_json_array(text)


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

    pending = [s for s in segments if s.dubbed_text is None]
    if not pending:
        return {"log": state.get("log", []) + ["✓ Translation: all segments cached"]}

    seg_map = {s.id: s for s in segments}
    target_lang = state["target_lang"]
    use_llm = _TRANSLATOR == "llm" and bool(os.environ.get("OPENROUTER_API_KEY"))
    mix = (state.get("lang_mix") or LANG_MIX).lower()
    if mix not in _MIX_TEXT:
        mix = "standard"

    if use_llm:
        # One group per instruction: plain lines together, each edit
        # instruction ("make it funny") as its own group.
        groups: dict[str | None, list[Segment]] = {}
        for seg in pending:
            groups.setdefault(seg.edit_instruction, []).append(seg)
        failed: list[Segment] = []
        for instruction, group in groups.items():
            failed += _translate_dialogue(group, segments, target_lang,
                                          instruction, state.get("domain"), mix=mix,
                                          latin=latin_allowed(state))
        if failed:
            _translate_bodhan_parallel(failed, target_lang, seg_map)
        notes = [
            f"✓ Translated {len(pending)} lines as one dialogue with {TRANSLATE_MODEL} "
            f"(register: {_REGISTER}, language mix: {mix}"
            f"{', English kept in English spelling' if latin_allowed(state) else ''}, "
            "speaker gender + timing aware)"
            + (f"; {len(failed)} fell back to Bodhan" if failed else "")
        ]
    else:
        bodhan_group: list[Segment] = []
        openrouter_groups: dict[str, list[Segment]] = {}
        for seg in pending:
            instruction = seg.edit_instruction or "default"
            if instruction == "default" or not _needs_openrouter(instruction):
                bodhan_group.append(seg)
            else:
                openrouter_groups.setdefault(instruction, []).append(seg)
        if bodhan_group:
            _translate_bodhan_parallel(bodhan_group, target_lang, seg_map)
        for instruction, group in openrouter_groups.items():
            _translate_openrouter(group, target_lang, instruction, seg_map)
        why = "TRANSLATOR=bodhan" if _TRANSLATOR != "llm" else "no OPENROUTER_API_KEY"
        notes = [f"✓ Translated {len(pending)} lines line-by-line with Bodhan ({why})"]

    # Shorten anything that cannot be spoken in its slot. Doing this before
    # synthesis is what keeps the dub from sounding hurried — time-stretching
    # can then stay within a range the ear does not notice.
    shortened = _compress_overlong(segments, state["target_lang"])
    if shortened:
        notes.append(
            f"✓ Shortened {shortened} over-long segment(s) to fit their time slots"
        )

    cache_dir = get_cache_dir(state["video_id"])
    with open(cache_dir / "translations.json", "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, ensure_ascii=False, indent=2)

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + notes,
    }


# ── Context-aware dialogue translation ────────────────────────────────────────

def _translate_dialogue(
    group: list[Segment], all_segments: list[Segment], target_lang: str,
    instruction: str | None, domain: str | None, mix: str = LANG_MIX,
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
        extra += (f"9. Style instruction for these lines, apply it exactly: "
                  f"{instruction}\n")
    system = _DIALOGUE_SYSTEM.format(
        lang=lang,
        register=_REGISTER_TEXT.get(_REGISTER, _REGISTER_TEXT["spoken"]).format(lang=lang),
        mix=_MIX_TEXT.get(mix, _MIX_TEXT["standard"]).format(lang=lang),
        script_rule=(_SCRIPT_LATIN if latin else _SCRIPT_NATIVE).format(lang=lang),
        extra=extra,
    )

    group = sorted(group, key=lambda s: s.start)
    chunks = [group[i:i + _CHUNK_LINES] for i in range(0, len(group), _CHUNK_LINES)]
    budgets = time_budgets(ordered, target_lang)

    def run(chunk: list[Segment]) -> dict[int, str]:
        lines = [{
            "id": s.id, "speaker": s.speaker, "gender": s.gender or "unknown",
            "seconds": budgets[s.id][0],
            "budget_chars": budgets[s.id][1],
            "text": s.text,
            **({"alt_text": s.asr_alt} if s.asr_alt and s.asr_alt != s.text else {}),
        } for s in chunk]
        payload = json.dumps({"context": context, "lines": lines}, ensure_ascii=False)
        resp = _openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": payload}],
            temperature=0.3,
            timeout=180,
        )
        items = _parse_items(resp.choices[0].message.content.strip())
        wanted = {s.id for s in chunk}
        return {it["id"]: str(it["text"]).strip() for it in items
                if isinstance(it, dict) and it.get("id") in wanted and it.get("text")}

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
            del done[s.id]                 # still wrong script → Bodhan fallback

    for s in group:
        if s.id in done:
            s.dubbed_text = done[s.id]
    return [s for s in group if s.id not in done]


# Unicode blocks each target language is written in.
_DEVANAGARI = [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]
_ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]
_BENGALI = [(0x0980, 0x09FF)]
_SCRIPTS: dict[str, list[tuple[int, int]]] = {
    "hi": _DEVANAGARI, "mr": _DEVANAGARI, "ne": _DEVANAGARI, "sa": _DEVANAGARI,
    "mai": _DEVANAGARI, "doi": _DEVANAGARI, "kok": _DEVANAGARI, "brx": _DEVANAGARI,
    "bn": _BENGALI, "as": _BENGALI, "mni": _BENGALI + [(0xABC0, 0xABFF)],
    "ta": [(0x0B80, 0x0BFF)], "te": [(0x0C00, 0x0C7F)], "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)], "gu": [(0x0A80, 0x0AFF)], "pa": [(0x0A00, 0x0A7F)],
    "or": [(0x0B00, 0x0B7F)], "ur": _ARABIC, "ks": _ARABIC, "sd": _ARABIC,
    "sat": [(0x1C50, 0x1C7F)],
}


def script_ok(text: str, lang_code: str, latin: bool = False) -> bool:
    """True if every letter in `text` belongs to the target language's script
    (or is basic Latin, when English words may stay in English spelling)."""
    import unicodedata
    blocks = _SCRIPTS.get(lang_code)
    if not blocks:
        return True                        # e.g. English target: nothing to check
    if latin:
        blocks = blocks + [(0x41, 0x5A), (0x61, 0x7A)]
    for ch in text:
        if not unicodedata.category(ch)[0] in ("L", "M"):
            continue                       # digits, punctuation, spaces are fine
        cp = ord(ch)
        if not any(a <= cp <= b for a, b in blocks):
            return False
    return True


# ── Length control ────────────────────────────────────────────────────────────

# Bodhan indic-speak averages roughly this many characters per second of speech
# in Devanagari/Indic scripts. Used only to predict overruns before synthesis,
# so a rough figure is fine.
_CHARS_PER_SEC = float(os.environ.get("TTS_CHARS_PER_SEC", "14"))

# Only rewrite a segment if it overruns its slot by more than this factor —
# below it, align_node absorbs the difference without an audible artifact.
# Deliberately loose: this pre-synthesis check uses a rough chars/sec guess,
# so it only catches egregious overruns. synthesize_node re-checks with the
# real clip lengths and shortens precisely.
_OVERRUN_TRIGGER = float(os.environ.get("OVERRUN_TRIGGER", "1.15"))
_BUDGET_BORROW = float(os.environ.get("BUDGET_BORROW_SECS", "0.6"))


def time_budgets(ordered: list[Segment], target_lang: str) -> dict[int, tuple[float, int]]:
    """
    {id: (seconds, budget_chars)} per line. Seconds is the line's real slot —
    up to the next line, borrowing trailing silence — and characters come from
    the voice's measured speaking rate in this language.
    """
    from .align import slot_seconds
    from ..tts_rate import chars_per_sec
    cps = chars_per_sec(target_lang.split("-")[0])
    out = {}
    for i, s in enumerate(ordered):
        # Borrowing silence is a safety margin for the fitter, not part of the
        # target: a line written to fill 3 s of borrowed pause keeps talking
        # long after the speaker's mouth has stopped.
        secs = min(slot_seconds(ordered, i, None), s.duration + _BUDGET_BORROW)
        out[s.id] = (round(secs, 2), max(4, int(secs * cps * 0.95)))
    return out

_COMPRESS_SYSTEM = """\
You shorten dubbed subtitle lines so they can be spoken aloud in a fixed time.

For each item you receive an id, the text, and a character budget.
Rewrite the text to fit within its budget while preserving meaning.

Rules:
1. Stay in the SAME language and script as the input. Do not translate.
2. Preserve the core meaning. Drop filler, hedges and redundancy first.
3. Keep technical terms, names and numbers intact.
4. The result must read as natural, speakable prose — not clipped notes.
5. Keep the same register (spoken vs formal) and the same gendered verb
   endings and pronouns as the input line.
6. Never exceed the budget. Coming in under it is fine.

Return ONLY valid JSON (no markdown):
[{"id": <int>, "text": "<shortened text>"}]
"""


def _compress_overlong(segments: list[Segment], target_lang: str) -> int:
    """
    Rewrite segments whose text is too long to speak in their slot.
    Returns the number actually shortened. No-ops without an OpenRouter key.
    """
    if not os.environ.get("OPENROUTER_API_KEY"):
        return 0

    budgets = time_budgets(sorted(segments, key=lambda s: s.start), target_lang)
    jobs: list[tuple[Segment, int]] = []
    for seg in segments:
        if not seg.dubbed_text:
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
        resp = _openrouter_client().chat.completions.create(
            model=os.environ.get("TRANSLIT_MODEL", "google/gemini-3.8-flash"),
            messages=[{"role": "system", "content": _TRANSLIT_SYSTEM.format(lang=lang)},
                      {"role": "user", "content": json.dumps(
                          [{"id": i, "text": t} for i, t in items], ensure_ascii=False)}],
            temperature=0.0, timeout=60)
        out = _parse_items(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"  ⚠ transliteration skipped: {exc}")
        return {}
    wanted = dict(items)
    return {it["id"]: str(it["text"]).strip() for it in out
            if isinstance(it, dict) and it.get("id") in wanted and it.get("text")
            and script_ok(str(it["text"]), lang_code)}


def has_latin(text: str | None) -> bool:
    import re
    return bool(text) and bool(re.search(r"[A-Za-z]", text))


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
        resp = _openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2,
        )
        items = _parse_items(resp.choices[0].message.content.strip())
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

    items = _parse_items(resp.choices[0].message.content.strip())
    for i, t in enumerate(items):
        # LLM sometimes re-indexes IDs; fall back to group position
        seg_id = t["id"] if t["id"] in seg_map else (group[i].id if i < len(group) else None)
        if seg_id is not None:
            seg_map[seg_id].dubbed_text = t["dubbed_text"]
