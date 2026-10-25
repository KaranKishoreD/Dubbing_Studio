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

# Default: Indic — Bodhan translates every line; the LLM only fixes lines
# that rule checks or the cheap screen flag.
_TRANSLATOR = os.environ.get("TRANSLATOR", "indic").lower()
# Opus 5.5 won the bake-off: every line inside its time window, fewest
# review fixes, fastest. Screening uses a cheap model; Opus only fixes
# lines that fail.
TRANSLATE_MODEL = os.environ.get("TRANSLATE_MODEL", "anthropic/claude-opus-5.5")
SCREEN_MODEL = os.environ.get("SCREEN_MODEL", "google/gemini-3.8-flash")
# Shortening / lengthening / length-fit are mechanical rewording: a cheap
# model (~5x cheaper than Opus). Opus only fixes flagged meaning problems.
LENGTH_MODEL = os.environ.get("LENGTH_MODEL", "google/gemini-3.8-flash")
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
             min_chars, max_chars, text.

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
6. Length WINDOW: min_chars–max_chars is how many {lang} characters the
   voice speaks in this line's time at a natural pace (measured, not
   guessed). Land INSIDE the window.
   - Above max_chars: say it the brief way a native speaker would — shorter
     words, restructure — never by cutting meaning.
   - Below min_chars: the dub would end while the speaker is still visibly
     talking. Use the fuller natural phrasing a native speaker would use for
     the WHOLE original sentence. Never pad with filler or new content.
   Either way NEVER drop a clause, a connecting word (but, so, because,
   then, only, also, still), a negation, a question, a name or a number.
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


# Bodhan reads English words written in English spelling (tested: "first
# week", "family", "chill" come out as English; preferred by ear over the
# Tamil-script spelling). BODHAN_LATIN=0 restores Tamil-script spelling.
BODHAN_LATIN = os.environ.get("BODHAN_LATIN", "1") == "1"


def latin_allowed(state) -> bool:
    """May English words stay in English (Latin) spelling for this engine?"""
    code = state["target_lang"].split("-")[0]
    if code == "en":
        return False
    engine = (state.get("tts_backend") or "bodhan").lower()
    if engine == "sarvam" and code in SARVAM_LANGS and os.environ.get("SARVAM_API_KEY"):
        return True
    return BODHAN_LATIN


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
    from .. import llm_budget
    return llm_budget.client()


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

    translator = (state.get("translator") or _TRANSLATOR).lower()
    if translator == "indic" and os.environ.get("BODHAN_TRANSLATE_KEY"):
        # Indic mode: Bodhan indic-translate does the translation; the LLM only
        # checks and corrects it below (meaning, gender agreement, naturalness,
        # length). Lines Bodhan cannot translate fall back to the LLM.
        failed = _translate_bodhan_parallel(pending, target_lang, seg_map)
        if failed and use_llm:
            failed = _translate_dialogue(failed, segments, target_lang, None,
                                         state.get("domain"), mix=mix,
                                         latin=latin_allowed(state))
        notes = [f"✓ Translated {len(pending)} lines with Bodhan indic-translate (Indic mode)"
                 + (f"; {len(failed)} could not be translated" if failed else "")]
    elif use_llm:
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
    has_key = bool(os.environ.get("OPENROUTER_API_KEY"))
    if has_key and _POLISH:
        # In Indic mode this is the LLM correction of Bodhan's translation.
        polished = native_polish(pending, segments, target_lang, mix, latin_allowed(state))
        from .. import llm_budget
        notes.append(f"✓ Checked every line (rules + {SCREEN_MODEL.split('/')[-1]} screen); "
                     f"{polished} flagged line(s) fixed with {(REVIEW_MODEL or TRANSLATE_MODEL).split('/')[-1]}"
                     f" — LLM spend so far ${llm_budget.spent():.3f}")
    if has_key:
        checked, fitted = fit_lengths(pending, segments, target_lang, mix, latin_allowed(state))
        if checked:
            notes.append(f"✓ Length fit: {checked} line(s) outside their time window, "
                         f"{fitted} replaced by a better-fitting version")

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
            "min_chars": budgets[s.id][2],
            "max_chars": budgets[s.id][1],
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
            max_tokens=12000,
        )
        items = _parse_items(resp.choices[0].message.content.strip())
        wanted = {s.id for s in chunk}
        return {_as_id(it.get("id")): str(it["text"]).strip() for it in items
                if isinstance(it, dict) and _as_id(it.get("id")) in wanted and it.get("text")}

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


# The dub should cover most of the time the speaker is visibly talking: a line
# that ends early leaves the original lips moving over silence (measured on
# the office video: 1.6 s of silence at the end of a 10 s line, and a whole
# clause dropped). So each line gets a minimum as well as a maximum length.
_MIN_FILL = float(os.environ.get("MIN_FILL", "0.85"))


def time_budgets(ordered: list[Segment], target_lang: str) -> dict[int, tuple[float, int, int]]:
    """
    {id: (seconds, max_chars, min_chars)} per line. Seconds is the line's real
    slot — up to the next line, borrowing a little trailing silence. max_chars
    fits that slot at the language's measured speaking rate; min_chars fills
    _MIN_FILL of the time the original speaker actually talks.
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
        max_chars = max(4, int(secs * cps * 0.95))
        min_chars = min(int(max_chars * 0.9), int(s.duration * _MIN_FILL * cps))
        out[s.id] = (round(secs, 2), max_chars, max(0, min_chars))
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


# ── Review: meaning, naturalness, length ──────────────────────────────────────
# One pass, after translation, that reads every line the way a native-speaking
# dubbing editor would and fixes only what is wrong:
#   - meaning: a dropped clause ("I know what you mean, I'm a bit the same"
#     vanished), a dropped "but", a flipped negation;
#   - naturalness: "நாலு வருஷமா இங்க, டிபார்ட்மெண்ட் மாறினேன்" where people say
#     "நாலு வருஷத்துக்கு அப்புறம் department மாறினேன்";
#   - length: outside the line's min–max window.
_POLISH = os.environ.get("NATIVE_POLISH", "1") == "1"
REVIEW_MODEL = os.environ.get("REVIEW_MODEL", "") or None   # default: TRANSLATE_MODEL

_REVIEW_SYSTEM = """\
You are a native {lang} speaker and senior dubbing editor checking a video's
{lang} dub line by line. Each line has the original (English), the current
{lang} dub, the speaker's gender and a measured length verdict.
Neighbouring lines are given for context.

A line FAILS if any of these is true:
  A. Meaning: anything in the original is missing or changed in the dub — a
     clause, a connecting word (but, so, because, then, only, also, still),
     a negation, a question, who did what, time order, a name or number.
  B. Naturalness: a real {lang} speaker would not say it this way in
     everyday speech (stiff, literal, bookish, translated-sounding).
  C. Length: its "length" field says "too short" or "too long". The length
     has already been measured for you — do NOT count characters yourself.

These lines were flagged by a checker ("flagged_because"). Confirm and fix
the real problem; if a flag is wrong and the line is fine, leave it out.

For every failing line, write the fixed line:
  1. Complete meaning of the original, in the natural spoken way.
  2. Speaker's gender agreement; same conversational register.
  3. English words: {english_rule}
  4. If it was too short/long, about that many characters longer/shorter.
     Never pad with filler or invent content.

Output ONLY the JSON array — no reasoning, no explanations, no counting:
[{{"id": <int>, "failed": "A"|"B"|"C"|"AB"|..., "text": "<fixed {lang} line>"}}]
Return [] if every line passes.
"""


# ── Screening: decide WHICH lines need the expensive fix ─────────────────────
# Reviewing every line with a top model cost ~$0.70 per 2-minute video.
# Instead: free rule checks + a cheap pass/fail screen (no rewriting), and
# the top model fixes only what was flagged. Biased toward flagging: a false
# alarm costs one extra line of review, a miss ships a wrong dub.
import re as _re

_NEG_EN = _re.compile(r"\b(not|no|never|nothing|nobody|none|nor|cannot)\b|n't\b", _re.I)
_BUT_EN = _re.compile(r"\b(but|however|though|although|yet)\b", _re.I)
_RULES = {   # target-language markers (only where we know them reliably)
    "ta": {"neg": ("இல்ல", "ல்ல", "மாட்ட", "வேண்டாம்", "கூடாது", "முடியாது", "ாத", "லை"),
           "but": ("ஆனா", "ஆனால்", "இருந்தாலும்", "ஆனாலும்", "but")},
    "hi": {"neg": ("नहीं", "न ", "मत", "कभी नहीं"), "but": ("लेकिन", "पर ", "मगर", "but")},
}
_SCREEN_SYSTEM = """\
You check a {lang} dub against the original line by line. Do NOT rewrite.
A line FAILS only if (a) meaning from the original is missing or changed —
a clause, a connecting word like "but", a negation, a question, who/when,
a name or number — or (b) a native {lang} speaker would clearly not say it
that way. Minor wording differences that keep the meaning are a PASS.
Return ONLY JSON listing failing lines with a short reason:
[{{"id": <int>, "issue": "<what is missing or wrong, max 12 words>"}}]
Return [] if all pass.
"""


def screen_lines(lines: list[Segment], target_lang: str) -> dict[int, str]:
    """{id: reason} for lines that need fixing. Free rules + cheap screen."""
    from .. import llm_budget
    code = target_lang.split("-")[0]
    lang = LANG_NAMES.get(code, code)
    flags: dict[int, str] = {}
    rules = _RULES.get(code)
    for s in lines:
        src, dub = s.text or "", s.dubbed_text or ""
        if "?" in src and "?" not in dub:
            flags[s.id] = "the question became a statement"
        if rules and _NEG_EN.search(src) and not any(m in dub for m in rules["neg"]):
            flags[s.id] = "the negation is missing"
        if rules and _BUT_EN.search(src) and not any(m in dub for m in rules["but"]):
            flags[s.id] = "the 'but' / contrast is missing"
        if _re.findall(r"\d+", src) and not _re.findall(r"\d+", dub) and len(dub) < 2 * len(src):
            flags.setdefault(s.id, "a number may be missing")
    if lines and llm_budget.allow(0.02):
        payload = [{"id": s.id, "original": s.text, "dub": s.dubbed_text} for s in lines]
        try:
            resp = _openrouter_client().chat.completions.create(
                model=SCREEN_MODEL,
                messages=[{"role": "system", "content": _SCREEN_SYSTEM.format(lang=lang)},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0.0, timeout=120, max_tokens=1500)
            for it in _parse_items(resp.choices[0].message.content.strip()):
                if isinstance(it, dict) and _as_id(it.get("id")) is not None:
                    flags.setdefault(_as_id(it["id"]), str(it.get("issue") or "flagged"))
        except Exception as exc:
            print(f"  ⚠ screening skipped: {exc}")
    return flags


def native_polish(lines: list[Segment], all_segments: list[Segment], target_lang: str,
                  mix: str, latin: bool) -> int:
    """Fix only the lines that fail screening or their length window.
    Returns how many changed."""
    from .. import llm_budget
    code = target_lang.split("-")[0]
    todo = [s for s in lines if s.dubbed_text and not s.keep_original]
    if not todo or code == "en":
        return 0
    ordered_all = sorted(all_segments, key=lambda s: s.start)
    b_all = time_budgets(ordered_all, target_lang)
    issues = screen_lines(todo, target_lang)
    for s in todo:
        v = _length_verdict(len(s.dubbed_text), b_all[s.id][2], b_all[s.id][1])
        if v != "ok":
            issues[s.id] = (issues.get(s.id, "") + "; " if s.id in issues else "") + v
    todo = [s for s in todo if s.id in issues]
    if not todo or not llm_budget.allow(0.10):
        return 0
    lang = LANG_NAMES.get(code, code)
    ordered = sorted(all_segments, key=lambda s: s.start)
    budgets = time_budgets(ordered, target_lang)
    pos = {s.id: k for k, s in enumerate(ordered)}
    system = _REVIEW_SYSTEM.format(lang=lang, english_rule=_english_rule(lang, mix, latin))
    changed = 0
    for i in range(0, len(todo), _CHUNK_LINES):
        chunk = todo[i:i + _CHUNK_LINES]
        payload = []
        for s in chunk:
            k = pos[s.id]
            payload.append({
                "id": s.id, "gender": s.gender or "unknown",
                "previous_original": ordered[k - 1].text if k > 0 else "",
                "original": s.text, "dub": s.dubbed_text,
                "next_original": ordered[k + 1].text if k + 1 < len(ordered) else "",
                "length": _length_verdict(len(s.dubbed_text), budgets[s.id][2],
                                          budgets[s.id][1]),
                "flagged_because": issues.get(s.id, "")})
        try:
            resp = _openrouter_client().chat.completions.create(
                model=REVIEW_MODEL or TRANSLATE_MODEL,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
                temperature=0.2, timeout=240, max_tokens=_MAX_OUT_TOKENS)
            items = _parse_items(resp.choices[0].message.content.strip())
        except Exception as exc:
            print(f"  ⚠ review skipped for a chunk: {exc}")
            continue
        by_id = {s.id: s for s in chunk}
        for it in items:
            if not isinstance(it, dict):
                continue
            seg = by_id.get(_as_id(it.get("id")))
            text = str(it.get("text") or "").strip()
            if seg and text and text != seg.dubbed_text and script_ok(text, code, latin):
                seg.dubbed_text = text
                changed += 1
    return changed


def _as_id(x):
    """Line ids come back as 3, "3", "S3" or 3.0 depending on the model."""
    import re
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"-?\d+", str(x))
    return int(m.group()) if m else None


def _length_verdict(n: int, lo: int, hi: int) -> str:
    if n < lo:
        return f"too short by ~{lo - n} characters"
    if n > hi:
        return f"too long by ~{n - hi} characters"
    return "ok"


# Every review-type call is capped: one review once returned 37,423 tokens
# (counting Tamil characters aloud) and cost $0.39 for a 2-minute video.
_MAX_OUT_TOKENS = int(os.environ.get("LLM_MAX_OUTPUT_TOKENS", "4000"))


def _english_rule(lang: str, mix: str, latin: bool) -> str:
    if mix == "pure":
        return "use English only for names and terms with no common equivalent."
    return ("keep the English words people really use, written in normal English "
            "spelling." if latin else
            f"keep the English words people really use, written in {lang} script.")


# ── Length fit: candidates, pick the one that fits ───────────────────────────
# Dubbing research: generating several translations of different lengths and
# picking the best-fitting one beats speeding audio up or leaving silence.
# Only lines still outside their window after review get this (one call).
_FIT_SYSTEM = """\
For each dubbed {lang} line below, write THREE alternative {lang} versions of
the SAME original sentence, all complete in meaning (no clause, connective,
negation or question dropped), all natural spoken {lang}, same speaker gender:
  "concise": as short as a native speaker would naturally say it,
  "natural": the normal way,
  "full":    the fuller way a native speaker would say it (no filler).
English words: {english_rule}
Return ONLY JSON: [{{"id": <int>, "concise": "...", "natural": "...", "full": "..."}}]
"""


def fit_lengths(lines: list[Segment], all_segments: list[Segment], target_lang: str,
                mix: str, latin: bool) -> tuple[int, int]:
    """Re-fit lines whose length is outside their window. Returns (checked, fixed)."""
    code = target_lang.split("-")[0]
    if code == "en":
        return 0, 0
    ordered = sorted(all_segments, key=lambda s: s.start)
    budgets = time_budgets(ordered, target_lang)
    out = [s for s in lines if s.dubbed_text and not s.keep_original
           and not (budgets[s.id][2] <= len(s.dubbed_text) <= budgets[s.id][1])]
    if not out:
        return 0, 0
    from .. import llm_budget
    if not llm_budget.allow(0.05):
        return len(out), 0
    lang = LANG_NAMES.get(code, code)
    payload = [{"id": s.id, "gender": s.gender or "unknown", "original": s.text,
                "current_dub": s.dubbed_text} for s in out]
    try:
        resp = _openrouter_client().chat.completions.create(
            model=LENGTH_MODEL,
            messages=[{"role": "system", "content": _FIT_SYSTEM.format(
                          lang=lang, english_rule=_english_rule(lang, mix, latin))},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            temperature=0.3, timeout=240, max_tokens=_MAX_OUT_TOKENS)
        items = _parse_items(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"  ⚠ length fit skipped: {exc}")
        return len(out), 0
    by_id = {s.id: s for s in out}
    fixed = 0
    for it in items:
        seg = by_id.get(_as_id(it.get("id"))) if isinstance(it, dict) else None
        if not seg:
            continue
        lo, hi = budgets[seg.id][2], budgets[seg.id][1]
        target = (lo + hi) / 2
        cands = [str(it.get(k) or "").strip() for k in ("concise", "natural", "full")]
        cands = [c for c in cands if c and script_ok(c, code, latin)] + [seg.dubbed_text]
        # Prefer candidates inside the window; among those, the natural one
        # nearest the middle. Otherwise whichever is nearest the window.
        def dist(c):
            n = len(c)
            return 0 if lo <= n <= hi else min(abs(n - lo), abs(n - hi))
        best = min(cands, key=lambda c: (dist(c), abs(len(c) - target)))
        if best != seg.dubbed_text:
            seg.dubbed_text = best
            fixed += 1
    return len(out), fixed


_EXPAND_SYSTEM = """\
Each item is a dubbed {lang} line whose spoken audio ends well before the
speaker stops talking on screen, so viewers see lips moving over silence.
Rewrite it so it fills its time: between min_chars and max_chars characters.

How:
1. First restore anything from the original (English) the dub left out —
   clauses, connecting words (but, so, because, then, only, also), hedges
   ("I mean", "maybe"), names. Most short dubs are short because they dropped
   something.
2. Then, if still short, use the fuller natural way a native speaker would say
   the same thing. Never add facts, jokes or content that is not in the
   original, and no meaningless filler.
3. Natural spoken {lang}, same speaker gender, same register.
   English words: {english_rule}

Return ONLY JSON: [{{"id": <int>, "text": "<{lang} line>"}}]
"""


def expand_to_window(jobs: list[dict], target_lang: str, latin: bool) -> dict[int, str]:
    """jobs: [{id, original, dub, gender, min_chars, max_chars}] → {id: longer line}."""
    from .. import llm_budget
    if not jobs or not os.environ.get("OPENROUTER_API_KEY") or not llm_budget.allow(0.05):
        return {}
    code = target_lang.split("-")[0]
    lang = LANG_NAMES.get(code, code)
    try:
        resp = _openrouter_client().chat.completions.create(
            model=LENGTH_MODEL,
            messages=[{"role": "system", "content": _EXPAND_SYSTEM.format(
                          lang=lang, english_rule=_english_rule(lang, LANG_MIX, latin))},
                      {"role": "user", "content": json.dumps(jobs, ensure_ascii=False)}],
            temperature=0.2, timeout=180, max_tokens=_MAX_OUT_TOKENS)
        items = _parse_items(resp.choices[0].message.content.strip())
    except Exception as exc:
        print(f"  ⚠ underrun expansion skipped: {exc}")
        return {}
    old = {j["id"]: j for j in jobs}
    out = {}
    for it in items:
        if not isinstance(it, dict) or _as_id(it.get("id")) not in old:
            continue
        text = str(it.get("text") or "").strip()
        j = old[_as_id(it["id"])]
        # Only a genuinely longer line that does not blow the maximum.
        if (text and script_ok(text, code, latin) and len(text) > len(j["dub"])
                and len(text) <= j["max_chars"] * 1.1):
            out[_as_id(it["id"])] = text
    return out


def condense_to_budget(jobs: list[tuple[int, str, int]]) -> dict[int, str]:
    """
    Shorten each (id, text, budget_chars) with one batched LLM call.
    Returns {id: shorter_text} for the ones that genuinely got shorter.
    Empty dict if no OpenRouter key or the call fails.
    """
    from .. import llm_budget
    if not jobs or not os.environ.get("OPENROUTER_API_KEY") or not llm_budget.allow(0.05):
        return {}
    payload = [{"id": i, "text": t, "budget_chars": b} for i, t, b in jobs]
    try:
        resp = _openrouter_client().chat.completions.create(
            model=LENGTH_MODEL,
            messages=[
                {"role": "system", "content": _COMPRESS_SYSTEM},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            temperature=0.2, max_tokens=_MAX_OUT_TOKENS,
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
) -> list[Segment]:
    """Bodhan indic-translate, line by line, 4 at a time with back-off on rate
    limits (10 at once with no retry crashed the step on the first 429).
    Returns the lines that could not be translated."""
    import time
    client = _bodhan_client()

    def _translate_one(seg: Segment) -> tuple[int, str | None]:
        for attempt in range(5):
            try:
                resp = client.chat.completions.create(
                    model="indic-translate",
                    messages=[{"role": "user", "content": seg.text}],
                    extra_body={"target_language_code": target_lang,
                                "target_script": "native"},
                )
                return seg.id, (resp.choices[0].message.content or "").strip() or None
            except Exception as exc:
                if "429" not in str(exc) or attempt == 4:
                    print(f"  ⚠ Bodhan translate failed for line {seg.id}: {exc}")
                    return seg.id, None
                time.sleep(2 * (attempt + 1))
        return seg.id, None

    failed = []
    with ThreadPoolExecutor(max_workers=min(len(group), 4) or 1) as pool:
        for seg_id, text in pool.map(_translate_one, group):
            if text:
                seg_map[seg_id].dubbed_text = text
            else:
                failed.append(seg_map[seg_id])
    return failed


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
