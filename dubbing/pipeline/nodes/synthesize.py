"""
TTS synthesis via Bodhan indic-speak — all segments synthesized in parallel.
₹6 / 10K input characters — 45 voices across 22 Indian languages.

Voice auto-selection
─────────────────────
Gender and speaker identity come from diarize_node, which clusters segments by
voice and pools F0 across each speaker's whole set of segments. One decision
per speaker, so a speaker keeps the same voice for the entire video.

Manual override: set voice_name in state to use a specific voice for all segments.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ..state import PipelineState, Segment
from ..cache import get_segments_dir, instruction_hash
from openai import OpenAI

# ── Complete voice catalogue (from Bodhan API docs) ───────────────────────────
# Source: https://console.bodhan.ai/api-docs/#text-to-speech-api
# Any voice can read any supported language — voice sets persona/timbre.

# Lists, not single names: a language may have more than one voice (Hindi has
# two female voices), and every one of the 45 must be reachable when casting.
# First entry is the language's primary voice.
_FEMALE_VOICES: dict[str, list[str]] = {
    "hi":  ["Kavya", "Suhani"],
    "as":  ["Prastuti"],
    "bn":  ["Ishita"],
    "brx": ["Gwrbw"],
    "doi": ["Preeti"],
    "gu":  ["Dhara"],
    "kn":  ["Deepika"],
    "kok": ["Anjali"],
    "ks":  ["Zoon"],
    "mai": ["Vaidehi"],
    "ml":  ["Lakshmi"],
    "mni": ["Thoibi"],
    "mr":  ["Anagha"],
    "ne":  ["Srijana"],
    "or":  ["Itishree"],
    "pa":  ["Kaur"],
    "sa":  ["Bharati"],
    "sat": ["Phulmani"],
    "sd":  ["Moomal"],
    "ta":  ["Anitha"],
    "te":  ["Sravani"],
    "ur":  ["Saba"],
}

_MALE_VOICES: dict[str, list[str]] = {
    "hi":  ["Amit"],
    "as":  ["Ankur"],
    "bn":  ["Sourav"],
    "brx": ["Sansuma"],
    "doi": ["Sham"],
    "gu":  ["Parth"],
    "kn":  ["Adarsh"],
    "kok": ["Sandeep"],
    "ks":  ["Ishfaq"],
    "mai": ["Madhukar"],
    "ml":  ["Kiran"],
    "mni": ["Chaoba"],
    "mr":  ["Chinmay"],
    "ne":  ["Sagar"],
    "or":  ["Akash"],
    "pa":  ["Manpreet"],
    "sa":  ["Aryaman"],
    "sat": ["Sibu"],
    "sd":  ["Rano"],
    "ta":  ["Arun"],
    "te":  ["Vamsi"],
    "ur":  ["Zaid"],
}

# English source/target reads best with the Hindi voices.
_LANG_ALIAS = {"en": "hi"}

ALL_FEMALE_VOICES = [v for vs in _FEMALE_VOICES.values() for v in vs]
ALL_MALE_VOICES   = [v for vs in _MALE_VOICES.values() for v in vs]
ALL_BODHAN_VOICES = ALL_FEMALE_VOICES + ALL_MALE_VOICES

# ── Sarvam Bulbul v3 ──────────────────────────────────────────────────────────
# A second voice engine (SARVAM_API_KEY). Every Bulbul voice speaks every
# supported language, so the pool is not per language. Ids carry a "sarvam:"
# prefix so a cast can mix engines and the cache key separates them. Gender
# is from the voice name, confirmed by measured pitch (voice_profiles).
SARVAM_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
_SARVAM_FEMALE = ["ritu", "priya", "neha", "pooja", "simran", "kavya", "ishita", "shreya",
                  "roopa", "tanya", "shruti", "suhani", "kavitha", "rupali"]
_SARVAM_MALE = ["aditya", "ashutosh", "rahul", "rohan", "amit", "dev", "ratan", "varun",
                "manan", "sumit", "kabir", "aayan", "shubh", "advait", "anand", "tarun",
                "sunny", "mani", "gokul", "vijay", "mohit", "rehan", "soham"]
SARVAM_VOICES = [f"sarvam:{v}" for v in _SARVAM_FEMALE + _SARVAM_MALE]
# Languages Bulbul speaks (BCP-47 with -IN).
_SARVAM_LANGS = {"hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "en"}

ALL_VOICES = ALL_BODHAN_VOICES + SARVAM_VOICES
# "bodhan" | "sarvam" — which engine automatic casting draws from.
TTS_BACKEND = os.environ.get("TTS_BACKEND", "bodhan").lower()


def is_sarvam(voice: str | None) -> bool:
    return bool(voice) and voice.startswith("sarvam:")


def voice_label(voice: str | None) -> str:
    """Display name: 'Anand (Sarvam)' / 'Arun'."""
    if not voice:
        return "—"
    return f"{voice.split(':', 1)[1].title()} (Sarvam)" if is_sarvam(voice) else voice


def _sarvam_gender(voice: str) -> str:
    return "female" if voice.split(":", 1)[1] in _SARVAM_FEMALE else "male"


def voice_catalogue() -> list[tuple[str, str, str]]:
    """[(voice id, "female"|"male", native language code or "any")] — Bodhan's
    45 voices plus Sarvam's 37."""
    out = []
    for gender, table in (("female", _FEMALE_VOICES), ("male", _MALE_VOICES)):
        for lang, voices in table.items():
            out += [(v, gender, lang) for v in voices]
    out += [(v, _sarvam_gender(v), "any") for v in SARVAM_VOICES]
    return out


def _sarvam_tts(text: str, lang_code: str, voice: str) -> bytes:
    """WAV bytes from Sarvam Bulbul. Retries rate limits with back-off."""
    import base64
    import time
    import requests
    if lang_code not in _SARVAM_LANGS:
        raise RuntimeError(f"Sarvam Bulbul does not support '{lang_code}'")
    lang = lang_code
    body = {"text": text, "target_language_code": f"{lang}-IN", "model": SARVAM_MODEL,
            "speaker": voice.split(":", 1)[1], "speech_sample_rate": 24000}
    for attempt in range(5):
        r = requests.post("https://api.sarvam.ai/text-to-speech", json=body, timeout=120,
                          headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]})
        if r.status_code == 429 and attempt < 4:
            time.sleep(2 * (attempt + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"Sarvam TTS {r.status_code}: {r.text[:200]}")
        return base64.b64decode(r.json()["audios"][0])
    raise RuntimeError("Sarvam TTS: rate limited")


def _bodhan_tts(client, text: str, lang_code: str, voice: str, style: str) -> bytes:
    """
    WAV bytes from Bodhan indic-speak. A rate-limit reply (429) is waited
    out and retried — 2, 4, 8, 16, 30 s — instead of leaving the line
    undubbed (one Small talk line went silent on a single 429).
    """
    import time
    waits = [2, 4, 8, 16, 30]
    for attempt in range(len(waits) + 1):
        try:
            return client.audio.speech.create(
                model="indic-speak", input=text, voice=voice,
                extra_body={"instructions": json.dumps({"lang": lang_code, "style": style})},
            ).content
        except Exception as exc:
            if "429" not in str(exc) or attempt == len(waits):
                raise
            print(f"  … Bodhan rate limit — retrying in {waits[attempt]}s")
            time.sleep(waits[attempt])


def tts_bytes(text: str, lang_code: str, voice: str, style: str, client=None) -> bytes:
    if is_sarvam(voice):
        return _sarvam_tts(text, lang_code, voice)
    return _bodhan_tts(client or _client(), text, lang_code, voice, style)


def preview_voice(voice: str, lang_code: str, text: str, out_path: Path) -> str:
    """Synthesize a short sample of `voice` reading `text` (cached on disk)."""
    if not out_path.exists():
        audio = tts_bytes(text, lang_code, voice, _AUTO_BASE_STYLE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio)
    return str(out_path)

_DEFAULT_FEMALE = "Kavya"
_DEFAULT_MALE   = "Amit"

# Used only when diarize could not measure pitch at all. Male is the safer
# default for lecture/talk content, which skews male; override with
# UNKNOWN_GENDER_FALLBACK=female or set a voice explicitly in the UI.
_UNKNOWN_GENDER_FALLBACK = os.environ.get("UNKNOWN_GENDER_FALLBACK", "male")

# ── Speaking styles ────────────────────────────────────────────────────────────
BODHAN_STYLES: dict[str, str] = {
    # Per-line: each line's measured delivery (prosody.py) picks its style,
    # on top of a neutral "educational lecture" base.
    "Auto (match original delivery)": "auto",
    "Educational / Lecture":    "educational lecture",
    "TV News / Formal":         "TV style news",
    "AIR News":                 "AIR style news",
    "General News":             "news",
    "Advertisement":            "advertisements",
    "Customer Care":            "Customer Care",
    "Happy / Energetic":        "happy",
    "Sad / Emotional":          "sad",
    "Angry":                    "anger",
    "Surprised / Dramatic":     "surprise",
    "Fear":                     "fear",
    "Disgusted":                "disgust",
    "Storytelling / Audiobook": "single person narration audiobook",
    "Children's Story":         "children's stories",
}
DEFAULT_STYLE_LABEL = "Auto (match original delivery)"
DEFAULT_STYLE       = "auto"
_AUTO_BASE_STYLE    = os.environ.get("AUTO_BASE_STYLE", "educational lecture")


def _line_style(seg: Segment, global_style: str) -> str:
    """Global style, or in auto mode the style matching this line's delivery."""
    if seg.style_override:             # set by an edit ("make line 5 sad")
        return seg.style_override
    if global_style != "auto":
        return global_style
    from ..prosody import BODHAN_FOR_DELIVERY
    return BODHAN_FOR_DELIVERY.get(seg.delivery or "neutral", _AUTO_BASE_STYLE)


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://api.bodhan.ai/v1",
        api_key=os.environ["BODHAN_SPEAK_KEY"],
    )


def _pick_voice(lang_code: str, gender: str) -> str:
    return _voice_pool(lang_code, gender)[0]


# ── Per-speaker casting ───────────────────────────────────────────────────────
# Bodhan ships roughly one female and one male voice per language, so a second
# same-gender speaker has to borrow a voice from another language. Any voice can
# read any supported language — the voice only sets timbre — but a voice from a
# closely related language carries the nearest phonetic habits, so prefer those.
_LANG_GROUPS: list[list[str]] = [
    ["ta", "ml", "te", "kn"],                                  # Dravidian
    ["hi", "mr", "gu", "pa", "ur", "mai", "doi", "sa", "ne"],   # Indo-Aryan, west/central
    ["bn", "as", "or"],                                        # Indo-Aryan, east
    ["ks", "sd", "kok"],                                       # other Indo-Aryan
    ["brx", "mni", "sat"],                                     # Tibeto-Burman / Munda
]


def _voice_pool(lang_code: str, gender: str, backend: str = "bodhan") -> list[str]:
    """
    Ordered voice candidates for a language: its own voice first, then voices
    from languages in the same family, then everything else. Distinct speakers
    take successive entries, so each gets an audibly different voice.
    Sarvam voices are not tied to a language; they are ordered by measured
    pitch from the middle out, so the first picks sound typical, not extreme.
    """
    if backend == "sarvam" and lang_code in _SARVAM_LANGS:
        prof = voice_profiles()
        vs = [v for v in SARVAM_VOICES if _sarvam_gender(v) == gender]
        f0 = [prof.get(v, {}).get("f0") for v in vs]
        known = sorted(x for x in f0 if x)
        mid = known[len(known) // 2] if known else 0
        return sorted(vs, key=lambda v: abs((prof.get(v, {}).get("f0") or mid) - mid))
    table = _MALE_VOICES if gender == "male" else _FEMALE_VOICES
    lang_code = _LANG_ALIAS.get(lang_code, lang_code)
    group = next((g for g in _LANG_GROUPS if lang_code in g), [])

    ordered: list[str] = []
    for key in [lang_code, *group, *table.keys()]:
        for v in table.get(key, []):
            if v not in ordered:
                ordered.append(v)
    return ordered or [_DEFAULT_MALE if gender == "male" else _DEFAULT_FEMALE]


# Usual speaking pitch per gender (Hz), for casting only.
_F0_RANGE = {"male": (80, 175), "female": (160, 300)}


def _assign_voices(
    segments: list[Segment], lang_code: str, override: str | None,
    voice_map: dict | None = None, backend: str = "bodhan",
) -> dict[int, str]:
    """
    Map each speaker to a voice. One decision per speaker, so a speaker keeps
    the same voice throughout, and two speakers never share one unless the
    catalogue runs out.
    """
    speakers: dict[int, str] = {}      # speaker id -> gender
    talk: dict[int, float] = {}        # speaker id -> seconds spoken
    for seg in segments:
        spk = seg.speaker if seg.speaker is not None else 0
        gender = seg.gender if seg.gender in ("male", "female") else _UNKNOWN_GENDER_FALLBACK
        # First gender seen for a speaker wins; diarize already made it uniform.
        speakers.setdefault(spk, gender)
        talk[spk] = talk.get(spk, 0.0) + seg.duration

    if override:
        return {spk: override for spk in speakers}

    # Voices picked in the UI win; everyone else is cast around them.
    chosen = {int(k): v for k, v in (voice_map or {}).items()
              if v and v in ALL_VOICES and str(k).lstrip("-").isdigit()}
    used: set[str] = set(chosen.values())
    assignment: dict[int, str] = {spk: v for spk, v in chosen.items() if spk in speakers}

    # Deal voices out per gender so same-gender speakers get different ones.
    # Whoever talks most gets the target language's own voice — they carry
    # the video, so they should sound the most native.
    # Each speaker gets the unused voice whose MEASURED pitch is closest to
    # theirs. The target language's own voices get a ~1.5 semitone
    # preference so a native voice wins unless a foreign one is clearly
    # closer. Without pitch data, pool order decides.
    import math
    prof = voice_profiles()
    # Speaker pitch: duration-weighted median of their lines, clamped to the
    # normal range for their gender. Laughs and tracker octave errors put
    # single lines at 240–290 Hz for a ~145 Hz man; from two such lines a
    # "male" voice at 208 Hz was cast, which swung wildly within lines.
    f0s: dict[int, list[tuple[float, float]]] = {}
    for seg in segments:
        if seg.f0:
            f0s.setdefault(seg.speaker if seg.speaker is not None else 0, []).append(
                (seg.f0, max(seg.duration, 0.1)))
    spk_f0 = {}
    for k, v in f0s.items():
        v.sort()
        half, acc = sum(w for _, w in v) / 2, 0.0
        for f, w in v:
            acc += w
            if acc >= half:
                lo, hi = _F0_RANGE.get(speakers.get(k, ""), (60, 400))
                spk_f0[k] = min(max(f, lo), hi)
                break
    native = {v for v, _, lang in voice_catalogue() if lang == _LANG_ALIAS.get(lang_code, lang_code)}

    for gender in ("male", "female"):
        pool = _voice_pool(lang_code, gender, backend)
        # Voices far outside the usual range for their gender (a "male"
        # voice at 208 Hz) are only used when nothing else is left.
        lo, hi = _F0_RANGE[gender]
        typical = [v for v in pool if v in native
                   or lo <= (prof.get(v, {}).get("f0") or (lo + hi) / 2) <= hi]
        pool = typical or pool
        same = [s for s, g in speakers.items() if g == gender and s not in assignment]
        for spk in sorted(same, key=lambda s: -talk[s]):
            free = [v for v in pool if v not in used] or pool
            target = spk_f0.get(spk)

            def cost(v, k):
                vf = prof.get(v, {}).get("f0")
                if not target or not vf:
                    return k                       # no data: keep pool order
                # Native voices first: voice cloning replaces the timbre with
                # the real speaker's, but pronunciation stays the TTS voice's.
                return abs(12 * math.log2(vf / target)) + (0 if v in native else 24)
            voice = min(free, key=lambda v: cost(v, free.index(v)))
            used.add(voice)
            assignment[spk] = voice
    return assignment


def synthesize_node(state: PipelineState) -> dict:
    segments    = [Segment.from_dict(s) for s in state["segments"]]
    target_lang = state["target_lang"]
    edit_ops    = state.get("edit_ops", [])
    edit_ids    = {sid for op in edit_ops for sid in op["segment_ids"]}
    lang_code   = target_lang.split("-")[0].lower()

    # Global voice override from UI (if set, used for all segments)
    global_voice_name  = state.get("voice_name") or None
    global_voice_style = state.get("voice_style") or DEFAULT_STYLE

    segs_dir = get_segments_dir(state["video_id"])
    client = _client()

    to_synth = [seg for seg in segments
                if seg.dubbed_text is not None and not seg.keep_original]
    for seg in segments:
        if seg.keep_original:          # edit: "leave this line as it was"
            seg.dubbed_audio = seg.aligned_audio = None
            seg.sync_clips = []

    # One voice per speaker, decided before any synthesis.
    backend = (state.get("tts_backend") or TTS_BACKEND).lower()
    if backend == "sarvam" and not os.environ.get("SARVAM_API_KEY"):
        backend = "bodhan"
    voice_map = dict(state.get("voice_map") or {})
    swapped = []
    if lang_code not in _SARVAM_LANGS:
        # Bulbul speaks 11 languages; for the rest every Sarvam pick falls
        # back to Bodhan, which covers all 22.
        backend = "bodhan"
        for k, v in list(voice_map.items()):
            if is_sarvam(v):
                swapped.append(voice_label(v))
                voice_map.pop(k)
        if is_sarvam(global_voice_name):
            swapped.append(voice_label(global_voice_name))
            global_voice_name = None
    voice_by_speaker = _assign_voices(segments, lang_code, global_voice_name,
                                      voice_map, backend)
    for seg in segments:
        seg.voice = voice_by_speaker.get(
            seg.speaker if seg.speaker is not None else 0
        )

    # Lines with English kept in English spelling (Sarvam translation) that a
    # Bodhan voice will read: Bodhan mangles Latin, so it gets a native-script
    # copy. The stored dubbed_text keeps the English spelling.
    from .translate import transliterate_latin, has_latin, BODHAN_LATIN
    speak_text: dict[int, str] = {}
    need = [(s.id, s.dubbed_text) for s in to_synth
            if not BODHAN_LATIN and not is_sarvam(s.voice)
            and has_latin(s.dubbed_text) and lang_code != "en"]
    if need:
        speak_text = transliterate_latin(need, lang_code)

    def _synth_one(seg: Segment) -> tuple[int, str | None]:
        # An explicit edit instruction ("make it sad") beats the measured delivery.
        seg_style = (seg.style_override or
                     _resolve_style(seg.edit_instruction, _line_style(seg, global_voice_style)))
        voice = seg.voice or _pick_voice(lang_code, _UNKNOWN_GENDER_FALLBACK)

        # The cache key must include the voice and style. Keying on segment id
        # alone meant a re-run with a different voice silently returned the
        # previous take — the audio never changed however the casting did.
        text = _speakable(speak_text.get(seg.id, seg.dubbed_text))
        key = instruction_hash(
            f"{seg.edit_instruction or ''}|{voice}|{seg_style}|{text}"
        )
        audio_path = segs_dir / f"s{seg.id:04d}_dubbed_{key}.wav"

        if audio_path.exists() and seg.id not in edit_ids:
            return seg.id, str(audio_path)

        # Male voice names are taken from the published catalogue and not all
        # are confirmed against the API, so fall back rather than fail the run.
        fallbacks = [voice]
        primary = _pick_voice(lang_code, seg.gender or _UNKNOWN_GENDER_FALLBACK)
        for alt in (primary, _DEFAULT_MALE if (seg.gender or "") == "male" else _DEFAULT_FEMALE):
            if alt not in fallbacks:
                fallbacks.append(alt)

        last_exc: Exception | None = None
        for attempt, candidate in enumerate(fallbacks):
            try:
                audio = tts_bytes(text, lang_code, candidate, seg_style, client)
                with open(audio_path, "wb") as f:
                    f.write(audio)
                if attempt:
                    print(f"  ⚠ seg {seg.id}: voice '{voice}' rejected; used '{candidate}'")
                    seg.voice = candidate
                return seg.id, str(audio_path)
            except Exception as exc:
                last_exc = exc

        print(f"  ✗ seg {seg.id}: all voices failed ({last_exc})")
        return seg.id, None

    seg_map = {s.id: s for s in segments}

    def _run(batch: list[Segment]) -> int:
        done = 0
        if not batch:
            return 0
        with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as pool:
            for future in as_completed([pool.submit(_synth_one, s) for s in batch]):
                seg_id, audio_path = future.result()
                if audio_path:
                    seg_map[seg_id].dubbed_audio = audio_path
                    done += 1
        return done

    processed = _run(to_synth)

    # ── Refit: shorten lines whose real audio cannot fit their slot ──────────
    # The translate step only guessed lengths. Now the real clip lengths are
    # known, so measure this voice's actual speaking rate, give each
    # overrunning line a precise character budget, rewrite it shorter and
    # re-synthesize once. This is what keeps the dub from sounding rushed.
    refit_note = _refit_overruns(segments, state.get("audio_path"), _run, lang_code)

    styles: dict[str, int] = {}
    for s in to_synth:
        st = _resolve_style(s.edit_instruction, _line_style(s, global_voice_style))
        styles[st] = styles.get(st, 0) + 1
    notes = [
        f"✓ Synthesized {processed} segments (lang: {lang_code}) — cast: "
        + _cast_summary(segments)
        + " — styles: " + ", ".join(f"{k}×{v}" for k, v in sorted(styles.items()))
    ]
    if swapped:
        notes.append(f"⚠ Sarvam does not speak {lang_code}; used Bodhan instead of "
                     + ", ".join(sorted(set(swapped))))
    if refit_note:
        notes.append(refit_note)

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + notes,
    }


_REFIT_ROUNDS = int(os.environ.get("REFIT_ROUNDS", "2"))
# Speed-up accepted before a line is rewritten shorter (see _refit_overruns).
_REWRITE_ABOVE = float(os.environ.get("REWRITE_ABOVE_STRETCH", "1.15"))
_REWRITE_SHORT = float(os.environ.get("REWRITE_SHORT_ABOVE_STRETCH", "1.3"))
# A dub under this share of the original speech is lengthened (lines of at
# least _GROW_MIN_SECS): a lecture line had 10.7 s of speech, 6.9 s of dub.
_GROW_BELOW = float(os.environ.get("GROW_BELOW_FILL", "0.8"))
_GROW_MIN_SECS = float(os.environ.get("GROW_MIN_SECS", "1.5"))


def _refit_overruns(segments: list[Segment], audio_path: str | None, run,
                    lang_code: str) -> str:
    """
    Rewrite-and-re-voice any line whose real audio would need more than
    MAX_STRETCH speed-up, up to _REFIT_ROUNDS times. Shortening the words is
    what keeps the voice at a natural pace; speeding it up is the last resort.
    """
    from .align import MAX_STRETCH, slot_seconds, trimmed_duration, _audio_len
    from .translate import condense_to_budget, expand_to_fill
    from .. import tts_rate

    ordered = sorted(segments, key=lambda s: s.start)
    total = _audio_len(audio_path)
    notes: list[str] = []
    cps_by_voice: dict[str, float] = {}

    for rnd in range(1, _REFIT_ROUNDS + 1):
        measured: list[tuple[Segment, float, float]] = []   # seg, clip secs, slot secs
        for i, seg in enumerate(ordered):
            if seg.dubbed_audio and seg.dubbed_text:
                try:
                    measured.append((seg, trimmed_duration(seg.dubbed_audio),
                                     slot_seconds(ordered, i, total)))
                except Exception:
                    pass

        # Characters per second each voice actually produced, ignoring tiny
        # clips whose length is mostly ramp-up. Stored for the next translation.
        rates: dict[str, list[float]] = {}
        for s, d, _ in measured:
            if d >= 0.4 and len(s.dubbed_text) >= 4:
                rates.setdefault(s.voice or "?", []).append(len(s.dubbed_text) / d)
        if rnd == 1:
            tts_rate.record(lang_code, rates)
        flat = sorted(r for rs in rates.values() for r in rs)
        if len(flat) < 2:
            return ""
        cps_all = flat[len(flat) // 2]
        cps_by_voice = {v: sorted(rs)[len(rs) // 2] if len(rs) >= 2 else cps_all
                        for v, rs in rates.items()}

        jobs, over = [], []
        for seg, dur, slot in measured:
            # Rewriting to save a few percent cost meaning and tone ("still"
            # dropped, a casual goodbye turned ceremonial); a slightly
            # faster line is better. A very short reply can only get terser,
            # so it is sped up more before anyone rewrites it.
            limit = max(MAX_STRETCH, _REWRITE_SHORT if slot < 1.5 else _REWRITE_ABOVE)
            if dur / slot > limit:
                cps = cps_by_voice.get(seg.voice or "?", cps_all)
                # Aim under the cap so the re-voiced clip fits without stretching.
                budget = int(slot * cps * (0.95 if rnd == 1 else 0.85))
                jobs.append((seg.id, seg.dubbed_text, max(budget, 4)))
                over.append(seg)
        # Lines that end well before the speaker stops: lips keep moving in
        # silence. Only clearly short ones, on lines long enough to matter.
        grow, under = [], []
        for seg, dur, slot in measured:
            if (seg not in over and seg.duration >= _GROW_MIN_SECS
                    and dur < _GROW_BELOW * seg.duration):
                cps = cps_by_voice.get(seg.voice or "?", cps_all)
                grow.append({"id": seg.id, "original": seg.text, "dub": seg.dubbed_text,
                             "min_chars": int(0.9 * seg.duration * cps),
                             "max_chars": int(min(slot, 1.05 * seg.duration) * cps)})
                under.append(seg)
        if not jobs and not grow:
            notes.append(f"✓ All clips fit their slots at natural pace "
                         f"(measured {cps_all:.1f} chars/s)")
            break

        new = condense_to_budget(jobs) if jobs else {}
        longer = expand_to_fill(grow, lang_code) if grow else {}
        redo = []
        for seg in over:
            if seg.id in new:
                seg.dubbed_text = new[seg.id]
                redo.append(seg)
        for seg in under:
            if seg.id in longer:
                seg.dubbed_text = longer[seg.id]
                redo.append(seg)
        if not redo:
            if jobs:
                why = ("no OPENROUTER_API_KEY" if not os.environ.get("OPENROUTER_API_KEY")
                       else "already as short as they can naturally be")
                notes.append(f"⚠ {len(jobs)} clip(s) still too long for their slot ({why}) "
                             "— they will be sped up")
            if grow:
                notes.append(f"⚠ {len(grow)} clip(s) end early — no natural way to lengthen "
                             "them; left as they are")
            break
        run(redo)
        notes.append(f"✓ Round {rnd}: re-voiced {sum(s in over for s in redo)} over-long and "
                     f"{sum(s in under for s in redo)} too-short line(s) "
                     f"(measured {cps_all:.1f} chars/s)")
    return "\n".join(notes)


def _speakable(text: str) -> str:
    """
    Drop notation a voice cannot read. The translation step is asked to
    write formulas as words; anything left ("R^d") made the voice say
    something odd for the symbol. Letters and digits around it stay.
    """
    import re
    t = re.sub(r"[\^_{}\\|~`]", " ", text or "")
    return re.sub(r"\s{2,}", " ", t).strip()


def _cast_summary(segments: list[Segment]) -> str:
    by_spk: dict[int, tuple[str, str, int]] = {}
    for s in segments:
        spk = s.speaker if s.speaker is not None else 0
        g, v, n = by_spk.get(spk, (s.gender or "?", s.voice or "?", 0))
        by_spk[spk] = (g, v, n + 1)
    return ", ".join(f"S{k} {g}→{v} ×{n}" for k, (g, v, n) in sorted(by_spk.items()))


def _resolve_style(edit_instruction: str | None, global_style: str) -> str:
    if not edit_instruction:
        return global_style
    low = edit_instruction.lower()
    if any(k in low for k in ("news", "formal", "serious")):
        return "TV style news"
    if any(k in low for k in ("happy", "energetic", "enthusiastic", "funny", "humor")):
        return "happy"
    if any(k in low for k in ("sad", "emotional", "grief")):
        return "sad"
    if any(k in low for k in ("angry", "anger", "aggressive")):
        return "anger"
    if any(k in low for k in ("surprised", "surprise", "dramatic", "shock")):
        return "surprise"
    if any(k in low for k in ("lecture", "explain", "educational", "slow", "clear")):
        return "educational lecture"
    if any(k in low for k in ("story", "narration", "audiobook")):
        return "single person narration audiobook"
    return global_style


# ── Voice profiles: how each catalogue voice actually sounds ─────────────────
# The catalogue only gives name and gender, so "a deeper, older man" was a
# guess. Each voice's pitch is measured once from a short sample and stored;
# the edit planner and casting see deep / medium / high per voice.
_PROFILES_PATH = Path("cache") / "_voice_profiles.json"


def voice_profiles() -> dict:
    try:
        return json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_voice_profiles(lang_code: str, text: str, sample_dir: Path) -> dict:
    """Synthesize `text` with every voice, measure median pitch, store tiers."""
    import time
    import librosa
    import numpy as np
    profiles = voice_profiles()
    for voice, gender, native in voice_catalogue():
        if voice in profiles:
            continue
        path = sample_dir / f"{voice}_{lang_code}.wav"
        for attempt in range(4):
            try:
                preview_voice(voice, lang_code, text, path)
                break
            except Exception as exc:
                if "429" not in str(exc) or attempt == 3:
                    print(f"  ✗ {voice}: {exc}")
                    break
                time.sleep(4 * (attempt + 1))
        if not path.exists():
            continue
        y, sr = librosa.load(str(path), sr=16000)
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        v = f0[voiced & ~np.isnan(f0)]
        if v.size < 10:
            continue
        profiles[voice] = {"gender": gender, "native": native,
                           "f0": round(float(np.median(v)), 1)}
        time.sleep(1.5)
    # Tiers within each gender: lowest third deep, highest third high.
    for g in ("male", "female"):
        vs = sorted((p["f0"], v) for v, p in profiles.items() if p["gender"] == g)
        for k, (_, v) in enumerate(vs):
            q = k / max(1, len(vs) - 1)
            profiles[v]["pitch"] = "deep" if q < 1 / 3 else "high" if q > 2 / 3 else "medium"
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    return profiles
