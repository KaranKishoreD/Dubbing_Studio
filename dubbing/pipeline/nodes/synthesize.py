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


def _voice_pool(lang_code: str, gender: str) -> list[str]:
    """
    Ordered voice candidates for a language: its own voice first, then voices
    from languages in the same family, then everything else. Distinct speakers
    take successive entries, so each gets an audibly different voice.
    """
    table = _MALE_VOICES if gender == "male" else _FEMALE_VOICES
    lang_code = _LANG_ALIAS.get(lang_code, lang_code)
    group = next((g for g in _LANG_GROUPS if lang_code in g), [])

    ordered: list[str] = []
    for key in [lang_code, *group, *table.keys()]:
        for v in table.get(key, []):
            if v not in ordered:
                ordered.append(v)
    return ordered or [_DEFAULT_MALE if gender == "male" else _DEFAULT_FEMALE]


def _assign_voices(
    segments: list[Segment], lang_code: str, override: str | None
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

    # Deal voices out per gender so same-gender speakers get different ones.
    # Whoever talks most gets the target language's own voice — they carry
    # the video, so they should sound the most native.
    used: set[str] = set()
    assignment: dict[int, str] = {}
    for gender in ("male", "female"):
        pool = _voice_pool(lang_code, gender)
        i = 0
        same = [s for s, g in speakers.items() if g == gender]
        for spk in sorted(same, key=lambda s: -talk[s]):
            while i < len(pool) and pool[i] in used:
                i += 1
            voice = pool[i] if i < len(pool) else pool[-1]
            used.add(voice)
            assignment[spk] = voice
            i += 1
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

    to_synth = [seg for seg in segments if seg.dubbed_text is not None]

    # One voice per speaker, decided before any synthesis.
    voice_by_speaker = _assign_voices(segments, lang_code, global_voice_name)
    for seg in segments:
        seg.voice = voice_by_speaker.get(
            seg.speaker if seg.speaker is not None else 0
        )

    def _synth_one(seg: Segment) -> tuple[int, str | None]:
        # An explicit edit instruction ("make it sad") beats the measured delivery.
        seg_style = _resolve_style(seg.edit_instruction, _line_style(seg, global_voice_style))
        voice = seg.voice or _pick_voice(lang_code, _UNKNOWN_GENDER_FALLBACK)

        # The cache key must include the voice and style. Keying on segment id
        # alone meant a re-run with a different voice silently returned the
        # previous take — the audio never changed however the casting did.
        key = instruction_hash(
            f"{seg.edit_instruction or ''}|{voice}|{seg_style}|{seg.dubbed_text}"
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
                response = client.audio.speech.create(
                    model="indic-speak",
                    input=seg.dubbed_text,
                    voice=candidate,
                    extra_body={"instructions": json.dumps(
                        {"lang": lang_code, "style": seg_style}
                    )},
                )
                with open(audio_path, "wb") as f:
                    f.write(response.content)
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
    refit_note = _refit_overruns(segments, state.get("audio_path"), _run)

    styles: dict[str, int] = {}
    for s in to_synth:
        st = _resolve_style(s.edit_instruction, _line_style(s, global_voice_style))
        styles[st] = styles.get(st, 0) + 1
    notes = [
        f"✓ Synthesized {processed} segments (lang: {lang_code}) — cast: "
        + _cast_summary(segments)
        + " — styles: " + ", ".join(f"{k}×{v}" for k, v in sorted(styles.items()))
    ]
    if refit_note:
        notes.append(refit_note)

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + notes,
    }


def _refit_overruns(segments: list[Segment], audio_path: str | None, run) -> str:
    from .align import MAX_STRETCH, slot_seconds, trimmed_duration, _audio_len
    from .translate import condense_to_budget

    ordered = sorted(segments, key=lambda s: s.start)
    total = _audio_len(audio_path)

    measured: list[tuple[Segment, float, float]] = []   # seg, clip secs, slot secs
    for i, seg in enumerate(ordered):
        if seg.dubbed_audio and seg.dubbed_text:
            try:
                measured.append((seg, trimmed_duration(seg.dubbed_audio),
                                 slot_seconds(ordered, i, total)))
            except Exception:
                pass

    # Characters per second this voice actually produced, ignoring tiny clips
    # whose length is mostly ramp-up.
    rates = [len(s.dubbed_text) / d for s, d, _ in measured
             if d >= 0.4 and len(s.dubbed_text) >= 4]
    if len(rates) < 2:
        return ""
    cps = float(sorted(rates)[len(rates) // 2])

    jobs, over = [], []
    for seg, dur, slot in measured:
        if dur / slot > MAX_STRETCH:
            # Aim slightly under the cap so the re-synthesized clip fits.
            budget = int(slot * MAX_STRETCH * cps * 0.92)
            jobs.append((seg.id, seg.dubbed_text, max(budget, 4)))
            over.append(seg)
    if not jobs:
        return f"✓ All clips fit their slots (measured {cps:.1f} chars/s)"

    new = condense_to_budget(jobs)
    if not new:
        return (f"⚠ {len(jobs)} clip(s) too long for their slot and could not be "
                "shortened (no OpenRouter key?) — they will be trimmed")
    redo = []
    for seg in over:
        if seg.id in new:
            seg.dubbed_text = new[seg.id]
            redo.append(seg)
    run(redo)
    return (f"✓ Shortened and re-voiced {len(redo)}/{len(jobs)} over-long line(s) "
            f"(measured {cps:.1f} chars/s)")


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
