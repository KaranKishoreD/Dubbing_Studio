"""
Casting: which voice each speaker gets.

One decision per speaker, so a speaker keeps the same voice all video.
Voices picked in the UI win; everyone else gets the target language's own
voice of their gender first (native pronunciation), and among candidates
the one whose measured pitch is closest to the speaker's.
"""

import os

from ..state import Segment
from .catalogue import (
    _FEMALE_VOICES,
    _LANG_ALIAS,
    _MALE_VOICES,
    ALL_VOICES,
    SARVAM_TTS_LANGS,
    SARVAM_VOICES,
    sarvam_gender,
    voice_catalogue,
    voice_profiles,
)

# ── Per-speaker casting ───────────────────────────────────────────────────────
# Bodhan ships roughly one female and one male voice per language, so a second
# same-gender speaker has to borrow a voice from another language. Any voice can
# read any supported language — the voice only sets timbre — but a voice from a
# closely related language carries the nearest phonetic habits, so prefer those.
_LANG_GROUPS: list[list[str]] = [
    ["ta", "ml", "te", "kn"],  # Dravidian
    ["hi", "mr", "gu", "pa", "ur", "mai", "doi", "sa", "ne"],  # Indo-Aryan, west/central
    ["bn", "as", "or"],  # Indo-Aryan, east
    ["ks", "sd", "kok"],  # other Indo-Aryan
    ["brx", "mni", "sat"],  # Tibeto-Burman / Munda
]


DEFAULT_FEMALE = "Kavya"

DEFAULT_MALE = "Amit"


# Used only when diarize could not measure pitch at all. Male is the safer
# default for lecture/talk content, which skews male; override with
# UNKNOWN_GENDER_FALLBACK=female or set a voice explicitly in the UI.
UNKNOWN_GENDER_FALLBACK = os.environ.get("UNKNOWN_GENDER_FALLBACK", "male")


def pick_voice(lang_code: str, gender: str) -> str:
    return _voice_pool(lang_code, gender)[0]


def _voice_pool(lang_code: str, gender: str, backend: str = "bodhan") -> list[str]:
    """
    Ordered voice candidates for a language: its own voice first, then voices
    from languages in the same family, then everything else. Distinct speakers
    take successive entries, so each gets an audibly different voice.
    Sarvam voices are not tied to a language; they are ordered by measured
    pitch from the middle out, so the first picks sound typical, not extreme.
    """
    if backend == "sarvam" and lang_code in SARVAM_TTS_LANGS:
        prof = voice_profiles()
        vs = [v for v in SARVAM_VOICES if sarvam_gender(v) == gender]
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
    return ordered or [DEFAULT_MALE if gender == "male" else DEFAULT_FEMALE]


# Usual speaking pitch per gender (Hz), for casting only.
_F0_RANGE = {"male": (80, 175), "female": (160, 300)}


def assign_voices(
    segments: list[Segment],
    lang_code: str,
    override: str | None,
    voice_map: dict | None = None,
    backend: str = "bodhan",
) -> dict[int, str]:
    """
    Map each speaker to a voice. One decision per speaker, so a speaker keeps
    the same voice throughout, and two speakers never share one unless the
    catalogue runs out.
    """
    speakers: dict[int, str] = {}  # speaker id -> gender
    talk: dict[int, float] = {}  # speaker id -> seconds spoken
    for seg in segments:
        spk = seg.speaker if seg.speaker is not None else 0
        gender = seg.gender if seg.gender in ("male", "female") else UNKNOWN_GENDER_FALLBACK
        # First gender seen for a speaker wins; diarize already made it uniform.
        speakers.setdefault(spk, gender)
        talk[spk] = talk.get(spk, 0.0) + seg.duration

    if override:
        return {spk: override for spk in speakers}

    # Voices picked in the UI win; everyone else is cast around them.
    chosen = {
        int(k): v
        for k, v in (voice_map or {}).items()
        if v and v in ALL_VOICES and str(k).lstrip("-").isdigit()
    }
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
    # normal range for their gender. Laughs and pitch-tracking errors can put
    # single lines an octave too high; the median and the clamp stop a few
    # such lines from choosing the wrong voice.
    f0s: dict[int, list[tuple[float, float]]] = {}
    for seg in segments:
        if seg.f0:
            f0s.setdefault(seg.speaker if seg.speaker is not None else 0, []).append(
                (seg.f0, max(seg.duration, 0.1))
            )
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
    native = {
        v for v, _, lang in voice_catalogue() if lang == _LANG_ALIAS.get(lang_code, lang_code)
    }

    for gender in ("male", "female"):
        pool = _voice_pool(lang_code, gender, backend)
        # Voices far outside the usual range for their gender (a "male"
        # voice at 208 Hz) are only used when nothing else is left.
        lo, hi = _F0_RANGE[gender]
        typical = [
            v
            for v in pool
            if v in native or lo <= (prof.get(v, {}).get("f0") or (lo + hi) / 2) <= hi
        ]
        pool = typical or pool
        same = [s for s, g in speakers.items() if g == gender and s not in assignment]
        for spk in sorted(same, key=lambda s: -talk[s]):
            free = [v for v in pool if v not in used] or pool
            target = spk_f0.get(spk)

            def cost(v, k):
                vf = prof.get(v, {}).get("f0")
                if not target or not vf:
                    return k  # no data: keep pool order
                # Native voices first: voice cloning replaces the timbre with
                # the real speaker's, but pronunciation stays the TTS voice's.
                return abs(12 * math.log2(vf / target)) + (0 if v in native else 24)

            voice = min(free, key=lambda v: cost(v, free.index(v)))
            used.add(voice)
            assignment[spk] = voice
    return assignment


def cast_summary(segments: list[Segment]) -> str:
    by_spk: dict[int, tuple[str, str, int]] = {}
    for s in segments:
        spk = s.speaker if s.speaker is not None else 0
        g, v, n = by_spk.get(spk, (s.gender or "?", s.voice or "?", 0))
        by_spk[spk] = (g, v, n + 1)
    return ", ".join(f"S{k} {g}→{v} ×{n}" for k, (g, v, n) in sorted(by_spk.items()))
