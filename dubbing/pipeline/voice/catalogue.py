"""
Which voices exist, and what is known about them.

Two engines:
  Bodhan indic-speak  ~45 voices covering all 22 scheduled Indian languages
  Sarvam Bulbul v3    37 voices for 11 languages (ids prefixed "sarvam:")

`voice_profiles()` holds each voice's measured pitch (from
tools/build_voice_profiles.py), used to cast a voice close to a speaker.
"""

import json
import os
from pathlib import Path

# Lists, not single names: a language may have more than one voice (Hindi has
# two female voices), and every one of the 45 must be reachable when casting.
# First entry is the language's primary voice.
_FEMALE_VOICES: dict[str, list[str]] = {
    "hi": ["Kavya", "Suhani"],
    "as": ["Prastuti"],
    "bn": ["Ishita"],
    "brx": ["Gwrbw"],
    "doi": ["Preeti"],
    "gu": ["Dhara"],
    "kn": ["Deepika"],
    "kok": ["Anjali"],
    "ks": ["Zoon"],
    "mai": ["Vaidehi"],
    "ml": ["Lakshmi"],
    "mni": ["Thoibi"],
    "mr": ["Anagha"],
    "ne": ["Srijana"],
    "or": ["Itishree"],
    "pa": ["Kaur"],
    "sa": ["Bharati"],
    "sat": ["Phulmani"],
    "sd": ["Moomal"],
    "ta": ["Anitha"],
    "te": ["Sravani"],
    "ur": ["Saba"],
}


_MALE_VOICES: dict[str, list[str]] = {
    "hi": ["Amit"],
    "as": ["Ankur"],
    "bn": ["Sourav"],
    "brx": ["Sansuma"],
    "doi": ["Sham"],
    "gu": ["Parth"],
    "kn": ["Adarsh"],
    "kok": ["Sandeep"],
    "ks": ["Ishfaq"],
    "mai": ["Madhukar"],
    "ml": ["Kiran"],
    "mni": ["Chaoba"],
    "mr": ["Chinmay"],
    "ne": ["Sagar"],
    "or": ["Akash"],
    "pa": ["Manpreet"],
    "sa": ["Aryaman"],
    "sat": ["Sibu"],
    "sd": ["Rano"],
    "ta": ["Arun"],
    "te": ["Vamsi"],
    "ur": ["Zaid"],
}


# English source/target reads best with the Hindi voices.
_LANG_ALIAS = {"en": "hi"}

ALL_FEMALE_VOICES = [v for vs in _FEMALE_VOICES.values() for v in vs]


ALL_MALE_VOICES = [v for vs in _MALE_VOICES.values() for v in vs]

ALL_BODHAN_VOICES = ALL_FEMALE_VOICES + ALL_MALE_VOICES


# ── Sarvam Bulbul v3 ──────────────────────────────────────────────────────────
# A second voice engine (SARVAM_API_KEY). Every Bulbul voice speaks every
# supported language, so the pool is not per language. Ids carry a "sarvam:"
# prefix so a cast can mix engines and the cache key separates them. Gender
# is from the voice name, confirmed by measured pitch (voice_profiles).
SARVAM_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")

_SARVAM_FEMALE = [
    "ritu",
    "priya",
    "neha",
    "pooja",
    "simran",
    "kavya",
    "ishita",
    "shreya",
    "roopa",
    "tanya",
    "shruti",
    "suhani",
    "kavitha",
    "rupali",
]


_SARVAM_MALE = [
    "aditya",
    "ashutosh",
    "rahul",
    "rohan",
    "amit",
    "dev",
    "ratan",
    "varun",
    "manan",
    "sumit",
    "kabir",
    "aayan",
    "shubh",
    "advait",
    "anand",
    "tarun",
    "sunny",
    "mani",
    "gokul",
    "vijay",
    "mohit",
    "rehan",
    "soham",
]


SARVAM_VOICES = [f"sarvam:{v}" for v in _SARVAM_FEMALE + _SARVAM_MALE]

# Languages Bulbul speaks (BCP-47 with -IN).
SARVAM_TTS_LANGS = {"hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "en"}

# Kokoro-82M (International mode): open-source, runs on the GPU. Voices per
# language and gender; ids prefixed "kokoro:". kokoro_lang gives Kokoro's
# own language code for each.
_KOKORO = {
    "en": (
        {
            "female": ["af_heart", "af_bella", "bf_emma"],
            "male": ["am_michael", "am_fenrir", "bm_george"],
        },
        "a",
    ),
    "es": ({"female": ["ef_dora"], "male": ["em_alex", "em_santa"]}, "e"),
    "fr": ({"female": ["ff_siwis"], "male": []}, "f"),
    "it": ({"female": ["if_sara"], "male": ["im_nicola"]}, "i"),
    "pt": ({"female": ["pf_dora"], "male": ["pm_alex", "pm_santa"]}, "p"),
    "ja": ({"female": ["jf_alpha", "jf_gongitsune"], "male": ["jm_kumo"]}, "j"),
    "zh": ({"female": ["zf_xiaobei", "zf_xiaoni"], "male": ["zm_yunjian", "zm_yunxi"]}, "z"),
}
KOKORO_LANGS = set(_KOKORO)
KOKORO_VOICES = [f"kokoro:{v}" for t, _ in _KOKORO.values() for vs in t.values() for v in vs]


def kokoro_pool(lang_code: str, gender: str) -> list[str]:
    """Kokoro voices for a language and gender (the other gender if it has none)."""
    table, _ = _KOKORO.get(lang_code, _KOKORO["en"])
    vs = table.get(gender) or table.get("female" if gender == "male" else "male") or []
    return [f"kokoro:{v}" for v in vs]


def kokoro_gender(voice: str) -> str:
    """Kokoro voice names start with the language letter, then f/m."""
    return "male" if voice.split(":", 1)[1][1] == "m" else "female"


def kokoro_lang(lang_code: str) -> str:
    return _KOKORO.get(lang_code, _KOKORO["en"])[1]


def is_kokoro(voice: str | None) -> bool:
    return bool(voice) and voice.startswith("kokoro:")


ALL_VOICES = ALL_BODHAN_VOICES + SARVAM_VOICES + KOKORO_VOICES

# Languages dubbed only in International mode (no Indian voice speaks them).
INTERNATIONAL_ONLY = {
    "es",
    "fr",
    "it",
    "pt",
    "ja",
    "zh",
    "ar",
    "bg",
    "ca",
    "cs",
    "cy",
    "da",
    "de",
    "el",
    "et",
    "eu",
    "fa",
    "fi",
    "he",
    "hu",
    "hy",
    "id",
    "is",
    "ka",
    "kk",
    "ko",
    "ku",
    "lb",
    "lt",
    "lv",
    "nl",
    "no",
    "pl",
    "ro",
    "ru",
    "sk",
    "sl",
    "sq",
    "sr",
    "sv",
    "sw",
    "th",
    "tr",
    "uk",
    "vi",
}


# "bodhan" | "sarvam" — which engine automatic casting draws from.
TTS_BACKEND = os.environ.get("TTS_BACKEND", "bodhan").lower()


def is_sarvam(voice: str | None) -> bool:
    return bool(voice) and voice.startswith("sarvam:")


def voice_label(voice: str | None) -> str:
    """Display name: 'Anand (Sarvam)' / 'Arun'."""
    if not voice:
        return "—"
    if is_kokoro(voice):
        return f"{voice.split(':', 1)[1].split('_', 1)[1].title()} (Kokoro)"
    if voice.startswith("piper:"):
        return f"{voice.split('-')[1].replace('_', ' ').title()} (Piper)"
    return f"{voice.split(':', 1)[1].title()} (Sarvam)" if is_sarvam(voice) else voice


def sarvam_gender(voice: str) -> str:
    return "female" if voice.split(":", 1)[1] in _SARVAM_FEMALE else "male"


def voice_catalogue() -> list[tuple[str, str, str]]:
    """[(voice id, "female"|"male", native language code or "any")] — Bodhan's
    45 voices plus Sarvam's 37."""
    out = []
    for gender, table in (("female", _FEMALE_VOICES), ("male", _MALE_VOICES)):
        for lang, voices in table.items():
            out += [(v, gender, lang) for v in voices]
    out += [(v, sarvam_gender(v), "any") for v in SARVAM_VOICES]
    for lang, (table, _) in _KOKORO.items():
        out += [(f"kokoro:{v}", g, lang) for g, vs in table.items() for v in vs]
    return out


# ── Voice profiles: how each catalogue voice actually sounds ─────────────────
# The catalogue only gives name and gender, which is not enough to pick "a
# deeper, older man". Each voice's pitch is measured once from a short sample and stored;
# the edit planner and casting see deep / medium / high per voice.
_PROFILES_PATH = Path("cache") / "_voice_profiles.json"


def voice_profiles() -> dict:
    try:
        return json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}
