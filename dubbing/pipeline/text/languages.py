"""
Language facts used across the pipeline: names, scripts, which engine can
read English words in English spelling, and how much English to keep.
"""

import os

LANG_NAMES = {
    "hi": "Hindi",
    "ta": "Tamil",
    "te": "Telugu",
    "kn": "Kannada",
    "ml": "Malayalam",
    "bn": "Bengali",
    "mr": "Marathi",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "ur": "Urdu",
    "ne": "Nepali",
    "sa": "Sanskrit",
    "mai": "Maithili",
    "kok": "Konkani",
    "doi": "Dogri",
    "ks": "Kashmiri",
    "sd": "Sindhi",
    "mni": "Manipuri",
    "brx": "Bodo",
    "sat": "Santali",
    "en": "English",
    # International mode (open voices: Kokoro, Piper)
    "es": "Spanish",
    "fr": "French",
    "it": "Italian",
    "pt": "Portuguese",
    "ja": "Japanese",
    "zh": "Chinese",
    "de": "German",
    "ru": "Russian",
    "ar": "Arabic",
    "ko": "Korean",
    "tr": "Turkish",
    "nl": "Dutch",
    "pl": "Polish",
    "uk": "Ukrainian",
    "vi": "Vietnamese",
    "id": "Indonesian",
    "th": "Thai",
    "fa": "Persian",
    "he": "Hebrew",
    "el": "Greek",
    "sv": "Swedish",
    "no": "Norwegian",
    "da": "Danish",
    "fi": "Finnish",
    "cs": "Czech",
    "sk": "Slovak",
    "hu": "Hungarian",
    "ro": "Romanian",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "sl": "Slovenian",
    "ca": "Catalan",
    "eu": "Basque",
    "cy": "Welsh",
    "is": "Icelandic",
    "et": "Estonian",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "ka": "Georgian",
    "hy": "Armenian",
    "kk": "Kazakh",
    "sw": "Swahili",
    "sq": "Albanian",
    "lb": "Luxembourgish",
    "ku": "Kurdish",
}


# Script for English words that stay English. Bodhan mispronounces Latin
# text, so for it they are transliterated; Sarvam Bulbul reads code-mixed
# text, so there they stay in Latin script and sound like English.
_SCRIPT_NATIVE = (
    "BUT write every word in {lang} script — transliterate loanwords and names\n"
    '   (e.g. "job" → the {lang}-script spelling of job). No Latin letters at all:\n'
    "   the text-to-speech voice reads {lang} script and mispronounces Latin."
)


_SCRIPT_LATIN = (
    "Words that stay English are written in normal English spelling "
    '(Latin letters), e.g. "office", "weekend", "actually" — the voice '
    "pronounces them as English. Everything else is in {lang} script. "
    "Never write {lang} words in Latin letters."
)


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
    "standard": (
        "Keep names, brands, places, numbers and common English loanwords "
        "that {lang} speakers actually say in English (job, office, weekend, "
        "team...),"
    ),
    "pure": (
        "PURE {lang}. Translate everything; keep English only for names, "
        "brands and technical terms that have no {lang} equivalent."
    ),
    "natural": (
        "NATURAL SPOKEN {lang}, the way a modern {lang} speaker really talks. "
        "Keep the English words such speakers keep in everyday conversation — "
        "common loanwords (office, meeting, weekend, phone, plan, problem, "
        "sorry, okay, actually, basically, seriously), technology and brand "
        "terms, and English the source speaker uses as a set phrase. Never "
        "replace a common English word with a rare pure-{lang} coinage nobody "
        "says. Grammar, verbs and connecting words stay {lang}. If a speaker "
        "already mixes {lang} and English, keep their level of mixing."
    ),
    "heavy": (
        "CODE-MIXED speech ({lang} + English, like urban young speakers or "
        "vloggers): keep many English words and short English phrases; {lang} "
        "supplies the grammar, verb endings and connectors."
    ),
}


# Unicode blocks each target language is written in.
_DEVANAGARI = [(0x0900, 0x097F), (0xA8E0, 0xA8FF)]


_ARABIC = [(0x0600, 0x06FF), (0x0750, 0x077F), (0xFB50, 0xFDFF), (0xFE70, 0xFEFF)]


_BENGALI = [(0x0980, 0x09FF)]


_SCRIPTS: dict[str, list[tuple[int, int]]] = {
    "hi": _DEVANAGARI,
    "mr": _DEVANAGARI,
    "ne": _DEVANAGARI,
    "sa": _DEVANAGARI,
    "mai": _DEVANAGARI,
    "doi": _DEVANAGARI,
    "kok": _DEVANAGARI,
    "brx": _DEVANAGARI,
    "bn": _BENGALI,
    "as": _BENGALI,
    "mni": _BENGALI + [(0xABC0, 0xABFF)],
    "ta": [(0x0B80, 0x0BFF)],
    "te": [(0x0C00, 0x0C7F)],
    "kn": [(0x0C80, 0x0CFF)],
    "ml": [(0x0D00, 0x0D7F)],
    "gu": [(0x0A80, 0x0AFF)],
    "pa": [(0x0A00, 0x0A7F)],
    "or": [(0x0B00, 0x0B7F)],
    "ur": _ARABIC,
    "ks": _ARABIC,
    "sd": _ARABIC,
    "sat": [(0x1C50, 0x1C7F)],
}


def script_ok(text: str, lang_code: str, latin: bool = False) -> bool:
    """True if every letter in `text` belongs to the target language's script
    (or is basic Latin, when English words may stay in English spelling)."""
    import unicodedata

    blocks = _SCRIPTS.get(lang_code)
    if not blocks:
        return True  # e.g. English target: nothing to check
    if latin:
        blocks = blocks + [(0x41, 0x5A), (0x61, 0x7A)]
    for ch in text:
        if unicodedata.category(ch)[0] not in ("L", "M"):
            continue  # digits, punctuation, spaces are fine
        cp = ord(ch)
        if not any(a <= cp <= b for a, b in blocks):
            return False
    return True


def has_latin(text: str | None) -> bool:
    import re

    return bool(text) and bool(re.search(r"[A-Za-z]", text))
