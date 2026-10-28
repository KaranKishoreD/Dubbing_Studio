"""
Fixed choices shown in the studio: target languages, content domains, the
pipeline stages in the progress tracker, and the voice list for casting.
"""

import os

from pipeline.text.languages import LANG_NAMES
from pipeline.voice.catalogue import is_sarvam, voice_catalogue, voice_label, voice_profiles
from pipeline.voice.styles import BODHAN_STYLES

SUPPORTED_LANGS = {
    "Hindi (हिन्दी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Assamese (অসমীয়া)": "as",
    "Urdu (اردو)": "ur",
    "English": "en",
}


LANG_LABEL = {v: k for k, v in SUPPORTED_LANGS.items()}

DOMAINS = [
    "education / lecture",
    "technology / CS",
    "general",
    "medicine",
    "law",
    "finance",
]


# ── Pipeline stages, as the customer sees them ────────────────────────────────
FULL_STAGES = [
    ("download", "Download"),
    ("separate", "Separate"),
    ("transcribe", "Transcribe"),
    ("diarize", "Speakers"),
    ("refine", "Clean up"),
    ("translate", "Translate"),
    ("synthesize", "Voice"),
    ("align", "Fit timing"),
    ("lipsync", "Lip sync"),
    ("assemble", "Export"),
]


EDIT_STAGES = FULL_STAGES[5:]

REVOICE_STAGES = FULL_STAGES[6:]


# ── Voice casting ─────────────────────────────────────────────────────────────
# One row per detected speaker. Six covers interviews and panels; more than
# that and diarization itself is the weak link, not the UI.
MAX_CAST_ROWS = 12

AUTO_VOICE = "Auto"


CATALOGUE = voice_catalogue()  # 23 female + 22 male

_PROFILES = voice_profiles()  # measured pitch per voice


def _choice(v: str, g: str, lang: str) -> tuple[str, str]:
    pitch = _PROFILES.get(v, {}).get("pitch")
    where = "Sarvam" if is_sarvam(v) else f"Bodhan · {LANG_NAMES.get(lang, lang)}"
    return (
        f"{voice_label(v).replace(' (Sarvam)', '')} — {g}"
        + (f" · {pitch}" if pitch else "")
        + f" · {where}",
        v,
    )


_BY_PITCH = sorted(CATALOGUE, key=lambda x: _PROFILES.get(x[0], {}).get("f0", 0))

VOICE_CHOICES = [AUTO_VOICE] + [
    _choice(v, g, l)
    for engine in ("sarvam", "bodhan")
    for g_order in ("female", "male")
    for v, g, l in _BY_PITCH
    if g == g_order and is_sarvam(v) == (engine == "sarvam")
]


HAS_SARVAM = bool(os.environ.get("SARVAM_API_KEY"))

# A short line per target language for voice previews.
PREVIEW_TEXT = {
    "hi": "नमस्ते, मैं इस वीडियो की आवाज़ हूँ।",
    "ta": "வணக்கம், நான் இந்த வீடியோவின் குரல்.",
    "te": "నమస్కారం, నేను ఈ వీడియో గొంతు.",
    "kn": "ನಮಸ್ಕಾರ, ನಾನು ಈ ವೀಡಿಯೊದ ಧ್ವನಿ.",
    "ml": "നമസ്കാരം, ഞാനാണ് ഈ വീഡിയോയുടെ ശബ്ദം.",
    "bn": "নমস্কার, আমি এই ভিডিওর কণ্ঠ।",
    "mr": "नमस्कार, मी या व्हिडिओचा आवाज आहे.",
    "gu": "નમસ્તે, હું આ વિડિયોનો અવાજ છું.",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਇਸ ਵੀਡੀਓ ਦੀ ਆਵਾਜ਼ ਹਾਂ।",
    "or": "ନମସ୍କାର, ମୁଁ ଏହି ଭିଡିଓର ସ୍ୱର।",
    "as": "নমস্কাৰ, মই এই ভিডিঅ'ৰ কণ্ঠ।",
    "ur": "السلام علیکم، میں اس ویڈیو کی آواز ہوں۔",
    "en": "Hello, I am the voice of this video.",
}


STYLES = sorted({v for v in BODHAN_STYLES.values() if v != "auto"})
