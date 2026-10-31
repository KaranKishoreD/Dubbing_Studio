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
    "Nepali (नेपाली)": "ne",
    "Sanskrit (संस्कृतम्)": "sa",
    "Maithili (मैथिली)": "mai",
    "Konkani (कोंकणी)": "kok",
    "Dogri (डोगरी)": "doi",
    "Kashmiri (کٲشُر)": "ks",
    "Sindhi (سنڌي)": "sd",
    "Manipuri (মৈতৈলোন্)": "mni",
    "Bodo (बड़ो)": "brx",
    "Santali (ᱥᱟᱱᱛᱟᱲᱤ)": "sat",
    "English": "en",
}


# International mode: LLM translation and Kokoro's open voices.
INTERNATIONAL_LANGS = {
    "English": "en",
    "Spanish (Español)": "es",
    "French (Français)": "fr",
    "Italian (Italiano)": "it",
    "Portuguese (Português)": "pt",
    "Japanese (日本語)": "ja",
    "Chinese (中文)": "zh",
    "German (Deutsch)": "de",
    "Russian (Русский)": "ru",
    "Arabic (العربية)": "ar",
    "Korean (한국어)": "ko",
    "Turkish (Türkçe)": "tr",
    "Dutch (Nederlands)": "nl",
    "Polish (Polski)": "pl",
    "Ukrainian (Українська)": "uk",
    "Vietnamese (Tiếng Việt)": "vi",
    "Indonesian (Bahasa Indonesia)": "id",
    "Thai (ไทย)": "th",
    "Persian (فارسی)": "fa",
    "Hebrew (עברית)": "he",
    "Greek (Ελληνικά)": "el",
    "Swedish (Svenska)": "sv",
    "Norwegian (Norsk)": "no",
    "Danish (Dansk)": "da",
    "Finnish (Suomi)": "fi",
    "Czech (Čeština)": "cs",
    "Slovak (Slovenčina)": "sk",
    "Hungarian (Magyar)": "hu",
    "Romanian (Română)": "ro",
    "Bulgarian (Български)": "bg",
    "Serbian (Српски)": "sr",
    "Slovenian (Slovenščina)": "sl",
    "Catalan (Català)": "ca",
    "Basque (Euskara)": "eu",
    "Welsh (Cymraeg)": "cy",
    "Icelandic (Íslenska)": "is",
    "Estonian (Eesti)": "et",
    "Latvian (Latviešu)": "lv",
    "Lithuanian (Lietuvių)": "lt",
    "Georgian (ქართული)": "ka",
    "Armenian (Հայերեն)": "hy",
    "Kazakh (Қазақ)": "kk",
    "Swahili (Kiswahili)": "sw",
    "Albanian (Shqip)": "sq",
    "Luxembourgish (Lëtzebuergesch)": "lb",
    "Kurdish (Kurdî)": "ku",
}

LANG_LABEL = {v: k for k, v in {**INTERNATIONAL_LANGS, **SUPPORTED_LANGS}.items()}

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

LIPSYNC_STAGES = FULL_STAGES[8:]


def fast_stages(stages: list) -> list:
    """The same stages in fast mode: lip sync only reuses finished shots."""
    return [(k, "Lip sync (off)" if k == "lipsync" else label) for k, label in stages]


# ── Voice casting ─────────────────────────────────────────────────────────────
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
    "ne": "नमस्ते, म यो भिडियोको आवाज हुँ।",
    "sa": "नमस्ते, अहम् अस्य चलचित्रस्य स्वरः अस्मि।",
    "mai": "प्रणाम, हम एहि वीडियोक आवाज छी।",
    "kok": "नमस्कार, हांव ह्या व्हिडियोचो आवाज.",
    "doi": "नमस्ते, में इस वीडियो दी आवाज़ आं।",
    "ks": "السلام علیکم، بہٕ چھُس یتھ ویڈیوکھ آواز۔",
    "sd": "السلام عليڪم، مان هن وڊيو جو آواز آهيان.",
    "mni": "খুরুমজরি, ঐহাক্না ভিডিও অসিগী খোনজেলনি।",
    "brx": "खुलुमबाय, आं बे भिडियोनि गाब।",
    "sat": "ᱡᱚᱦᱟᱨ",
    "en": "Hello, I am the voice of this video.",
    "es": "Hola, soy la voz de este video.",
    "fr": "Bonjour, je suis la voix de cette vidéo.",
    "it": "Ciao, sono la voce di questo video.",
    "pt": "Olá, eu sou a voz deste vídeo.",
    "ja": "こんにちは、この動画の声です。",
    "zh": "你好，我是这个视频的声音。",
}


STYLES = sorted({v for v in BODHAN_STYLES.values() if v != "auto"})
