"""
Piper voices (International mode): open-source TTS for 53 languages
(rhasspy/piper-voices on Hugging Face). Kokoro covers English and six world
languages with more natural voices; Piper covers everything else Kokoro does
not, and French men (Kokoro has only a female French voice).

Voices are downloaded on first use into dubbing/voices/piper/ (~60 MB each).
Piper's catalogue has no gender, so each voice's gender is measured once from
its speaking pitch on a sample and saved in voices/piper/genders.json.
Voice ids are "piper:<key>", e.g. "piper:fr_FR-tom-medium".
"""

import io
import json
import threading
import urllib.request
import wave
from pathlib import Path

_DIR = Path(__file__).resolve().parents[2] / "voices" / "piper"
_BASE = "https://huggingface.co/rhasspy/piper-voices/resolve/main/"
_QUALITY = {"high": 0, "medium": 1, "low": 2, "x_low": 3}
_PER_LANG = 4  # candidate voices per language (distinct speakers when possible)
_LOCK = threading.Lock()
_LOADED: dict = {}


def catalogue() -> dict:
    """Piper's voices.json, downloaded once."""
    path = _DIR / "voices.json"
    if not path.exists():
        _DIR.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(_BASE + "voices.json", path)
    return json.loads(path.read_text(encoding="utf-8"))


def languages() -> set[str]:
    """Language codes (ISO 639-1, e.g. "de") Piper has voices for."""
    try:
        return {v["language"]["family"] for v in catalogue().values()}
    except Exception:
        return set()


def candidates(lang: str) -> list[str]:
    """Up to _PER_LANG voice keys for a language: single-speaker voices first,
    best quality first, one per speaker name."""
    vs = [(k, v) for k, v in catalogue().items() if v["language"]["family"] == lang]
    vs.sort(key=lambda kv: (kv[1]["num_speakers"] > 1, _QUALITY.get(kv[1]["quality"], 9), kv[0]))
    out, names = [], set()
    for k, v in vs:
        if v["name"] in names:
            continue
        names.add(v["name"])
        out.append(k)
    return out[:_PER_LANG]


def _ensure(key: str) -> Path:
    model = _DIR / f"{key}.onnx"
    if not model.exists():
        files = catalogue()[key]["files"]
        for rel in files:
            if rel.endswith((".onnx", ".onnx.json")):
                dst = _DIR / Path(rel).name
                urllib.request.urlretrieve(_BASE + rel, dst)
    return model


def _voice(key: str):
    from piper import PiperVoice

    if key not in _LOADED:
        _LOADED[key] = PiperVoice.load(str(_ensure(key)))
    return _LOADED[key]


def tts(text: str, voice_id: str) -> bytes:
    """WAV bytes from a Piper voice ("piper:<key>")."""
    key = voice_id.split(":", 1)[1]
    with _LOCK:
        v = _voice(key)
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            v.synthesize_wav(text, w)
    return buf.getvalue()


def gender(key: str) -> str:
    """ "male" or "female" for a Piper voice, from the same wav2vec2 voice
    classifiers used for speakers (pitch alone mislabelled low female voices),
    on a short sample; cached in genders.json."""
    path = _DIR / "genders.json"
    try:
        known = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        known = {}
    if key in known:
        return known[key]
    import librosa
    import soundfile as sf

    from ..speech.gender import classify_genders
    from ..state import Segment

    text = "Hello, one two three four five. This is a short sample of my speaking voice."
    y, sr = sf.read(io.BytesIO(tts(text, f"piper:{key}")), dtype="float32")
    y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    seg = Segment.from_dict(
        {"id": 0, "start": 0.0, "end": len(y) / 16000, "speaker": 0, "text": "", "lang": "en"}
    )
    p_female = classify_genders(y, [seg]).get(0, 0.5)
    known[key] = "female" if p_female >= 0.5 else "male"
    path.write_text(json.dumps(known, indent=1), encoding="utf-8")
    return known[key]


def pool(lang: str, want: str) -> list[str]:
    """Voice ids for a language, the wanted gender first."""
    keys = candidates(lang)
    same = [k for k in keys if gender(k) == want]
    return [f"piper:{k}" for k in same + [k for k in keys if k not in same]]
