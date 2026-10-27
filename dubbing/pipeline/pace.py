"""
Speaker pace: dub each person at their own tempo.

A slow lecturer and a fast chatty speaker were both read in one style
("educational lecture"), and length was the only check, so a short line
could fill its time yet sound much slower than the person on screen (24%
slower on a lecture line, measured). Here:

  factor(speaker)  syllables per second of SPEAKING time (Whisper word
                   timestamps, pauses excluded) relative to typical speech.
                   Used to scale how many characters each line may hold.
  style(speaker)   the Bodhan style whose measured speed for that voice is
                   closest to the speaker's tempo. One style per speaker for
                   the whole video, so the voice never changes mid-talk.

Syllables are counted from the source text: vowel groups for Latin script,
letters not silenced by a virama for Indic scripts.
"""

import json
import os
import re
from pathlib import Path

# Typical articulation rate (syllables per second of speech, pauses
# excluded) for conversational speech; a factor of 1.0 is "normal".
_NORMAL_SYLL_PER_SEC = float(os.environ.get("NORMAL_SYLL_PER_SEC", "5.0"))
_FACTOR_RANGE = (0.75, 1.3)
# Neutral styles that differ mainly in tempo; emotional ones are not picked
# for pace (an edit or the measured delivery can still choose them).
PACE_STYLES = ("single person narration audiobook", "educational lecture",
               "Customer Care", "news", "TV style news", "advertisements")
_RATES_PATH = Path("cache") / "_style_rates.json"
# Viramas (vowel killers) of the Indic scripts.
_VIRAMA = set("\u094d\u09cd\u0a4d\u0acd\u0b4d\u0bcd\u0c4d\u0ccd\u0d4d")


def syllables(word: str) -> int:
    w = word.lower()
    if re.search(r"[a-z]", w):
        groups = re.findall(r"[aeiouy]+", w)
        n = len(groups) - (1 if w.endswith("e") and len(groups) > 1 else 0)
        return max(1, n)
    # Indic: one syllable per consonant or independent vowel letter that is
    # not silenced by a following virama (नमस्ते → न म स्ते = 3).
    import unicodedata
    n = 0
    for i, c in enumerate(w):
        if unicodedata.category(c) == "Lo" and not (i + 1 < len(w) and w[i + 1] in _VIRAMA):
            n += 1
    return n


def line_rate(seg) -> float | None:
    """Syllables per second of speaking time, from word timestamps."""
    words = [w for w in (seg.words or []) if w.get("end", 0) > w.get("start", 0)]
    if len(words) < 3:
        return None
    talk = sum(w["end"] - w["start"] for w in words)
    syl = sum(syllables(w.get("word", "")) for w in words)
    return syl / talk if talk > 0.3 else None


def speaker_factors(segments) -> dict[int, float]:
    """{speaker: tempo factor}, duration-weighted over their lines."""
    by: dict[int, list[tuple[float, float]]] = {}
    for s in segments:
        r = line_rate(s)
        if r:
            by.setdefault(s.speaker or 0, []).append((r, s.duration))
    out = {}
    for spk, v in by.items():
        v.sort()
        half, acc = sum(w for _, w in v) / 2, 0.0
        for r, w in v:
            acc += w
            if acc >= half:
                f = r / _NORMAL_SYLL_PER_SEC
                out[spk] = min(max(f, _FACTOR_RANGE[0]), _FACTOR_RANGE[1])
                break
    return out


def style_rates() -> dict:
    try:
        return json.loads(_RATES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def measure_styles(voice: str, lang_code: str, sample: str, tts) -> dict[str, float]:
    """
    Characters per second of each PACE_STYLES style for this voice, from one
    sample sentence, cached across runs (cache/_style_rates.json).
    tts(text, lang, voice, style) -> wav bytes.
    """
    import io
    import soundfile as sf
    from .nodes.align import _load_trimmed
    rates = style_rates()
    out = {}
    for style in PACE_STYLES:
        key = f"{lang_code}|{voice}|{style}"
        if key not in rates:
            try:
                y, sr = sf.read(io.BytesIO(tts(sample, lang_code, voice, style)),
                                dtype="float32", always_2d=True)
                tmp = _RATES_PATH.parent / "_style_sample.wav"
                sf.write(str(tmp), y.mean(axis=1), sr)
                yt, sr = _load_trimmed(str(tmp))
                tmp.unlink(missing_ok=True)
                if len(yt) > sr * 0.5:
                    rates[key] = round(len(sample) / (len(yt) / sr), 2)
            except Exception as exc:
                print(f"  ⚠ style speed for {voice}/{style}: {exc}")
                continue
        if key in rates:
            out[style] = rates[key]
    _RATES_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RATES_PATH.write_text(json.dumps(rates, ensure_ascii=False, indent=1), encoding="utf-8")
    return out


def pick_style(factor: float, rates: dict[str, float], default: str) -> str:
    """The style whose speed, relative to this voice's usual speed, is
    closest to the speaker's tempo factor."""
    if not rates:
        return default
    base = rates.get(default) or sorted(rates.values())[len(rates) // 2]
    return min(rates, key=lambda st: abs(rates[st] / base - factor))
