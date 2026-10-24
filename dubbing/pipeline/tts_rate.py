"""
Measured speaking rate of each TTS voice, in characters per second.

A fixed 14 chars/s guess is badly wrong across Indic scripts: a Tamil or
Malayalam line packs more characters per spoken syllable than Hindi, so the
translator was handed budgets it could not honour and the overrun was then
"fixed" by speeding the voice up. Every synthesis run measures the real rate
of each voice it used and stores it here, so the next translation gets a
budget in the voice's own units.
"""

import json
import os
import threading
from pathlib import Path

_PATH = Path("cache") / "_tts_rates.json"
_DEFAULT = float(os.environ.get("TTS_CHARS_PER_SEC", "14"))
_LOCK = threading.Lock()
# Weight of a new measurement against the stored value.
_ALPHA = 0.5


def _load() -> dict:
    try:
        return json.loads(_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def chars_per_sec(lang: str, voice: str | None = None) -> float:
    """Best known rate for this voice in this language, else the language, else default."""
    d = _load()
    return float(d.get(f"{lang}|{voice}") or d.get(lang) or _DEFAULT)


def record(lang: str, rates_by_voice: dict[str, list[float]]) -> None:
    """Fold this run's measured rates (chars/s per clip) into the stored values."""
    def median(xs):
        xs = sorted(xs)
        return xs[len(xs) // 2]

    all_rates = [r for rs in rates_by_voice.values() for r in rs]
    if not all_rates:
        return
    with _LOCK:
        d = _load()
        updates = {f"{lang}|{v}": median(rs) for v, rs in rates_by_voice.items() if len(rs) >= 2}
        updates[lang] = median(all_rates)
        for k, new in updates.items():
            old = d.get(k)
            d[k] = round(new if old is None else (1 - _ALPHA) * old + _ALPHA * new, 2)
        _PATH.parent.mkdir(parents=True, exist_ok=True)
        _PATH.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
