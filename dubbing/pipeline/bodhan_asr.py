"""
Re-transcribe each final line with Bodhan indic-transcribe.

Bodhan returns plain text — no timestamps — and accepts at most 30 s per
request, so it cannot replace Whisper outright. Whisper still provides the
timing and word positions used for diarization; this step then replaces each
single-speaker line's TEXT with Bodhan's transcription of that line's clean
(Demucs) vocals. Lines that fail keep Whisper's text.

Results are cached per line (by time range), so re-runs do not re-bill.
"""

import io
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

from .state import Segment

_SR = 16000
_MAX_SECS = 29.0          # API limit is 30 s per request
_PAD = 0.15               # keep word edges Whisper's timing may clip


def enabled(source_lang: str | None = None) -> bool:
    """
    ASR_TEXT=auto (default): Bodhan re-transcribes only Indic-language
    sources. On English speech Whisper large-v3 is clearly better — it
    punctuates, keeps case, and Bodhan mis-hears words that flip the meaning
    ("your father" → "my father") or returns garbage on some lines.
    ASR_TEXT=bodhan forces it, ASR_TEXT=whisper disables it.
    """
    mode = os.environ.get("ASR_TEXT", "bodhan").lower()
    if not os.environ.get("BODHAN_TRANSCRIBE_KEY") or mode == "whisper":
        return False
    return mode == "bodhan" or (source_lang or "en").split("-")[0] != "en"


def retranscribe(segments: list[Segment], audio: np.ndarray, cache_dir: Path,
                 language: str | None) -> tuple[int, int]:
    """Replace seg.text in place. Returns (replaced, failed)."""
    from openai import OpenAI
    client = OpenAI(base_url="https://api.bodhan.ai/v1",
                    api_key=os.environ["BODHAN_TRANSCRIBE_KEY"], timeout=60)
    cache_file = cache_dir / "bodhan_asr.json"
    try:
        cache = json.loads(cache_file.read_text(encoding="utf-8"))
    except Exception:
        cache = {}

    def one(seg: Segment) -> tuple[Segment, str | None]:
        key = f"{seg.start:.2f}-{seg.end:.2f}-{language or 'auto'}"
        if key in cache:
            return seg, cache[key]
        a = max(0, int((seg.start - _PAD) * _SR))
        b = min(len(audio), int((seg.end + _PAD) * _SR))
        parts = []
        try:
            for s in range(a, b, int(_MAX_SECS * _SR)):
                chunk = audio[s:min(b, s + int(_MAX_SECS * _SR))]
                if len(chunk) < _SR * 0.3:
                    continue
                parts.append(_call(client, chunk, language))
        except Exception as exc:
            print(f"  ⚠ Bodhan ASR failed for line at {seg.start:.1f}s: {exc}")
            return seg, None
        text = " ".join(p for p in parts if p).strip()
        return seg, text or None

    replaced = failed = 0
    # Bodhan rate-limits hard (429 at 6 parallel); 3 plus back-off is reliable.
    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        for seg, text in pool.map(one, segments):
            key = f"{seg.start:.2f}-{seg.end:.2f}-{language or 'auto'}"
            if text:
                cache[key] = text
                # Keep Whisper's version too: each recogniser fails on
                # different lines (Bodhan is weaker on very short ones), and
                # the translator reconciles the two.
                seg.asr_alt = seg.text
                seg.text = text
                replaced += 1
            else:
                failed += 1
    cache_file.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    return replaced, failed


_WORKERS = int(os.environ.get("BODHAN_ASR_WORKERS", "3"))


def _call(client, chunk: np.ndarray, language: str | None, attempts: int = 5) -> str:
    import time
    for k in range(attempts):
        buf = io.BytesIO()
        sf.write(buf, chunk, _SR, format="WAV", subtype="PCM_16")
        buf.seek(0)
        kwargs = {"model": "indic-transcribe", "file": ("line.wav", buf, "audio/wav")}
        if language:
            kwargs["language"] = language
        try:
            return client.audio.transcriptions.create(**kwargs).text.strip()
        except Exception as exc:
            if "429" not in str(exc) or k == attempts - 1:
                raise
            time.sleep(1.5 * (2 ** k))       # 1.5, 3, 6, 12 s
    return ""
