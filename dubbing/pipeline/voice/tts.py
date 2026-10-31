"""
Text → speech, for one line at a time.

`tts_bytes()` sends the text to the engine the voice belongs to (Bodhan or
Sarvam) and returns WAV bytes. `speakable()` removes symbols a voice cannot
read aloud (^, _, {, } …) before anything is spoken.
"""

import json
import os
from pathlib import Path

from openai import OpenAI

from .catalogue import SARVAM_MODEL, SARVAM_TTS_LANGS, is_kokoro, is_sarvam, kokoro_lang


def sarvam_tts(text: str, lang_code: str, voice: str) -> bytes:
    """WAV bytes from Sarvam Bulbul. Retries rate limits with back-off."""
    import base64
    import time

    import requests

    if lang_code not in SARVAM_TTS_LANGS:
        raise RuntimeError(f"Sarvam Bulbul does not support '{lang_code}'")
    lang = lang_code
    body = {
        "text": text,
        "target_language_code": f"{lang}-IN",
        "model": SARVAM_MODEL,
        "speaker": voice.split(":", 1)[1],
        "speech_sample_rate": 24000,
    }
    for attempt in range(5):
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            json=body,
            timeout=120,
            headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]},
        )
        if r.status_code == 429 and attempt < 4:
            time.sleep(2 * (attempt + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"Sarvam TTS {r.status_code}: {r.text[:200]}")
        return base64.b64decode(r.json()["audios"][0])
    raise RuntimeError("Sarvam TTS: rate limited")


def bodhan_tts(client, text: str, lang_code: str, voice: str, style: str) -> bytes:
    """
    WAV bytes from Bodhan indic-speak, through the shared speak gate
    (pipeline/bodhan_gate.py): limited parallel calls, and a rate-limit
    reply pauses every caller and is retried instead of leaving the line
    undubbed.
    """
    from ..bodhan_gate import SPEAK

    return SPEAK.call(
        lambda: (
            client.audio.speech.create(
                model="indic-speak",
                input=text,
                voice=voice,
                extra_body={"instructions": json.dumps({"lang": lang_code, "style": style})},
            ).content
        )
    )


_KOKORO_PIPES: dict = {}
_KOKORO_LOCK = __import__("threading").Lock()


def kokoro_tts(text: str, lang_code: str, voice: str) -> bytes:
    """WAV bytes (24 kHz) from Kokoro-82M, on the GPU if free. One pipeline
    per language, shared; calls are serialised (the model is small and fast,
    and one GPU model is not safe to call from several threads at once)."""
    import io

    import numpy as np
    import soundfile as sf

    with _KOKORO_LOCK:
        code = kokoro_lang(lang_code)
        if code not in _KOKORO_PIPES:
            from kokoro import KPipeline

            _KOKORO_PIPES[code] = KPipeline(lang_code=code, repo_id="hexgrad/Kokoro-82M")
        parts = [
            np.asarray(a) for _, _, a in _KOKORO_PIPES[code](text, voice=voice.split(":", 1)[1])
        ]
    if not parts:
        raise RuntimeError("Kokoro produced no audio")
    buf = io.BytesIO()
    sf.write(buf, np.concatenate(parts), 24000, format="WAV")
    return buf.getvalue()


def tts_bytes(text: str, lang_code: str, voice: str, style: str, client=None) -> bytes:
    if is_kokoro(voice):
        return kokoro_tts(text, lang_code, voice)
    if voice.startswith("piper:"):
        from .piper_voices import tts as piper_tts

        return piper_tts(text, voice)
    if is_sarvam(voice):
        return sarvam_tts(text, lang_code, voice)
    return bodhan_tts(client or bodhan_speak_client(), text, lang_code, voice, style)


def preview_voice(voice: str, lang_code: str, text: str, out_path: Path) -> str:
    """Synthesize a short sample of `voice` reading `text` (cached on disk)."""
    from .styles import AUTO_BASE_STYLE  # styles imports this module

    if not out_path.exists():
        audio = tts_bytes(text, lang_code, voice, AUTO_BASE_STYLE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio)
    return str(out_path)


def bodhan_speak_client() -> OpenAI:
    from ..bodhan_gate import bodhan_client

    return bodhan_client("BODHAN_SPEAK_KEY")


def speakable(text: str) -> str:
    """
    Drop notation a voice cannot read. The translation step is asked to
    write formulas as words; anything left ("R^d") made the voice say
    something odd for the symbol. Letters and digits around it stay.
    """
    import re

    t = re.sub(r"[\^_{}\\|~`]", " ", text or "")
    return re.sub(r"\s{2,}", " ", t).strip()
