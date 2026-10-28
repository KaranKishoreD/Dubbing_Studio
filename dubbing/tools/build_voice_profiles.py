"""
Measure how every catalogue voice sounds and store it.

Speaks one sample sentence with each voice, measures its median pitch, and
writes cache/_voice_profiles.json (gender, native language, pitch, pitch
tier). Casting uses these numbers to give each speaker the voice closest to
their own pitch. Run once, or again when the voice catalogue changes:

    cd dubbing && python -m tools.build_voice_profiles ta "வணக்கம், இது ஒரு சோதனை."
"""

import json
import sys
from pathlib import Path

from pipeline.voice.catalogue import _PROFILES_PATH, voice_catalogue, voice_profiles
from pipeline.voice.tts import preview_voice


def build_voice_profiles(lang_code: str, text: str, sample_dir: Path) -> dict:
    """Synthesize `text` with every voice, measure median pitch, store tiers."""
    import time

    import librosa
    import numpy as np

    profiles = voice_profiles()
    for voice, gender, native in voice_catalogue():
        if voice in profiles:
            continue
        path = sample_dir / f"{voice}_{lang_code}.wav"
        for attempt in range(4):
            try:
                preview_voice(voice, lang_code, text, path)
                break
            except Exception as exc:
                if "429" not in str(exc) or attempt == 3:
                    print(f"  ✗ {voice}: {exc}")
                    break
                time.sleep(4 * (attempt + 1))
        if not path.exists():
            continue
        y, sr = librosa.load(str(path), sr=16000)
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        v = f0[voiced & ~np.isnan(f0)]
        if v.size < 10:
            continue
        profiles[voice] = {"gender": gender, "native": native, "f0": round(float(np.median(v)), 1)}
        time.sleep(1.5)
    # Tiers within each gender: lowest third deep, highest third high.
    for g in ("male", "female"):
        vs = sorted((p["f0"], v) for v, p in profiles.items() if p["gender"] == g)
        for k, (_, v) in enumerate(vs):
            q = k / max(1, len(vs) - 1)
            profiles[v]["pitch"] = "deep" if q < 1 / 3 else "high" if q > 2 / 3 else "medium"
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=1, ensure_ascii=False), encoding="utf-8")
    return profiles


if __name__ == "__main__":
    lang, text = sys.argv[1], sys.argv[2]
    done = build_voice_profiles(lang, text, Path("cache") / "_voice_previews")
    print(f"{len(done)} voice profiles in {_PROFILES_PATH}")
