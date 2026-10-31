"""
Audition: hear one line the way a change would voice it, before applying.

The change (one plan action, like the studio's buttons build) is applied to
a copy of the dub; the chosen line is then translated if the change needs
it, spoken by the voice and style the full run would use, and cloned to the
speaker's own voice when the full run would clone it (not for a voice the
user picked). The sample is cached under cache/<id>/audition/, so playing
it again is free. One TTS call per new sample; a language change adds one
translation.
"""

import hashlib
import json
import os
from pathlib import Path

from pipeline.cache import CACHE_ROOT
from pipeline.edits.edit_planner import apply_plan
from pipeline.state import Segment


def audition(pid: str, lang: str, proj: dict, line: int, action: dict) -> str:
    """Path (relative to the project) of a WAV of `line` after `action`."""
    from pipeline.voice import voice_clone as vcl
    from pipeline.voice.casting import assign_voices
    from pipeline.voice.styles import line_style, resolve_style
    from pipeline.voice.tts import speakable, tts_bytes

    settings = proj.get("settings") or {}
    saved = settings.get("voice_map") or {}
    plan = {"requests": [{"request": "audition", "actions": [action]}]}
    segs, vmap, _, _ = apply_plan(proj["segments"], plan, saved, lang)
    segments = [Segment.from_dict(d) for d in segs]
    seg = next((s for s in segments if s.id == line), None)
    if seg is None:
        raise ValueError("No such line.")
    if seg.keep_original:
        raise ValueError("This line keeps the original audio — nothing to hear in a new voice.")
    speak_lang = seg.lang_override or lang
    if not seg.dubbed_text:  # a new language or a corrected transcript: translate this line
        from pipeline.text.line_by_line import translate_with_bodhan

        translate_with_bodhan([seg], speak_lang, {seg.id: seg})
        if not seg.dubbed_text:
            raise ValueError("Could not translate this line for the sample.")

    voices = assign_voices(segments, lang, settings.get("voice_name"), vmap)
    voice = voices.get(seg.speaker or 0)
    style = seg.style_override or resolve_style(
        seg.edit_instruction, line_style(seg, settings.get("voice_style") or "auto")
    )
    if style == "auto":
        style = "educational lecture"
    text = speakable(seg.dubbed_text)
    d = CACHE_ROOT / pid
    key = hashlib.sha1(
        json.dumps([voice, style, speak_lang, text, seg.clone_ref, sorted(vmap.items())]).encode()
    ).hexdigest()[:16]
    out = d / "audition" / f"{key}.wav"
    if out.exists():
        return f"audition/{out.name}"
    out.parent.mkdir(parents=True, exist_ok=True)
    tts = out.with_name(f"{key}_tts.wav")
    tts.write_bytes(tts_bytes(text, speak_lang, voice, style))

    # Cloned like the full run would: not for a voice the user chose.
    chosen = {int(k) for k in vmap if str(k).isdigit()}
    vocals = (
        next(iter((d / "separated").glob("*_vocals.wav")), None)
        if (d / "separated").exists()
        else None
    )
    if (seg.speaker or 0) not in chosen and vcl.ENABLED and vcl.available() and vocals:
        if (seg.clone_ref or "").startswith("file:"):
            ref = str(d / seg.clone_ref[5:])  # the user's own recording
        else:
            ref = vcl.references(segments, str(vocals), d).get(seg.speaker or 0)
        if ref:
            job = {
                "source": str(Path(vcl._padded(str(tts), out.parent)).resolve()),
                "target": str(Path(ref).resolve()),
                "out": str(out.resolve()),
            }
            env = {"SEEDVC_SINGING": "1", "SEEDVC_AUTO_F0": "1"} if vcl.KEEP_INTONATION else None
            res = vcl._run([job], d, env, tag="audition")
            if res.get(job["out"]) == "ok" and out.exists():
                return f"audition/{out.name}"
    os.replace(tts, out)
    return f"audition/{out.name}"
