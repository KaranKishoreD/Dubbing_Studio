"""
Stage 7 — Synthesize: speak every translated line.

  1. Cast one voice per speaker (voice/casting.py), so a speaker keeps the
     same voice for the whole video.
  2. Pick each speaker's speaking style from their tempo (voice/styles.py).
  3. Speak all lines in parallel with the chosen engine (voice/tts.py).
     Each clip is cached under a hash of voice + style + text, so a re-run
     only re-speaks lines whose words, voice or style changed.
  4. Measure the real clips and rewrite + re-speak lines that are much too
     long or too short for their time (voice/refit.py).

A voice name or style set in the UI overrides the automatic choices.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..cache import get_segments_dir, instruction_hash
from ..state import PipelineState, Segment
from ..voice.casting import (
    DEFAULT_FEMALE,
    DEFAULT_MALE,
    UNKNOWN_GENDER_FALLBACK,
    assign_voices,
    cast_summary,
    pick_voice,
)
from ..voice.catalogue import (
    INTERNATIONAL_ONLY,
    KOKORO_LANGS,
    SARVAM_TTS_LANGS,
    TTS_BACKEND,
    is_kokoro,
    is_sarvam,
    voice_label,
)
from ..voice.refit import refit_lengths
from ..voice.styles import DEFAULT_STYLE, line_style, pick_speaker_styles, resolve_style
from ..voice.tts import bodhan_speak_client, speakable, tts_bytes


def synthesize_node(state: PipelineState) -> dict:
    segments = [Segment.from_dict(s) for s in state["segments"]]
    target_lang = state["target_lang"]
    edit_ops = state.get("edit_ops", [])
    edit_ids = {sid for op in edit_ops for sid in op["segment_ids"]}
    lang_code = target_lang.split("-")[0].lower()

    # Global voice override from UI (if set, used for all segments)
    global_voice_name = state.get("voice_name") or None
    global_voice_style = state.get("voice_style") or DEFAULT_STYLE

    segs_dir = get_segments_dir(state["video_id"])
    client = bodhan_speak_client()

    to_synth = [seg for seg in segments if seg.dubbed_text is not None and not seg.keep_original]
    # Only lines without audio are voiced: a new dub voices every line, an
    # edit only the lines it cleared. Lines it did not touch keep their audio
    # even if the casting or style rules would now choose differently.
    fresh = {s.id for s in to_synth if not s.dubbed_audio or not os.path.exists(s.dubbed_audio)}
    whole = len(fresh) == len(to_synth)
    for seg in segments:
        if seg.keep_original:  # edit: "leave this line as it was"
            seg.dubbed_audio = seg.aligned_audio = None
            seg.sync_clips = []

    # One voice per speaker, decided before any synthesis.
    backend = (state.get("tts_backend") or TTS_BACKEND).lower()
    if backend == "sarvam" and not os.environ.get("SARVAM_API_KEY"):
        backend = "bodhan"
    voice_map = dict(state.get("voice_map") or {})
    swapped = []
    if lang_code not in SARVAM_TTS_LANGS:
        # Bulbul speaks 11 languages; for the rest every Sarvam pick falls
        # back to Bodhan, which covers all 22.
        backend = "bodhan"
        for k, v in list(voice_map.items()):
            if is_sarvam(v):
                swapped.append(voice_label(v))
                voice_map.pop(k)
        if is_sarvam(global_voice_name):
            swapped.append(voice_label(global_voice_name))
            global_voice_name = None

    # International mode: open voices (Kokoro, then Piper for the languages
    # Kokoro lacks); languages no Indian voice speaks always use them.
    def _open(v):
        return is_kokoro(v) or str(v or "").startswith("piper:")

    if (state.get("mode") == "international" and lang_code in KOKORO_LANGS) or (
        lang_code in INTERNATIONAL_ONLY
    ):
        backend = "kokoro"
        voice_map = {k: v for k, v in voice_map.items() if _open(v)}
        if global_voice_name and not _open(global_voice_name):
            global_voice_name = None
    voice_by_speaker = assign_voices(segments, lang_code, global_voice_name, voice_map, backend)
    for seg in segments:
        seg.voice = voice_by_speaker.get(seg.speaker if seg.speaker is not None else 0)
    # Which lines carry a clear feeling, and what kind of video this is.
    genre, emotion_note = "", ""
    if global_voice_style == "auto":
        from ..voice import emotion

        genre, emotion_note = emotion.label_lines(segments, state, only=fresh)
    # A speaker's base style only changes on a whole new voicing, so an edited
    # line sounds like the speaker's other lines.
    pace_note = pick_speaker_styles(segments, lang_code, global_voice_style, genre if whole else "")

    # Lines with English kept in English spelling (Sarvam translation) that a
    # Bodhan voice will read: Bodhan mangles Latin, so it gets a native-script
    # copy. The stored dubbed_text keeps the English spelling.
    from ..text.languages import BODHAN_LATIN, has_latin
    from ..text.transliterate import transliterate_latin

    # A line an edit put in another language is spoken in that language.
    line_lang = {s.id: s.lang_override or lang_code for s in to_synth}
    speak_text: dict[int, str] = {}
    need: dict[str, list] = {}
    for s in to_synth:
        if (
            s.id in fresh
            and not BODHAN_LATIN
            and not is_sarvam(s.voice)
            and not is_kokoro(s.voice)
            and not str(s.voice).startswith("piper:")
            and has_latin(s.dubbed_text)
            and line_lang[s.id] != "en"
        ):
            need.setdefault(line_lang[s.id], []).append((s.id, s.dubbed_text))
    for lang, items in need.items():
        speak_text.update(transliterate_latin(items, lang))

    def _synth_one(seg: Segment) -> tuple[int, str | None]:
        # An explicit edit instruction ("make it sad") beats the measured delivery.
        seg_style = seg.style_override or resolve_style(
            seg.edit_instruction, line_style(seg, global_voice_style)
        )
        lang = line_lang[seg.id]
        voice = seg.voice or pick_voice(lang, UNKNOWN_GENDER_FALLBACK)
        if is_sarvam(voice) and lang not in SARVAM_TTS_LANGS:
            voice = pick_voice(lang, seg.gender or UNKNOWN_GENDER_FALLBACK)

        # The cache key includes voice, style and text, so changing any of
        # them makes a new clip instead of reusing the old one.
        text = speakable(speak_text.get(seg.id, seg.dubbed_text))
        key = instruction_hash(
            f"{seg.edit_instruction or ''}|{voice}|{seg_style}|{text}"
            + (f"|{lang}" if lang != lang_code else "")
        )
        audio_path = segs_dir / f"s{seg.id:04d}_dubbed_{key}.wav"

        if audio_path.exists() and seg.id not in edit_ids:
            return seg.id, str(audio_path)

        # Male voice names are taken from the published catalogue and not all
        # are confirmed against the API, so fall back rather than fail the run.
        fallbacks = [voice]
        primary = pick_voice(lang, seg.gender or UNKNOWN_GENDER_FALLBACK)
        for alt in (primary, DEFAULT_MALE if (seg.gender or "") == "male" else DEFAULT_FEMALE):
            if alt not in fallbacks:
                fallbacks.append(alt)

        last_exc: Exception | None = None
        for attempt, candidate in enumerate(fallbacks):
            try:
                audio = tts_bytes(text, lang, candidate, seg_style, client)
                with open(audio_path, "wb") as f:
                    f.write(audio)
                if attempt:
                    print(f"  ⚠ seg {seg.id}: voice '{voice}' rejected; used '{candidate}'")
                    seg.voice = candidate
                return seg.id, str(audio_path)
            except Exception as exc:
                last_exc = exc

        print(f"  ✗ seg {seg.id}: all voices failed ({last_exc})")
        return seg.id, None

    seg_map = {s.id: s for s in segments}

    def _run(batch: list[Segment]) -> int:
        done = 0
        if not batch:
            return 0
        with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as pool:
            for future in as_completed([pool.submit(_synth_one, s) for s in batch]):
                seg_id, audio_path = future.result()
                if audio_path:
                    seg_map[seg_id].dubbed_audio = audio_path
                    done += 1
        return done

    processed = _run([s for s in to_synth if s.id in fresh])

    # ── Refit: shorten lines whose real audio cannot fit their slot ──────────
    # The translate step only guessed lengths. Now the real clip lengths are
    # known, so measure this voice's actual speaking rate, give each
    # overrunning line a precise character budget, rewrite it shorter and
    # re-synthesize once. This is what keeps the dub from sounding rushed.
    refit_note = refit_lengths(segments, state.get("audio_path"), _run, lang_code, only=fresh)

    styles: dict[str, int] = {}
    for s in to_synth:
        if s.id not in fresh:
            continue
        st = resolve_style(s.edit_instruction, line_style(s, global_voice_style))
        styles[st] = styles.get(st, 0) + 1
    notes = (
        ([emotion_note] if emotion_note else [])
        + ([pace_note] if pace_note else [])
        + [
            f"✓ Synthesized {processed} segments (lang: {lang_code}) — cast: "
            + cast_summary(segments)
            + " — styles: "
            + ", ".join(f"{k}×{v}" for k, v in sorted(styles.items()))
        ]
    )
    if swapped:
        notes.append(
            f"⚠ Sarvam does not speak {lang_code}; used Bodhan instead of "
            + ", ".join(sorted(set(swapped)))
        )
    if refit_note:
        notes.append(refit_note)

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + notes,
    }
