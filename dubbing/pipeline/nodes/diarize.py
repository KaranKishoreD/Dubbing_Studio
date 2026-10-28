"""
Stage 4 — Diarize: who says each line, their gender, and how they say it.

Order matters. In fast conversation one Whisper segment can hold two people,
and joining segments into sentences can glue a question to its answer. So
speakers are found first, at word level, and sentences are joined afterwards:

  1. Find speaker turns: pyannote, or ECAPA window clustering as a fallback
     (speech/speaker_turns.py).
  2. Give every word a speaker and split segments where the speaker changes.
  3. Join the pieces into sentences, never across two speakers.
  4. Re-transcribe Indic-language lines with Bodhan for better text.
  5. Use faces to split two characters the voice model merged (video/faces.py).
  6. Gender per speaker: voice classifiers, pitch as fallback, and faces
     overrule both when clear (speech/gender.py).
  7. Delivery per line (calm, energetic, quick...) measured against that
     speaker's own baseline (speech/prosody.py).
"""

import os
from pathlib import Path

import numpy as np

from ..speech.gender import GENDER_METHOD, assign_genders, classify_genders, pitch
from ..speech.speaker_turns import (
    BACKEND,
    SR,
    cluster_windows,
    embed,
    forced_speaker_count,
    make_windows,
    pyannote_turns,
    renumber_speakers,
    split_by_speaker,
)
from ..state import PipelineState, Segment


def diarize_node(state: PipelineState) -> dict:
    from ..speech import prosody
    from .transcribe import merge_into_sentences

    raw = sorted([Segment.from_dict(s) for s in state["segments"]], key=lambda s: s.start)
    log = state.get("log", [])
    vocals_path = state.get("vocals_path")
    if not raw or not vocals_path or not Path(vocals_path).exists():
        merged = [Segment.from_dict(d) for d in merge_into_sentences([s.to_dict() for s in raw])]
        for s in merged:
            s.speaker = 0
        return {
            "segments": [s.to_dict() for s in merged],
            "speaker_profiles": {},
            "log": log + ["⚠ Diarize skipped — no vocals track"],
        }

    import librosa

    audio, _ = librosa.load(vocals_path, sr=SR, mono=True)
    forced = forced_speaker_count(state)

    # ── 1-3. Who spoke when → every word → speaker-pure pieces ──────────────
    # pyannote (GPU, overlap-aware, trained for exactly this) when available;
    # the ECAPA sliding-window clustering below is the fallback.
    pieces: list[Segment] = []
    turns, method = None, ""
    if BACKEND == "pyannote":
        try:
            turns, method = pyannote_turns(audio, forced)
        except Exception as exc:
            print(f"  ⚠ pyannote diarization failed, using ECAPA: {exc}")
            method = f"pyannote failed ({type(exc).__name__}) → "
    if turns:
        for seg in raw:
            mine = [t for t in turns if t[1] > seg.start - 0.5 and t[0] < seg.end + 0.5]
            pieces.extend(split_by_speaker(seg, mine, by_overlap=True))
    else:
        windows = make_windows(raw)
        method += "ECAPA windows"
        try:
            emb = embed([audio[int(a * SR) : int(b * SR)] for _, a, b in windows])
            wlab = cluster_windows(emb, windows, forced)
        except Exception as exc:
            method = f"fallback: one speaker ({type(exc).__name__})"
            wlab = np.zeros(len(windows), dtype=int)
        for i, seg in enumerate(raw):
            mine = [(a, b, int(wlab[k])) for k, (si, a, b) in enumerate(windows) if si == i]
            pieces.extend(split_by_speaker(seg, mine))
    pieces = renumber_speakers(pieces)
    split_count = len(pieces) - len(raw)

    # ── 4. Sentences, within one speaker only ────────────────────────────────
    segments = [Segment.from_dict(d) for d in merge_into_sentences([p.to_dict() for p in pieces])]

    # ── 4b. Better text for each line from Bodhan indic-transcribe ───────────
    from ..cache import get_cache_dir
    from ..speech import bodhan_asr

    asr_note = "  text: Whisper (set BODHAN_TRANSCRIBE_KEY to use Bodhan)"
    lang = os.environ.get("SOURCE_LANG") or (segments[0].lang if segments else None)
    asr_note = (
        f"  text: Whisper large-v3 (source {lang or '?'}; Bodhan re-transcription "
        "is used only for Indic-language sources)"
    )
    if bodhan_asr.enabled(lang):
        ok, bad = bodhan_asr.retranscribe(segments, audio, get_cache_dir(state["video_id"]), lang)
        asr_note = f"  text: Bodhan indic-transcribe on {ok} line(s)" + (
            f", Whisper kept on {bad} that failed" if bad else ""
        )

    # ── 4c. Faces: split characters the voice diarizer merged ───────────────
    from ..video import faces as face_speakers

    face_note = ""
    try:
        face_note = face_speakers.split_speakers(
            segments, state.get("video_path"), get_cache_dir(state["video_id"])
        )
    except Exception as exc:
        print(f"  ⚠ speakers from faces skipped: {exc}")

    # ── 5. Gender per speaker, delivery per line ─────────────────────────────
    per_spk_f0: dict[int, list[float]] = {}
    line_feats: dict[int, dict] = {}
    for seg in segments:
        y = audio[int(seg.start * SR) : int(min(seg.end, seg.start + 8) * SR)]
        f0, track, voiced = pitch(y) if seg.duration >= 0.5 else (0.0, None, None)
        seg.f0 = round(f0, 1) if f0 > 0 else None
        if track is not None:
            m = prosody.measure(y, track, voiced)
            if m:
                line_feats[seg.id] = m
        if f0 > 0:
            per_spk_f0.setdefault(seg.speaker, []).extend([f0] * max(1, int(round(seg.duration))))

    profiles: dict[str, dict] = {}
    for spk in sorted({s.speaker for s in segments}):
        f0s = per_spk_f0.get(spk, [])
        profiles[str(spk)] = {
            "f0": round(float(np.median(f0s)), 1) if f0s else None,
            "n_segments": sum(1 for s in segments if s.speaker == spk),
            "seconds": round(sum(s.duration for s in segments if s.speaker == spk), 1),
        }
    relabelled = assign_genders(profiles)
    gender_note = "pitch"
    if GENDER_METHOD == "classifier":
        try:
            probs = classify_genders(audio, segments)
            for spk, p_f in probs.items():
                profiles[str(spk)]["gender"] = "female" if p_f > 0.5 else "male"
                profiles[str(spk)]["p_female"] = round(p_f, 2)
            relabelled = False
            gender_note = "voice classifier"
        except Exception as exc:
            print(f"  ⚠ gender classifier failed, using pitch: {exc}")
    # Faces decide when they are clear: voice classifiers can mistake a man
    # singing high for a woman.
    try:
        fa = face_speakers.analysis(state.get("video_path"), get_cache_dir(state["video_id"]))
        fg = face_speakers.speaker_genders(segments, fa) if fa else {}
    except Exception as exc:
        print(f"  ⚠ face gender skipped: {exc}")
        fg = {}
    changed_by_face = []
    for spk, (g, agree, n) in fg.items():
        prof = profiles.get(str(spk))
        if prof and prof.get("gender") != g:
            changed_by_face.append(f"S{spk}→{g} ({n} lines, {agree:.0%} agree)")
        if prof:
            prof["gender"] = g
    if fg:
        gender_note += " + faces" + (
            f" (corrected {', '.join(changed_by_face)})" if changed_by_face else ""
        )
    for seg in segments:
        seg.gender = profiles[str(seg.speaker)]["gender"]

    labels = prosody.label_lines(line_feats, {s.id: s.speaker for s in segments})
    tally: dict[str, int] = {}
    for seg in segments:
        seg.delivery, seg.pace = labels.get(seg.id, ("neutral", None))
        tally[seg.delivery] = tally.get(seg.delivery, 0) + 1

    summary = ", ".join(
        f"S{k}={p['gender']} ({p['f0'] or '?'}Hz, {p['n_segments']} lines, {p['seconds']}s)"
        for k, p in profiles.items()
    )
    notes = [
        f"✓ Diarize [{method}{', forced ' + str(forced) if forced else ''}]: "
        f"{len(profiles)} speaker(s) — {summary}",
        f"  {len(raw)} ASR segments → split at {split_count} speaker change(s) "
        f"→ {len(segments)} single-speaker lines",
        asr_note,
    ] + ([face_note] if face_note else [])
    notes.append(f"  gender from {gender_note}")
    if relabelled:
        notes.append("  (highest-pitched speaker relabelled female by pitch ratio)")
    notes.append(
        "✓ Delivery (vs each speaker's own baseline): "
        + ", ".join(f"{k}×{v}" for k, v in sorted(tally.items()))
    )

    return {
        "segments": [s.to_dict() for s in segments],
        "speaker_profiles": profiles,
        "log": log + notes,
    }
