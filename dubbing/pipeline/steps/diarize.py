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

    import hashlib
    import json

    from ..cache import get_cache_dir

    forced = forced_speaker_count(state)
    # The result depends only on the source video, not the dub language:
    # saved, so every language of a video gets the same speakers, and the
    # second language skips this stage's minute of work.
    key = hashlib.sha1(
        json.dumps(
            # face_voice_check: saved results from before face splits had
            # to agree with the voice are not reused
            [
                [s.to_dict() for s in raw],
                forced,
                BACKEND,
                GENDER_METHOD,
                "face_voice_check",
                "second_opinion_v2",
            ],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()[:16]
    saved = get_cache_dir(state["video_id"]) / f"diarized_{key}.json"
    if saved.exists():
        data = json.loads(saved.read_text(encoding="utf-8"))
        return {
            "segments": data["segments"],
            "speaker_profiles": data["speaker_profiles"],
            "log": log + data["log"] + ["  (speakers reused from this video's earlier run)"],
        }

    import librosa

    audio, _ = librosa.load(vocals_path, sr=SR, mono=True)

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
    # Second opinion: pyannote can miss a voice heard for only a few seconds
    # (a 20 s clip with the teacher's 4 s came out as one person, though the
    # voices are far apart: 0.19 similarity against 0.68 within each). So
    # each pyannote speaker's 1.5 s windows are clustered by voiceprint; a
    # speaker is split only if it holds two clearly different voices (below
    # SPLIT_SIM, each with ≥ SPLIT_MIN_SECS). One person's calm and raised
    # voice (0.58 apart) stays one person.
    windows, wlab = make_windows(raw), None
    if turns and not forced:
        try:
            emb = embed([audio[int(a * SR) : int(b * SR)] for _, a, b in windows])
            turns, n_split = _split_mixed_speakers(turns, windows, emb)
            if n_split:
                method += f" + voiceprint check split {n_split} mixed speaker(s)"
        except Exception as exc:
            print(f"  ⚠ voiceprint check skipped: {exc}")
    if not turns:
        try:
            emb = embed([audio[int(a * SR) : int(b * SR)] for _, a, b in windows])
            wlab = cluster_windows(emb, windows, forced)
        except Exception as exc:
            print(f"  ⚠ voiceprint windows failed: {exc}")
    if turns:
        for seg in raw:
            mine = [t for t in turns if t[1] > seg.start - 0.5 and t[0] < seg.end + 0.5]
            pieces.extend(split_by_speaker(seg, mine, by_overlap=True))
    else:
        method += "ECAPA windows"
        if wlab is None:
            method = f"{method} (fallback: one speaker)"
            wlab = np.zeros(len(windows), dtype=int)
        for i, seg in enumerate(raw):
            mine = [(a, b, int(wlab[k])) for k, (si, a, b) in enumerate(windows) if si == i]
            pieces.extend(split_by_speaker(seg, mine))
    pieces = renumber_speakers(pieces)
    split_count = len(pieces) - len(raw)

    # ── 4. Sentences, within one speaker only ────────────────────────────────
    segments = [Segment.from_dict(d) for d in merge_into_sentences([p.to_dict() for p in pieces])]

    # ── 4b. Better text for each line from Bodhan indic-transcribe ───────────
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
    voice_sim = _voice_similarity(segments, audio)
    try:
        face_note = face_speakers.split_speakers(
            segments,
            state.get("video_path"),
            get_cache_dir(state["video_id"]),
            voice_sim=voice_sim,
        )
    except Exception as exc:
        print(f"  ⚠ speakers from faces skipped: {exc}")

    # ── 4d. One person heard as two: a raised or shouting voice can come out
    # of the diarizer as a second speaker. Merge speakers whose voices match.
    merge_note = "" if forced else _merge_same_voices(segments, voice_sim)
    if merge_note:
        segments = renumber_speakers(segments)
        face_note = (face_note + "\n" if face_note else "") + merge_note

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

    saved.write_text(
        json.dumps(
            {
                "segments": [s.to_dict() for s in segments],
                "speaker_profiles": profiles,
                "log": notes,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return {
        "segments": [s.to_dict() for s in segments],
        "speaker_profiles": profiles,
        "log": log + notes,
    }


def _voice_similarity(segments: list[Segment], audio: np.ndarray):
    """
    voice_sim(ids_a, ids_b) for the face step: cosine similarity of the
    average ECAPA voiceprint of two groups of lines. Each line is embedded
    once, on first use.
    """
    by_id = {s.id: s for s in segments}
    cache: dict[int, np.ndarray] = {}

    def vec(ids: list[int]) -> np.ndarray:
        todo = [i for i in ids if i not in cache]
        if todo:
            clips = [audio[int(by_id[i].start * SR) : int(by_id[i].end * SR)] for i in todo]
            for i, e in zip(todo, embed(clips)):
                cache[i] = e / (np.linalg.norm(e) or 1.0)
        v = np.mean([cache[i] for i in ids], axis=0)
        return v / (np.linalg.norm(v) or 1.0)

    def sim(ids_a: list[int], ids_b: list[int]) -> float:
        if not ids_a or not ids_b:
            return 0.0
        return float(vec(ids_a) @ vec(ids_b))

    return sim


# Voice similarity at or above which two diarized "speakers" are one person
# (a raised voice, a shout, another mic). Measured on a film scene: the same
# actor calm vs shouting 0.58, different people −0.01–0.22. Kept well above
# faces.SAME_VOICE (0.35), so a face-based split is never merged back.
MERGE_VOICE = float(os.environ.get("MERGE_SAME_VOICE", "0.50"))


def _merge_same_voices(segments: list[Segment], voice_sim) -> str:
    """Merge the most similar pair of speakers while it is ≥ MERGE_VOICE."""
    merged = []
    while True:
        by: dict[int, list[int]] = {}
        for s in segments:
            by.setdefault(s.speaker, []).append(s.id)
        spk = sorted(by)
        best, pair = MERGE_VOICE, None
        for i, a in enumerate(spk):
            for b in spk[i + 1 :]:
                try:
                    v = voice_sim(by[a], by[b])
                except Exception:
                    continue
                if v >= best:
                    best, pair = v, (a, b)
        if pair is None:
            break
        keep, drop = sorted(pair, key=lambda k: -len(by[k]))  # the one heard most stays
        for s in segments:
            if s.speaker == drop:
                s.speaker = keep
        merged.append(f"S{drop}→S{keep} ({best:.2f})")
    if not merged:
        return ""
    return f"✓ Same voice merged: {', '.join(merged)} — one person, e.g. a raised voice"


SPLIT_SIM = float(os.environ.get("SPLIT_SAME_SPEAKER_SIM", "0.35"))
SPLIT_MIN_SECS = float(os.environ.get("SPLIT_MIN_SECS", "1.5"))


def _split_mixed_speakers(turns, windows, emb: np.ndarray):
    """
    (turns, how many speakers were split). Windows are given to the pyannote
    speaker they overlap most; within one speaker, windows are clustered by
    voiceprint (average linkage, cosine). If two or more clusters of
    ≥ SPLIT_MIN_SECS are below SPLIT_SIM from each other, that speaker's
    turns are replaced by its windows, each labelled with its cluster.
    """
    from sklearn.cluster import AgglomerativeClustering

    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    owner = []
    for _, a, b in windows:
        ov: dict[int, float] = {}
        for x, y, l in turns:
            o = min(b, y) - max(a, x)
            if o > 0:
                ov[l] = ov.get(l, 0.0) + o
        owner.append(max(ov, key=ov.get) if ov else None)
    owner = np.array(owner, dtype=object)
    next_label = max(t[2] for t in turns) + 1
    out, n_split = list(turns), 0
    for spk in sorted({t[2] for t in turns}):
        idx = np.where(owner == spk)[0]
        if len(idx) < 4:
            continue
        lab = AgglomerativeClustering(
            n_clusters=None, metric="cosine", linkage="average", distance_threshold=1 - SPLIT_SIM
        ).fit_predict(emb[idx])
        hop = 0.75
        big = [c for c in set(lab.tolist()) if (lab == c).sum() * hop >= SPLIT_MIN_SECS]
        if len(big) < 2:
            continue
        cent = {c: emb[idx][lab == c].mean(0) for c in big}
        cent = {c: v / (np.linalg.norm(v) + 1e-9) for c, v in cent.items()}
        if max(float(cent[x] @ cent[y]) for x in big for y in big if x < y) >= SPLIT_SIM:
            continue
        # Small clusters go to their nearest big one.
        new = {}
        for c in set(lab.tolist()):
            if c in big:
                new[c] = c
            else:
                v = emb[idx][lab == c].mean(0)
                new[c] = max(big, key=lambda b: float(v @ cent[b]))
        names = {c: (spk if k == 0 else next_label + k - 1) for k, c in enumerate(sorted(big))}
        next_label += len(big) - 1
        out = [t for t in out if t[2] != spk]
        for i, c in zip(idx, lab):
            _, a, b = windows[i]
            out.append((a, b, names[new[c]]))
        n_split += 1
    return sorted(out), n_split
