"""
Speaker diarization, speaker-pure segmentation, per-speaker gender, delivery.

Order matters. In fast conversation a Whisper segment can contain two people,
and sentence merging can glue a question to its answer. Deciding "who" per
merged sentence then mixes voices, which both scrambles the clustering and
gives half a line the wrong voice. So, as WhisperX/pyannote do:

  1. Embed short SLIDING WINDOWS of speech (SpeechBrain ECAPA-TDNN).
  2. Cluster the windows into speakers; fold tiny clusters into their nearest
     neighbour; optionally force the speaker count (NUM_SPEAKERS).
  3. Give every WORD the speaker of the window centred nearest to it, smooth
     out sub-second flickers, and split Whisper segments where the speaker
     changes.
  4. Only now join pieces into sentences — and never across speakers.
  5. Gender per speaker from pitch pooled over all their audio; delivery per
     line relative to that speaker's own baseline.
"""

import os
from pathlib import Path

import numpy as np

from ..state import PipelineState, Segment

_WIN = float(os.environ.get("DIARIZE_WIN_SECS", "1.5"))
_HOP = float(os.environ.get("DIARIZE_HOP_SECS", "0.75"))
_MIN_WIN = 0.5

# Window-level same-speaker cosine similarity. Short windows are noisier than
# whole segments, so this sits lower than a segment-level threshold would.
_SAME_SPEAKER_SIM = float(os.environ.get("DIARIZE_SAME_SPEAKER_SIM", "0.35"))

# Folding spurious clusters. Size alone is the wrong test — an interviewer
# who asks one 1.5 s question is a real speaker. What separates "same person,
# different intonation" (a man's rising questions split off) from "different
# person" is how close the cluster's centroid sits to another cluster.
# Measured: spurious 0.39-0.49 to their true speaker; real speakers -0.05-0.32.
_TINY_SECS = float(os.environ.get("DIARIZE_TINY_SECS", "1.0"))      # always fold
_MERGE_SIM = float(os.environ.get("DIARIZE_MERGE_SIM", "0.45"))     # fold at any size
_SMALL_SECS = float(os.environ.get("DIARIZE_SMALL_SECS", "3.0"))
_MERGE_SIM_SMALL = float(os.environ.get("DIARIZE_MERGE_SIM_SMALL", "0.33"))
_MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "4"))

# A speaker run shorter than this inside a segment is treated as a flicker.
_MIN_RUN = float(os.environ.get("DIARIZE_MIN_RUN_SECS", "0.6"))

_FEMALE_F0_HZ = float(os.environ.get("FEMALE_F0_HZ", "180"))
_GENDER_RATIO = float(os.environ.get("DIARIZE_GENDER_RATIO", "1.55"))
_F0_MIN, _F0_MAX = 60.0, 400.0
_SR = 16000

_MODEL_DIR = Path("cache") / "_models" / "ecapa"
_encoder = None


def diarize_node(state: PipelineState) -> dict:
    from .transcribe import merge_into_sentences
    from .. import prosody

    raw = sorted(
        [Segment.from_dict(s) for s in state["segments"]], key=lambda s: s.start
    )
    log = state.get("log", [])
    vocals_path = state.get("vocals_path")
    if not raw or not vocals_path or not Path(vocals_path).exists():
        merged = [Segment.from_dict(d) for d in merge_into_sentences([s.to_dict() for s in raw])]
        for s in merged:
            s.speaker = 0
        return {"segments": [s.to_dict() for s in merged], "speaker_profiles": {},
                "log": log + ["⚠ Diarize skipped — no vocals track"]}

    import librosa
    audio, _ = librosa.load(vocals_path, sr=_SR, mono=True)
    forced = _forced_speaker_count(state)

    # ── 1-2. Windows → embeddings → speakers ─────────────────────────────────
    windows = _make_windows(raw)
    method = "ECAPA windows"
    try:
        emb = _embed([audio[int(a * _SR):int(b * _SR)] for _, a, b in windows])
        wlab = _cluster_windows(emb, windows, forced)
    except Exception as exc:
        method = f"fallback: one speaker ({type(exc).__name__})"
        wlab = np.zeros(len(windows), dtype=int)

    # ── 3. Words → speakers → speaker-pure pieces ────────────────────────────
    pieces: list[Segment] = []
    for i, seg in enumerate(raw):
        mine = [(a, b, int(wlab[k])) for k, (si, a, b) in enumerate(windows) if si == i]
        pieces.extend(_split_by_speaker(seg, mine))
    pieces = _renumber_speakers(pieces)
    split_count = len(pieces) - len(raw)

    # ── 4. Sentences, within one speaker only ────────────────────────────────
    segments = [Segment.from_dict(d)
                for d in merge_into_sentences([p.to_dict() for p in pieces])]

    # ── 4b. Better text for each line from Bodhan indic-transcribe ───────────
    from .. import bodhan_asr
    from ..cache import get_cache_dir
    asr_note = "  text: Whisper (set BODHAN_TRANSCRIBE_KEY to use Bodhan)"
    if bodhan_asr.enabled():
        lang = os.environ.get("SOURCE_LANG") or (segments[0].lang if segments else None)
        ok, bad = bodhan_asr.retranscribe(segments, audio, get_cache_dir(state["video_id"]), lang)
        asr_note = (f"  text: Bodhan indic-transcribe on {ok} line(s)"
                    + (f", Whisper kept on {bad} that failed" if bad else ""))

    # ── 5. Gender per speaker, delivery per line ─────────────────────────────
    per_spk_f0: dict[int, list[float]] = {}
    line_feats: dict[int, dict] = {}
    for seg in segments:
        y = audio[int(seg.start * _SR):int(min(seg.end, seg.start + 8) * _SR)]
        f0, track, voiced = _pitch(y) if seg.duration >= 0.5 else (0.0, None, None)
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
    relabelled = _assign_genders(profiles)
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
    ]
    if relabelled:
        notes.append("  (highest-pitched speaker relabelled female by pitch ratio)")
    notes.append("✓ Delivery (vs each speaker's own baseline): "
                 + ", ".join(f"{k}×{v}" for k, v in sorted(tally.items())))

    return {"segments": [s.to_dict() for s in segments],
            "speaker_profiles": profiles, "log": log + notes}


# ── Windows + embeddings ──────────────────────────────────────────────────────

def _make_windows(segs: list[Segment]) -> list[tuple[int, float, float]]:
    """(segment index, start, end). Windows never cross a segment boundary."""
    out = []
    for i, s in enumerate(segs):
        d = s.duration
        if d <= _WIN:
            c = (s.start + s.end) / 2
            half = max(d, _MIN_WIN) / 2
            out.append((i, max(0.0, c - half), c + half))
            continue
        t = s.start
        while t + _WIN < s.end:
            out.append((i, t, t + _WIN))
            t += _HOP
        out.append((i, s.end - _WIN, s.end))
    return out


def _load_encoder():
    global _encoder
    if _encoder is None:
        from speechbrain.inference.speaker import EncoderClassifier
        from speechbrain.utils.fetching import LocalStrategy
        _encoder = EncoderClassifier.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(_MODEL_DIR.resolve()),
            run_opts={"device": "cuda" if _cuda() else "cpu"},
            local_strategy=LocalStrategy.COPY,   # Windows: no symlinks
        )
    return _encoder


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _embed(clips: list[np.ndarray], batch: int = 32) -> np.ndarray:
    """Batched ECAPA embeddings, L2-normalised."""
    import torch
    enc = _load_encoder()
    out = []
    for k in range(0, len(clips), batch):
        group = [c if c.size >= _SR // 4 else np.pad(c, (0, _SR // 4 - c.size))
                 for c in clips[k:k + batch]]
        n = max(len(c) for c in group)
        wavs = np.zeros((len(group), n), dtype=np.float32)
        for j, c in enumerate(group):
            wavs[j, :len(c)] = c
        lens = torch.tensor([len(c) / n for c in group], dtype=torch.float32)
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(wavs), lens).squeeze(1).cpu().numpy()
        out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
    return np.vstack(out)


def _cluster_windows(emb: np.ndarray, windows, forced: int | None) -> np.ndarray:
    n = emb.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    if forced:
        k = min(forced, n)
        if k == 1:
            return np.zeros(n, dtype=int)
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average").fit_predict(emb)
        return labels

    labels = AgglomerativeClustering(
        n_clusters=None, metric="cosine", linkage="average",
        distance_threshold=1.0 - _SAME_SPEAKER_SIM,
    ).fit_predict(emb)

    # Seconds of speech per cluster (window count × hop is a fair proxy).
    def secs(lab):
        return sum(min(b - a, _HOP) for (_, a, b), l in zip(windows, labels) if l == lab)

    # Repeatedly fold the smallest qualifying cluster into its nearest centroid.
    while True:
        uniq = sorted(set(labels))
        if len(uniq) <= 1:
            break
        sizes = {l: secs(l) for l in uniq}
        cent = {l: emb[labels == l].mean(0) for l in uniq}
        cent = {l: v / (np.linalg.norm(v) + 1e-9) for l, v in cent.items()}
        over_cap = len(uniq) > _MAX_SPEAKERS
        folded = False
        for l in sorted(uniq, key=lambda x: sizes[x]):
            target = max((m for m in uniq if m != l), key=lambda m: float(cent[l] @ cent[m]))
            sim = float(cent[l] @ cent[target])
            if (sizes[l] < _TINY_SECS
                    or sim >= _MERGE_SIM
                    or (sizes[l] < _SMALL_SECS and sim >= _MERGE_SIM_SMALL)
                    or over_cap):
                labels = np.where(labels == l, target, labels)
                folded = True
                break
        if not folded:
            break
    return labels


def _forced_speaker_count(state) -> int | None:
    v = state.get("num_speakers") or os.environ.get("NUM_SPEAKERS", "")
    try:
        v = int(v)
        return v if v >= 1 else None
    except (TypeError, ValueError):
        return None


# ── Word-level speaker assignment ─────────────────────────────────────────────

def _split_by_speaker(seg: Segment, wins: list[tuple[float, float, int]]) -> list[Segment]:
    if not wins:
        seg.speaker = 0
        return [seg]
    words = [w for w in (seg.words or []) if w.get("end", 0) > w.get("start", 0) - 1e-6]
    if len(set(l for _, _, l in wins)) == 1 or len(words) < 2:
        seg.speaker = _majority(wins, seg.start, seg.end)
        return [seg]

    def label_at(t: float) -> int:
        return min(wins, key=lambda w: abs((w[0] + w[1]) / 2 - t))[2]

    labs = [label_at((w["start"] + w["end"]) / 2) for w in words]

    # Smooth: absorb runs shorter than _MIN_RUN into the longer neighbour run.
    changed = True
    while changed:
        changed = False
        runs = _runs(labs)
        if len(runs) <= 1:
            break
        for r, (a, b, lab) in enumerate(runs):
            dur = words[b - 1]["end"] - words[a]["start"]
            if dur >= _MIN_RUN:
                continue
            prev = runs[r - 1] if r > 0 else None
            nxt = runs[r + 1] if r + 1 < len(runs) else None
            def rdur(run):
                return words[run[1] - 1]["end"] - words[run[0]]["start"] if run else -1
            new = (prev if rdur(prev) >= rdur(nxt) else nxt)[2]
            labs[a:b] = [new] * (b - a)
            changed = True
            break

    out = []
    for a, b, lab in _runs(labs):
        ws = words[a:b]
        out.append(Segment(
            id=seg.id, start=round(ws[0]["start"], 3), end=round(ws[-1]["end"], 3),
            text="".join(w["word"] for w in ws).strip(), lang=seg.lang,
            words=ws, speaker=lab,
        ))
    return out


def _runs(labs: list[int]) -> list[tuple[int, int, int]]:
    runs, a = [], 0
    for i in range(1, len(labs) + 1):
        if i == len(labs) or labs[i] != labs[a]:
            runs.append((a, i, labs[a]))
            a = i
    return runs


def _majority(wins, start, end) -> int:
    weight: dict[int, float] = {}
    for a, b, lab in wins:
        weight[lab] = weight.get(lab, 0.0) + max(0.0, min(b, end) - max(a, start))
    return max(weight, key=weight.get)


def _renumber_speakers(segs: list[Segment]) -> list[Segment]:
    """Speaker 0 = first voice heard, 1 = next new voice, ..."""
    mapping: dict[int, int] = {}
    for s in sorted(segs, key=lambda s: s.start):
        mapping.setdefault(s.speaker, len(mapping))
        s.speaker = mapping[s.speaker]
    return sorted(segs, key=lambda s: s.start)


# ── Pitch + gender ────────────────────────────────────────────────────────────

def _pitch(y: np.ndarray) -> tuple[float, np.ndarray | None, np.ndarray | None]:
    """(median F0, pyin f0 track, voiced flags). Track is reused by prosody."""
    if y.size < _SR * 0.3 or float(np.max(np.abs(y))) < 1e-3:
        return 0.0, None, None
    import librosa
    try:
        f0, voiced, _ = librosa.pyin(y, fmin=_F0_MIN, fmax=_F0_MAX, sr=_SR, frame_length=1024)
    except Exception:
        return 0.0, None, None
    v = f0[voiced & ~np.isnan(f0)]
    v = v[v > _F0_MIN * 1.15]          # frames pinned to the floor are creak/noise
    return (float(np.median(v)) if v.size >= 5 else 0.0), f0, voiced


def _assign_genders(profiles: dict[str, dict]) -> bool:
    """
    Absolute threshold first. Only if every measured speaker lands on the same
    side AND the highest is at least _GENDER_RATIO × the lowest is the top end
    relabelled — a gap that size is a man/woman pair, not two men.
    """
    fallback = os.environ.get("UNKNOWN_GENDER_FALLBACK", "male")
    measured = {k: p for k, p in profiles.items() if p["f0"]}
    for p in profiles.values():
        p["gender"] = (("female" if p["f0"] >= _FEMALE_F0_HZ else "male")
                       if p["f0"] else fallback)
    if len(measured) < 2 or len({p["gender"] for p in measured.values()}) > 1:
        return False
    lo = min(measured.values(), key=lambda p: p["f0"])
    hi = max(measured.values(), key=lambda p: p["f0"])
    if hi["f0"] / lo["f0"] < _GENDER_RATIO:
        return False
    mid = (lo["f0"] * hi["f0"]) ** 0.5
    for p in measured.values():
        p["gender"] = "female" if p["f0"] >= mid else "male"
    return True
