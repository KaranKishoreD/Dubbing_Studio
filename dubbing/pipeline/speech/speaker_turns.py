"""
Who spoke when: find speaker turns and split lines so each has one speaker.

  - Default (DIARIZE_BACKEND=pyannote): pyannote gives overlap-aware speaker
    turns on the GPU. Two "speakers" with near-identical voiceprints who never
    talk over each other are merged (one person changing tone).
  - Fallback (no HF_TOKEN, or pyannote fails): cut speech into short sliding
    windows, embed each with SpeechBrain ECAPA, and cluster the windows.
    Tiny clusters that sit close to a bigger one are folded into it.
  - Either way, every word takes the speaker of the turn/window around it,
    sub-second flickers are smoothed away, and a Whisper segment is split
    wherever the speaker changes.

Used by nodes/diarize.py.
"""

import os
from pathlib import Path

import numpy as np

from ..state import Segment

SR = 16000  # all speaker analysis runs on 16 kHz mono

# Fallback windows: 1.5 s long, a new one every 0.75 s, none shorter than 0.5 s.
_WIN = float(os.environ.get("DIARIZE_WIN_SECS", "1.5"))
_HOP = float(os.environ.get("DIARIZE_HOP_SECS", "0.75"))
_MIN_WIN = 0.5

# Window-level same-speaker cosine similarity. Short windows are noisier than
# whole segments, so this sits lower than a segment-level threshold would.
_SAME_SPEAKER_SIM = float(os.environ.get("DIARIZE_SAME_SPEAKER_SIM", "0.35"))

# Folding spurious clusters. Size alone is the wrong test: someone who asks
# one short question is still a real speaker. What separates "same person,
# different intonation" from "different person" is how close the cluster's
# centre sits to another cluster. Typical cosine similarity: a split-off
# piece of one person 0.39-0.49, two different people -0.05-0.32.
_TINY_SECS = float(os.environ.get("DIARIZE_TINY_SECS", "1.0"))  # always fold
_MERGE_SIM = float(os.environ.get("DIARIZE_MERGE_SIM", "0.45"))  # fold at any size
_SMALL_SECS = float(os.environ.get("DIARIZE_SMALL_SECS", "3.0"))
_MERGE_SIM_SMALL = float(os.environ.get("DIARIZE_MERGE_SIM_SMALL", "0.33"))  # fold if small
_MAX_SPEAKERS = int(os.environ.get("MAX_SPEAKERS", "12"))  # upper limit, not a target

# A speaker run shorter than this inside a segment is treated as a flicker.
_MIN_RUN = float(os.environ.get("DIARIZE_MIN_RUN_SECS", "0.6"))

# "pyannote" (default) or "ecapa"
BACKEND = os.environ.get("DIARIZE_BACKEND", "pyannote").lower()
_PYANNOTE_MODEL = os.environ.get("PYANNOTE_MODEL", "pyannote/speaker-diarization-community-1")

# Two pyannote speakers whose voiceprints are at least this similar, and who
# never talk at the same time, are one person split by a change of tone
# (questions vs. statements, laughing). Two different men with similar
# voices measure well below this, so they stay apart.
_PYANNOTE_MERGE_SIM = float(os.environ.get("PYANNOTE_MERGE_SIM", "0.75"))

# Models are loaded once, on first use.
_MODEL_DIR = Path("cache") / "_models" / "ecapa"
_encoder = None
_pipeline = None


# ── Fallback: ECAPA voiceprints of short sliding windows ─────────────────────


def make_windows(segs: list[Segment]) -> list[tuple[int, float, float]]:
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
            run_opts={"device": "cuda" if cuda_available() else "cpu"},
            local_strategy=LocalStrategy.COPY,  # Windows: no symlinks
        )
    return _encoder


def cuda_available() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except Exception:
        return False


def embed(clips: list[np.ndarray], batch: int = 32) -> np.ndarray:
    """Batched ECAPA embeddings, L2-normalised."""
    import torch

    enc = _load_encoder()
    out = []
    for k in range(0, len(clips), batch):
        group = [
            c if c.size >= SR // 4 else np.pad(c, (0, SR // 4 - c.size))
            for c in clips[k : k + batch]
        ]
        n = max(len(c) for c in group)
        wavs = np.zeros((len(group), n), dtype=np.float32)
        for j, c in enumerate(group):
            wavs[j, : len(c)] = c
        lens = torch.tensor([len(c) / n for c in group], dtype=torch.float32)
        with torch.no_grad():
            e = enc.encode_batch(torch.from_numpy(wavs), lens).squeeze(1).cpu().numpy()
        out.append(e / (np.linalg.norm(e, axis=1, keepdims=True) + 1e-9))
    return np.vstack(out)


def cluster_windows(emb: np.ndarray, windows, forced: int | None) -> np.ndarray:
    n = emb.shape[0]
    if n == 1:
        return np.zeros(1, dtype=int)
    from sklearn.cluster import AgglomerativeClustering

    if forced:
        k = min(forced, n)
        if k == 1:
            return np.zeros(n, dtype=int)
        labels = AgglomerativeClustering(
            n_clusters=k, metric="cosine", linkage="average"
        ).fit_predict(emb)
        return labels

    labels = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
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
            if (
                sizes[l] < _TINY_SECS
                or sim >= _MERGE_SIM
                or (sizes[l] < _SMALL_SECS and sim >= _MERGE_SIM_SMALL)
                or over_cap
            ):
                labels = np.where(labels == l, target, labels)
                folded = True
                break
        if not folded:
            break
    return labels


# ── Default: pyannote speaker turns ──────────────────────────────────────────


def _load_pyannote():
    global _pipeline
    if _pipeline is None:
        import torch
        from pyannote.audio import Pipeline

        tok = os.environ.get("HF_TOKEN") or None
        try:
            _pipeline = Pipeline.from_pretrained(_PYANNOTE_MODEL, token=tok)
        except TypeError:  # pyannote < 4
            _pipeline = Pipeline.from_pretrained(_PYANNOTE_MODEL, use_auth_token=tok)
        if _pipeline is None:
            raise RuntimeError(
                f"could not load {_PYANNOTE_MODEL} — HF_TOKEN set and model terms accepted?"
            )
        if torch.cuda.is_available():
            _pipeline.to(torch.device("cuda"))
    return _pipeline


def pyannote_turns(
    audio: np.ndarray, forced: int | None
) -> tuple[list[tuple[float, float, int]], str]:
    """
    [(start, end, speaker)] from pyannote, run on the Demucs vocals so music
    and effects cannot pose as a speaker. Exclusive (one speaker at a time)
    turns are used when the pipeline provides them, since each dubbed line
    gets exactly one voice.
    """
    import torch

    pipe = _load_pyannote()
    kwargs = {"num_speakers": forced} if forced else {"max_speakers": _MAX_SPEAKERS}
    wav = {"waveform": torch.from_numpy(audio[None, :].astype(np.float32)), "sample_rate": SR}
    import pyannote.audio

    if int(pyannote.audio.__version__.split(".")[0]) >= 4:
        out = pipe(wav, **kwargs)  # embeddings come back on the output
    else:
        out = pipe(wav, return_embeddings=True, **kwargs)

    emb = None
    if isinstance(out, tuple):  # pyannote 3.x with return_embeddings
        ann, emb = out
    elif hasattr(out, "speaker_diarization"):  # pyannote 4.x DiarizeOutput
        ann = getattr(out, "exclusive_speaker_diarization", None) or out.speaker_diarization
        emb = getattr(out, "speaker_embeddings", None)
    else:
        ann = out
    labels = list(ann.labels())
    turns = [
        (float(seg.start), float(seg.end), labels.index(lab))
        for seg, _, lab in ann.itertracks(yield_label=True)
    ]
    if not turns:
        raise RuntimeError("pyannote found no speech")

    merged = 0
    if emb is not None and not forced and len(labels) > 1:
        turns, merged = _merge_same_voice(turns, np.asarray(emb, dtype=np.float32))
    # Fold speakers with under _TINY_SECS of speech into whoever is talking
    # around them — a cough or a one-word backchannel is not a cast member.
    secs: dict[int, float] = {}
    for a, b, l in turns:
        secs[l] = secs.get(l, 0.0) + b - a
    if len(secs) > 1:
        main = max(secs, key=secs.get)
        tiny = {l for l, v in secs.items() if v < _TINY_SECS}
        if tiny and len(tiny) < len(secs):
            turns = [(a, b, main if l in tiny else l) for a, b, l in turns]
    note = f"pyannote {_PYANNOTE_MODEL.split('/')[-1]}"
    if merged:
        note += f", merged {merged} split voice(s)"
    return sorted(turns), note


def _merge_same_voice(turns, emb: np.ndarray):
    """
    Merge pyannote clusters that are the same person: voiceprints nearly
    identical AND they never overlap in time (two people can talk at once;
    one person cannot). Similar-sounding different people fail the first
    test, so they are kept apart.
    """
    emb = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
    n = emb.shape[0]
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def overlap(i, j):
        ti = [(a, b) for a, b, l in turns if l == i]
        tj = [(a, b) for a, b, l in turns if l == j]
        return sum(max(0.0, min(b1, b2) - max(a1, a2)) for a1, b1 in ti for a2, b2 in tj)

    merged = 0
    pairs = sorted(
        (
            (float(emb[i] @ emb[j]), i, j)
            for i in range(n)
            for j in range(i + 1, n)
            if np.isfinite(emb[i]).all() and np.isfinite(emb[j]).all()
        ),
        reverse=True,
    )
    for sim, i, j in pairs:
        if sim < _PYANNOTE_MERGE_SIM:
            break
        ri, rj = find(i), find(j)
        if ri != rj and overlap(i, j) < 0.5:
            parent[rj] = ri
            merged += 1
    return [(a, b, find(l)) for a, b, l in turns], merged


def forced_speaker_count(state) -> int | None:
    v = state.get("num_speakers") or os.environ.get("NUM_SPEAKERS", "")
    try:
        v = int(v)
        return v if v >= 1 else None
    except (TypeError, ValueError):
        return None


# ── Give every word a speaker, split lines where the speaker changes ─────────


def split_by_speaker(
    seg: Segment, wins: list[tuple[float, float, int]], by_overlap: bool = False
) -> list[Segment]:
    """
    wins are (start, end, speaker): sliding windows (ECAPA) or diarization
    turns (pyannote, by_overlap=True — a word goes to the turn it overlaps
    most, not the nearest window centre).
    """
    if not wins:
        seg.speaker = 0
        return [seg]
    words = [w for w in (seg.words or []) if w.get("end", 0) > w.get("start", 0) - 1e-6]
    if len(set(l for _, _, l in wins)) == 1 or len(words) < 2:
        seg.speaker = _majority(wins, seg.start, seg.end)
        return [seg]

    def label_at(t: float) -> int:
        return min(wins, key=lambda w: abs((w[0] + w[1]) / 2 - t))[2]

    def label_for(w: dict) -> int:
        a, b = w["start"], w["end"]
        best, lab = 0.0, None
        for s0, e0, l in wins:
            ov = min(b, e0) - max(a, s0)
            if ov > best:
                best, lab = ov, l
        if lab is not None:
            return lab
        mid = (a + b) / 2  # word in a gap between turns → nearest turn edge
        return min(
            wins, key=lambda x: 0 if x[0] <= mid <= x[1] else min(abs(mid - x[0]), abs(mid - x[1]))
        )[2]

    labs = (
        [label_for(w) for w in words]
        if by_overlap
        else [label_at((w["start"] + w["end"]) / 2) for w in words]
    )

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
        out.append(
            Segment(
                id=seg.id,
                start=round(ws[0]["start"], 3),
                end=round(ws[-1]["end"], 3),
                text="".join(w["word"] for w in ws).strip(),
                lang=seg.lang,
                words=ws,
                speaker=lab,
                asr_flag=seg.asr_flag,
            )
        )
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


def renumber_speakers(segs: list[Segment]) -> list[Segment]:
    """Speaker 0 = first voice heard, 1 = next new voice, ..."""
    mapping: dict[int, int] = {}
    for s in sorted(segs, key=lambda s: s.start):
        mapping.setdefault(s.speaker, len(mapping))
        s.speaker = mapping[s.speaker]
    return sorted(segs, key=lambda s: s.start)
