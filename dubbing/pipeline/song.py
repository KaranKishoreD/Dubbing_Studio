"""
Sung lines: keep the original instead of dubbing them as speech.

Whisper transcribes lyrics like dialogue, and the transcript refiner calls
them speech, so a trailer's title song was translated and read out by a
TTS voice. Two independent signals on the Demucs vocal track:

  held notes  share of voiced 150 ms windows whose pitch stays within 0.6
              semitone — singing sustains notes, speech glides. Measured:
              speech 0.00–0.06, a sung chant 0.35.
  classifier  AudioSet AST "singing/chant/rapping/…" score against its
              "speech" score.

A line counts as sung when it is mostly voiced and holds notes
(≥ _HELD with voiced ≥ _VOICED), or the classifier clearly hears singing.
Results are cached per video. SONG_DETECT=1 enables it.
"""

import json
import os
from pathlib import Path

import numpy as np

ENABLED = os.environ.get("SONG_DETECT", "1") == "1"
_HELD = float(os.environ.get("SONG_HELD_SHARE", "0.25"))
_VOICED = 0.6
_SECTION_GAP = 3.0           # seconds between lines of one song section
_SECTION_SPEECH = 0.35       # a neighbour clearly heard as speech stays speech
_AST = "MIT/ast-finetuned-audioset-10-10-0.4593"
_SING = ("Singing", "Choir", "Chant", "Male singing", "Female singing", "Child singing",
         "Rapping", "Humming", "Vocal music", "Song")
_SPEECH = ("Speech", "Male speech, man speaking", "Female speech, woman speaking",
           "Conversation", "Narration, monologue")


def sung_lines(segments, vocals_path: str, cache_dir: Path) -> dict[int, dict]:
    """{segment id: {"held", "voiced", "sing", "speech"}} for lines that are sung."""
    st = os.stat(vocals_path)
    path = cache_dir / "songs.json"
    key = f"{st.st_size}|{int(st.st_mtime)}|{_HELD}|sections2"
    spans = [(s.id, round(s.start, 2), round(s.end, 2)) for s in segments]
    try:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("key") == key and cached.get("spans") == [list(x) for x in spans]:
            return {int(k): v for k, v in cached["sung"].items()}
    except Exception:
        pass
    sung, scores = _detect(segments, vocals_path)
    sung = _sections(segments, sung, scores)
    path.write_text(json.dumps({"key": key, "spans": spans, "sung": sung, "scores": scores}),
                    encoding="utf-8")
    return sung


def _sections(segments, sung: dict, scores: dict) -> dict:
    """
    A song does not stop between lines: a line next to a sung line (gap
    ≤ _SECTION_GAP s) that is not clearly SPEECH joins the song. Line-by-
    line detection found 2 of 7 lines in a stage duet (fast singing over a
    loud band); the other 5 were dubbed as speech.
    """
    ordered = sorted(segments, key=lambda s: s.start)
    grew = True
    while grew:
        grew = False
        for i, s in enumerate(ordered):
            if s.id in sung:
                continue
            sc = scores.get(s.id)
            if not sc or sc["speech"] >= _SECTION_SPEECH:
                continue
            near = [o for o in ordered[max(0, i - 1):i + 2]
                    if o.id in sung and o is not s
                    and max(o.start - s.end, s.start - o.end) <= _SECTION_GAP]
            if near:
                sung[s.id] = {**sc, "section": True}
                grew = True
    return sung


def _detect(segments, vocals_path: str) -> dict[int, dict]:
    import librosa
    y, _ = librosa.load(vocals_path, sr=16000, mono=True)
    clf = _classifier()
    out, scores = {}, {}
    for s in segments:
        clip = y[int(s.start * 16000):int(s.end * 16000)]
        if len(clip) < 16000 * 0.8:
            continue
        held, voiced = _held_notes(clip)
        sing, speech = clf(clip) if clf else (0.0, 1.0)
        sc = {"held": round(held, 2), "voiced": round(voiced, 2),
              "sing": round(sing, 2), "speech": round(speech, 2)}
        scores[s.id] = sc
        if (held >= _HELD and voiced >= _VOICED) or (sing >= 0.4 and sing >= 0.6 * speech):
            out[s.id] = sc
    return out, scores


def _held_notes(clip: np.ndarray) -> tuple[float, float]:
    import librosa
    f0, voiced, _ = librosa.pyin(clip, fmin=70, fmax=600, sr=16000, frame_length=1024,
                                 hop_length=160)
    st = 12 * np.log2(np.where(voiced, f0, np.nan) / 100)
    win, held = 15, 0                                  # 15 hops = 150 ms
    for i in range(len(st) - win):
        w = st[i:i + win]
        if np.all(np.isfinite(w)) and np.nanmax(w) - np.nanmin(w) < 0.6:
            held += 1
    n_voiced = int(np.sum(voiced))
    return held / max(1, n_voiced), n_voiced / max(1, len(st))


def _classifier():
    """(sing, speech) scores for a clip, or None if the model is unavailable."""
    try:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification
        fx = ASTFeatureExtractor.from_pretrained(_AST)
        model = ASTForAudioClassification.from_pretrained(_AST).eval()
        dev = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2e9 else "cpu"
        model.to(dev)
        lab = model.config.id2label
        sing = [i for i, n in lab.items() if n in _SING]
        speech = [i for i, n in lab.items() if n in _SPEECH]
    except Exception as exc:
        print(f"  ⚠ song classifier unavailable ({exc}) — using held notes only")
        return None

    def score(clip):
        wins = [clip[i:i + 64000] for i in range(0, max(1, len(clip) - 8000), 32000)]
        wins = [w for w in wins if len(w) > 4000] or [clip]
        with torch.no_grad():
            p = torch.sigmoid(model(**{k: v.to(dev) for k, v in fx(
                wins, sampling_rate=16000, return_tensors="pt").items()}).logits).cpu().numpy()
        return float(p[:, sing].max(1).mean()), float(p[:, speech].max(1).mean())
    return score
