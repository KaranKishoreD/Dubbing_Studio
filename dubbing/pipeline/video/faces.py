"""
Speakers from faces: split voices the audio diarizer merged.

Over loud music, short lines from many characters all sound alike to
pyannote, so a film trailer with ten characters can come out as two or
three speakers sharing one voice. The video says more: during each line,
whose mouth is moving on screen. Face identities (ArcFace clusters, the same
analysis lip sync uses) give each line a "talking face"; when one diarized
speaker's lines show two or more different talking faces, each extra face
becomes its own speaker.

It only acts when more talking faces are seen than the diarizer found
speakers — two faces and two speakers (an interview) is left alone. A line
with no clear talking face (off screen, voice-over, two mouths moving) keeps
its diarized speaker. FACE_SPEAKERS=1 enables it.
"""

import os
from pathlib import Path

import numpy as np

ENABLED = os.environ.get("FACE_SPEAKERS", "1") == "1"
_FACE_SAME_SIM = 0.40  # same person if ArcFace similarity ≥ this
_MIN_PRESENT = 0.4  # face on screen for ≥ 40% of the line's samples
_MIN_MOUTH = 0.02  # talking (still face ≈ 0.000, speech ≈ 0.05)
_CLEAR = 0.7  # runner-up mouth motion must be < 70% of the best
_MIN_SECS = 1.0  # a face needs this much speech to become a speaker


def analysis(video_path: str, cache_dir: Path) -> dict | None:
    """faces.npz (shared with lip sync), computed here if missing."""
    faces = cache_dir / "faces.npz"
    if faces.exists() and not {"yaw", "male"} <= set(np.load(faces).files):
        faces.unlink()
    if not faces.exists():
        try:
            from .latentsync_runner import run_worker

            run_worker(
                ["--analyze", str(Path(video_path).resolve()), str(faces.resolve())],
                cache_dir / "face_analysis.log",
            )
        except Exception as exc:
            print(f"  ⚠ face analysis for speakers skipped: {exc}")
    return dict(np.load(faces)) if faces.exists() else None


def _identities(a: dict) -> np.ndarray:
    from sklearn.cluster import AgglomerativeClustering

    t, bbox, emb = a["t"], a["bbox"], a["emb"]
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])
    big = area >= np.percentile(area, 20)
    ident = np.full(len(t), -1)
    if big.sum() >= 2:
        ident[big] = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1 - _FACE_SAME_SIM,
        ).fit_predict(emb[big])
    return ident


def talking_faces(segments, a: dict) -> dict[int, int]:
    """{segment id: face identity clearly talking through it}."""
    t, mouth = a["t"], a["mouth"]
    step = int(a["step"]) / float(a["fps"])
    ident = _identities(a)
    out = {}
    for s in segments:
        k = np.where((t >= s.start) & (t < s.end) & (ident >= 0))[0]
        n = max(1, int(round(s.duration / step)))
        if len(k) == 0:
            continue
        scores = []
        for f in set(ident[k].tolist()):
            kk = k[ident[k] == f]
            present = len(np.unique(t[kk])) / n
            mm = mouth[kk]
            mm = mm[np.isfinite(mm)]
            if present >= _MIN_PRESENT and len(mm) >= 3:
                scores.append((float(np.median(mm)), f))
        scores.sort(reverse=True)
        if (
            scores
            and scores[0][0] >= _MIN_MOUTH
            and (len(scores) == 1 or scores[1][0] < _CLEAR * scores[0][0])
        ):
            out[s.id] = scores[0][1]
    return out


# Voice similarity (ECAPA, cosine of line-group centres) at or above which a
# second face of one diarized speaker is taken to be the same person seen on
# another camera shot, not a new character. Measured on a film scene (3
# Idiots, three men): the same actor across shots 0.51–0.64, different
# people 0.00–0.22.
SAME_VOICE = float(os.environ.get("FACE_SPLIT_SAME_VOICE", "0.35"))


def split_speakers(segments, video_path: str, cache_dir: Path, voice_sim=None) -> str:
    """
    Give extra talking faces of one diarized speaker their own speaker id,
    but only when the voice agrees. In films the camera often shows the
    listener, or the same actor from another angle, so a second face alone
    is weak evidence. voice_sim(ids_a, ids_b) → cosine similarity of the two
    groups of lines' voices; a split is skipped when it is ≥ SAME_VOICE.
    """
    if not ENABLED or not video_path or not os.path.exists(video_path):
        return ""
    a = analysis(video_path, cache_dir)
    if not a or len(a["t"]) == 0:
        return ""
    face_of = talking_faces(segments, a)
    secs: dict[int, float] = {}
    for s in segments:
        if s.id in face_of:
            secs[face_of[s.id]] = secs.get(face_of[s.id], 0.0) + s.duration
    faces = {f for f, v in secs.items() if v >= _MIN_SECS}
    speakers = {s.speaker for s in segments}
    if len(faces) <= len(speakers):
        return ""  # nothing the diarizer missed

    # Per speaker: their main face keeps the speaker id; other faces become
    # new speakers (one id per face, even across diarized speakers).
    next_id = max(speakers) + 1
    new_id: dict[int, int] = {}
    moved = kept = 0
    for spk in sorted(speakers):
        mine: dict[int, float] = {}
        for s in segments:
            f = face_of.get(s.id)
            if s.speaker == spk and f in faces:
                mine[f] = mine.get(f, 0.0) + s.duration
        if len(mine) < 2:
            continue
        main = max(mine, key=mine.get)
        main_ids = [s.id for s in segments if s.speaker == spk and face_of.get(s.id) == main]
        same_voice = set()
        if voice_sim is not None:
            for f in mine:
                if f == main:
                    continue
                f_ids = [s.id for s in segments if s.speaker == spk and face_of.get(s.id) == f]
                try:
                    if voice_sim(f_ids, main_ids) >= SAME_VOICE:
                        same_voice.add(f)
                        kept += len(f_ids)
                except Exception as exc:
                    print(f"  ⚠ voice check for a face split failed: {exc}")
        for s in segments:
            f = face_of.get(s.id)
            if s.speaker == spk and f in faces and f != main and f not in same_voice:
                if f not in new_id:
                    new_id[f], next_id = next_id, next_id + 1
                s.speaker = new_id[f]
                moved += 1
    note = (
        f"  faces: {kept} line(s) on another camera shot kept with their speaker (same voice)"
        if kept
        else ""
    )
    if not moved:
        return note
    return (
        f"✓ Speakers from faces: {len(faces)} talking faces vs {len(speakers)} voices — "
        f"{moved} line(s) moved to {len(new_id)} new speaker(s)" + (f"\n{note}" if note else "")
    )


def speaker_genders(segments, a: dict) -> dict[int, tuple[str, float, int]]:
    """
    {speaker: (gender, agreement, lines)} from the faces clearly talking or
    singing during each speaker's lines. Voice-based gender classifiers are
    trained on speech: a man singing at 240–270 Hz came out "female 1.00".
    A face's gender does not change when its owner sings. Only speakers
    whose lines show a talking face on at least 2 lines, agreeing ≥ 60%,
    are returned.
    """
    if "male" not in a:
        return {}
    ident = _identities(a)
    face_of = talking_faces(segments, a)
    male_of = {}
    for f in set(face_of.values()):
        v = a["male"][ident == f]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            male_of[f] = float(np.mean(v))
    votes: dict[int, list[tuple[float, float]]] = {}
    for s in segments:
        f = face_of.get(s.id)
        if f in male_of and abs(male_of[f] - 0.5) >= 0.25:  # confident face
            votes.setdefault(s.speaker, []).append((male_of[f], s.duration))
    out = {}
    for spk, v in votes.items():
        if len(v) < 2:
            continue
        w = sum(d for _, d in v)
        male = sum(d for m, d in v if m > 0.5) / w
        g, agree = ("male", male) if male >= 0.5 else ("female", 1 - male)
        if agree >= 0.6:
            out[spk] = (g, round(agree, 2), len(v))
    return out
