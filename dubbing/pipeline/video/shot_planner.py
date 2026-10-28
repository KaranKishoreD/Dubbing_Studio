"""
Which shots of each line get lip-synced.

A line is cut into shots at camera cuts. A shot is synced only if the
speaker's own face is on screen, turned at most ~40°, and actually moving
its mouth — so a listener in a reverse shot, a photo on a poster or a
voice-over never gets new lips.

Faces are matched to speakers by screen time and lip motion. Shots that
matching misses are added when one face is clearly the one talking
(`_talking_face`), unless that face belongs to a different speaker.
"""

import os

import numpy as np

from ..state import Segment

# Same person if face embeddings (ArcFace) are at least this similar.
_FACE_SAME_SIM = float(os.environ.get("FACE_SAME_SIM", "0.40"))


# Share of a shot's samples in which the speaker must be the main face.
# Face detection misses some frames even when the speaker is on screen the
# whole time, so a stricter value drops shots that should be synced. A shot
# where LatentSync then loses the face keeps its original frames.
_SHOT_FACE_FRAC = float(os.environ.get("SHOT_FACE_FRAC", "0.5"))


# Yaw proxy = nose offset from the eye midpoint / eye distance. ~0.35 ≈ 20°,
# ~0.7 ≈ 40° (three-quarter view).
_MAX_YAW = float(os.environ.get("LIPSYNC_MAX_YAW", "0.7"))


# Share of a person's detections that must be usable to be a candidate.
_MIN_USABLE = float(os.environ.get("LIPSYNC_MIN_USABLE", "0.3"))


_MIN_SHOT_SECS = float(os.environ.get("MIN_SYNC_SHOT_SECS", "0.4"))


# Median mouth-region change between samples below which a face is not
# talking (static photo / freeze-frame). Measured: still image 0.000,
# talking faces 0.05-0.08.
_MIN_MOUTH_MOTION = float(os.environ.get("MIN_MOUTH_MOTION", "0.01"))


# Per-shot talking face (see _talking_face).
_LONE_MOUTH = float(os.environ.get("LONE_FACE_MOUTH", "0.012"))


_TALK_MOUTH = float(os.environ.get("TALK_FACE_MOUTH", "0.02"))


def plan_sync_shots(
    analysis: dict, segments: list[Segment], spans: dict[int, tuple[float, float]]
) -> tuple[dict, str]:
    """
    {segment id: [(start, end, face_cx, face_cy), ...]} — the parts of each line's span where
    the camera is on the speaker. Faces are clustered into identities, and
    each identity is matched to the diarized speaker whose lines it moves its
    mouth during (lip motion, not screen time: in shot/reverse-shot the
    listener is on screen about as often as the speaker).
    """
    from scipy.optimize import linear_sum_assignment
    from sklearn.cluster import AgglomerativeClustering

    t, bbox = analysis["t"], analysis["bbox"]
    # "Usable" face = turned at most _MAX_YAW. Studio shots are often at a
    # three-quarter angle; LatentSync aligns faces by landmarks and handles
    # those, so only faces turned further away are skipped.
    frontal = (analysis["yaw"] < _MAX_YAW) if "yaw" in analysis else analysis["frontal"]
    mouth, emb, cuts = analysis["mouth"], analysis["emb"], list(analysis["cuts"])
    if len(t) == 0:
        return {}, "no faces in the video"
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])

    # Identities: cluster the reasonably sized faces; tiny background faces
    # are never lip-synced and only add noise.
    big = area >= np.percentile(area, 20)
    ident = np.full(len(t), -1)
    if big.sum() >= 2:
        lab = AgglomerativeClustering(
            n_clusters=None,
            metric="cosine",
            linkage="average",
            distance_threshold=1 - _FACE_SAME_SIM,
        ).fit_predict(emb[big])
        ident[big] = lab
    elif big.sum() == 1:
        ident[big] = 0

    # Who is speaking at each sample.
    def speaker_at(x: float) -> int | None:
        for s in segments:
            if s.start <= x <= s.end:
                return s.speaker if s.speaker is not None else 0
        return None

    spk_at = np.array([(-1 if (v := speaker_at(x)) is None else v) for x in t])

    # Only faces seen mostly frontal can be lip-synced; the big blurred
    # over-the-shoulder heads in a reverse shot are not candidates (and camera
    # motion makes their "mouth" look busy).
    ids = [
        i
        for i in set(ident.tolist())
        if i >= 0 and (ident == i).sum() >= 10 and frontal[ident == i].mean() >= _MIN_USABLE
    ]
    # A face seen in a handful of frames (a poster, a background extra) can
    # score a perfect screen-time share by luck; real cast members recur.
    if ids:
        top = max((ident == i).sum() for i in ids)
        ids = [i for i in ids if (ident == i).sum() >= max(10, 0.05 * top)]
    spks = sorted({s.speaker if s.speaker is not None else 0 for s in segments})
    if not ids:
        return {}, "no recurring faces"

    # Score[id][spk] combines two clues:
    #  - screen time: share of this face's frontal frames that fall while spk
    #    talks. Decisive in interviews/lectures, where the camera cuts to
    #    whoever is speaking.
    #  - lip motion: how much more its mouth moves while spk talks than while
    #    others do. Decisive in shot/reverse-shot, where the listener is on
    #    screen as often as the speaker (screen time ≈ 50/50 there).
    # Neither clue is reliable alone, so both are used.
    score = np.zeros((len(ids), len(spks)))
    for a, i in enumerate(ids):
        f = (ident == i) & frontal & (spk_at >= 0)
        m = f & np.isfinite(mouth)
        for b, s in enumerate(spks):
            share = (f & (spk_at == s)).sum() / max(1, f.sum())
            on, off = m & (spk_at == s), m & (spk_at != s)
            lip = 0.0
            if on.sum() >= 3 and off.sum() >= 3:
                lip = (mouth[on].mean() - mouth[off].mean()) / (mouth[m].std() + 1e-6)
            score[a, b] = share + 0.5 * lip
    rows, cols = linear_sum_assignment(-score)
    face_of = {spks[c]: ids[r] for r, c in zip(rows, cols) if score[r, c] > 0.2}

    plan: dict[int, list[tuple[float, float]]] = {}
    for seg in segments:
        spk = seg.speaker if seg.speaker is not None else 0
        if seg.id not in spans or spk not in face_of:
            continue
        a, b = spans[seg.id]
        edges = [a] + [c for c in cuts if a < c < b] + [b]
        for x, y in zip(edges, edges[1:]):
            if y - x < _MIN_SHOT_SECS:
                continue
            inside = np.where((t >= x) & (t < y))[0]
            if len(inside) == 0:
                continue
            # Is the speaker's own face on screen, frontal, through the shot?
            mine = inside[(ident[inside] == face_of[spk]) & frontal[inside]]
            n_frames = max(1, int(round((y - x) * float(analysis["fps"]) / int(analysis["step"]))))
            # Only a face that is actually talking in the original gets new
            # lips: a host's photo on an end card, a poster or a freeze-frame
            # has no mouth motion (0.000 vs ~0.06 for real speech), and a
            # voice-over across it must not animate it.
            mm = mouth[mine]
            mm = mm[np.isfinite(mm)]
            talking = len(mm) >= 3 and float(np.median(mm)) >= _MIN_MOUTH_MOTION
            if talking and len(np.unique(t[mine])) >= _SHOT_FACE_FRAC * n_frames:
                b = bbox[mine]
                # Where the face sits, so LatentSync syncs this face and not
                # the larger blurred head of the listener in the foreground.
                cx = float(np.median((b[:, 0] + b[:, 2]) / 2))
                cy = float(np.median((b[:, 1] + b[:, 3]) / 2))
                plan.setdefault(seg.id, []).append((round(float(x), 3), round(float(y), 3), cx, cy))

    # Shots the speaker→face matching missed: sync the face that is clearly
    # talking in that shot, whoever the diarizer says is speaking. When the
    # speaker labels are wrong (many voices over loud music) the matching
    # above finds few shots; this per-shot pass covers what it misses without
    # relying on labels. It only ADDS shots.
    extra = 0.0
    for seg in segments:
        if seg.id not in spans:
            continue
        a, b = spans[seg.id]
        have = plan.get(seg.id, [])
        edges = [a] + [c for c in cuts if a < c < b] + [b]
        for x, y in zip(edges, edges[1:]):
            if y - x < _MIN_SHOT_SECS or any(px < y and x < py for px, py, *_ in have):
                continue
            pick = _talking_face(
                t, ident, frontal, mouth, bbox, x, y, float(analysis["fps"]), int(analysis["step"])
            )
            if not pick:
                continue
            face, cx, cy = pick
            spk = seg.speaker if seg.speaker is not None else 0
            # A face known to be ANOTHER speaker is the listener reacting
            # (shot/reverse-shot): never give them this line's words.
            if face in face_of.values() and face_of.get(spk) != face:
                continue
            plan.setdefault(seg.id, []).append((round(float(x), 3), round(float(y), 3), cx, cy))
            extra += y - x
    note = ", ".join(f"S{s + 1}→face {face_of[s]}" for s in sorted(face_of))
    return plan, (
        f"faces matched by screen time + lip motion: {note or 'none'}"
        + (f"; +{extra:.0f}s from per-shot talking faces" if extra else "")
    )


def _talking_face(t, ident, frontal, mouth, bbox, x, y, fps, step):
    """
    (face id, cx, cy) of the face clearly talking in [x, y), or None. A lone dominant
    face (others under a third of its width) needs only slight mouth motion
    — a calm talker moves the jaw little (0.019 measured); among comparable
    faces the talker must move clearly more than the rest.
    """
    k = np.where((t >= x) & (t < y) & (ident >= 0) & frontal)[0]
    n = max(1, int(round((y - x) * fps / step)))
    cands = []
    for f in set(ident[k].tolist()):
        kk = k[ident[k] == f]
        if len(np.unique(t[kk])) < 0.5 * n:
            continue
        mm = mouth[kk]
        mm = mm[np.isfinite(mm)]
        if len(mm) < 3:
            continue
        w = float(np.median(bbox[kk, 2] - bbox[kk, 0]))
        cands.append((float(np.median(mm)), w, kk, f))
    if not cands:
        return None
    cands.sort(key=lambda c: -c[1])
    lone = len(cands) == 1 or cands[1][1] < cands[0][1] / 3
    if lone:
        m, _, kk, fid = cands[0]
        if m < _LONE_MOUTH:
            return None
    else:
        cands.sort(key=lambda c: -c[0])
        if cands[0][0] < _TALK_MOUTH or cands[1][0] >= 0.7 * cands[0][0]:
            return None
        kk, fid = cands[0][2], cands[0][3]
    bb = bbox[kk]
    return (
        fid,
        float(np.median((bb[:, 0] + bb[:, 2]) / 2)),
        float(np.median((bb[:, 1] + bb[:, 3]) / 2)),
    )
