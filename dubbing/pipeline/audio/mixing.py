"""
Audio helpers for the final mix (used by nodes/assemble.py).

  - read_stereo / read_mono / fit_length: load audio at a given sample rate.
  - balance_dialogue: keep each dubbed line as audible over the music as the
    original voice was.
  - render_songs: sung lines re-sung in the target language (song mode).
"""

import os
from pathlib import Path

from ..state import PipelineState, Segment

# How clear a dubbed line must sit over the background (see balance_dialogue):
# at least this many dB, lifting the dub by at most _MAX_LIFT and dipping the
# background by at most _MAX_DUCK.
_MIN_MARGIN = float(os.environ.get("DIALOGUE_MIN_MARGIN_DB", "8"))
_MAX_LIFT = float(os.environ.get("DIALOGUE_MAX_LIFT_DB", "6"))
_MAX_DUCK = float(os.environ.get("BED_MAX_DUCK_DB", "8"))


def read_stereo(path: str, target_sr: int | None = None):
    import numpy as np
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    if y.shape[1] == 1:
        y = np.repeat(y, 2, axis=1)
    elif y.shape[1] > 2:
        y = y[:, :2]
    if target_sr and sr != target_sr:
        import librosa

        y = librosa.resample(y.T, orig_sr=sr, target_sr=target_sr).T.astype(np.float32)
        sr = target_sr
    return y, sr


def read_mono(path: str, target_sr: int):
    import soundfile as sf

    y, sr = sf.read(path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if sr != target_sr:
        import librosa

        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
    return y


def fit_length(y, n: int):
    import numpy as np

    if len(y) >= n:
        return y[:n]
    return np.pad(y, ((0, n - len(y)), (0, 0)))


def balance_dialogue(mix, dubs: list, state: PipelineState, sr: int) -> int:
    """
    Keep every dubbed line as clear over the background as the original
    voice was (and at least _MIN_MARGIN dB clear), so loud music does not
    bury it. Where a line
    falls short, the dub is raised (≤ _MAX_LIFT dB) and the background is
    dipped just around it (≤ _MAX_DUCK dB, 100 ms ramps). Lines already clear
    — speech over quiet rooms — are untouched. Edits `dubs` gains in place,
    `mix` (background) in place; returns lines changed.
    """
    import numpy as np

    vocals_path = state.get("vocals_path")
    if not vocals_path or not Path(vocals_path).exists() or not dubs:
        return 0
    voc = read_mono(vocals_path, target_sr=sr)
    bg = mix.mean(axis=1)
    hop = int(0.05 * sr)

    def frames_db(x):
        n = len(x) // hop
        if n < 3:
            return None
        r = np.sqrt((x[: n * hop].reshape(n, hop) ** 2).mean(1)) + 1e-9
        return 20 * np.log10(r)

    duck = np.zeros(len(mix), dtype=np.float32)  # dB of dip wanted
    changed = 0
    for k, (a, y, seg) in enumerate(dubs):
        b = min(len(mix), a + len(y))
        ov = frames_db(voc[int(seg.start * sr) : int(seg.end * sr)])
        ob = frames_db(bg[int(seg.start * sr) : int(seg.end * sr)])
        dv, db = frames_db(y[: b - a]), frames_db(bg[a:b])
        if ov is None or ob is None or dv is None or db is None:
            continue
        orig_margin = np.percentile(ov, 90) - np.median(ob)
        dub_margin = np.percentile(dv, 90) - np.median(db)
        short = max(orig_margin + 1.0, _MIN_MARGIN) - dub_margin
        if short <= 0.5:
            continue
        lift = min(short, _MAX_LIFT)
        y *= 10 ** (lift / 20)
        dip = min(short - lift, _MAX_DUCK)
        if dip > 0.5:
            p = int(0.15 * sr)
            duck[max(0, a - p) : min(len(mix), b + p)] = np.maximum(
                duck[max(0, a - p) : min(len(mix), b + p)], dip
            )
        changed += 1
        dubs[k] = (a, y, seg)
    if duck.any():
        ramp = int(0.1 * sr)
        kernel = np.ones(ramp, dtype=np.float32) / ramp
        duck = np.convolve(duck, kernel, mode="same")
        mix *= (10 ** (-duck / 20))[:, None]
    return changed


def render_songs(
    segments: list[Segment], state: PipelineState, cache_dir: Path, sr: int, total: int
) -> list:
    """[(start sample, mono vocal)] for every sung piece song mode could render."""
    import numpy as np

    from ..music import song_mode

    sung = [s for s in segments if s.sung and s.keep_original and s.dubbed_text]
    if not sung or not song_mode.available() or not state.get("vocals_path"):
        return []
    voc = read_mono(state["vocals_path"], target_sr=sr)
    out = []
    for seg in sung:
        y = song_mode.render(seg, state, cache_dir, state["target_lang"].split("-")[0], sr)
        if y is None or not len(y):
            continue
        a = int(seg.start * sr)
        ref = voc[a : a + len(y)]
        g = np.sqrt(np.mean(ref**2) + 1e-12) / (np.sqrt(np.mean(y**2)) + 1e-9)
        out.append((a, (y * min(g, 8.0)).astype(np.float32)))
    return out
