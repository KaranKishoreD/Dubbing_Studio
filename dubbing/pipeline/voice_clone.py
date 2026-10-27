"""
Speaker voice cloning: make every dubbed line sound like the person on
screen, not like a stock TTS voice.

The TTS voice reads the line (it knows the language and pronounces it
right); Seed-VC (zero-shot voice conversion) then re-voices that reading in
the original speaker's timbre. This is SoniTranslate's "voice imitation"
approach. It is language-independent — words, timing and intonation come
from the TTS — and the output is exactly as long as the input, so fitting
and lip sync are unaffected.

Measured on an interview clip (ECAPA similarity to the real speaker):
stock TTS ≈ 0.0, OpenVoice V2 0.08–0.23, Seed-VC 0.39–0.67.

Every line of a speaker is converted against the SAME reference (up to 25 s
of their clean original lines from the Demucs vocal track), so the voice no
longer drifts from line to line. A conversion that changes the line's
length or no longer follows its loudness envelope is discarded and the TTS
clip is kept.

VOICE_CLONE=1 enables it. Runs in dubbing/SeedVC/.venv-vc as a worker.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np
import soundfile as sf

ENABLED = os.environ.get("VOICE_CLONE", "1") == "1"
_DIR = Path(__file__).resolve().parents[1] / "SeedVC"
_PY = _DIR / ".venv-vc" / "bin" / "python"
_WORKER = Path(__file__).resolve().parent / "seedvc_worker.py"
_REF_SECS = 25.0             # Seed-VC uses at most 25 s of reference
_MIN_ENV_CORR = 0.6          # measured 0.75–0.86 on good conversions
_MAX_F0_OFF_ST = 5.0         # clone pitch this far from the speaker → rejected
_GPU_NEED_GB = 4.0
_PAD = 0.25                  # seconds of silence around each line (see _padded)
_LATE_TAIL = 0.12            # how much later than the TTS a cloned line may end


def available() -> bool:
    return ENABLED and _PY.exists() and (_DIR / "inference.py").exists()


def references(segments, vocals_path: str, cache_dir: Path) -> dict[int, str]:
    """{speaker: reference wav}: their longest lines not overlapping anyone else."""
    voc, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
    voc = voc.mean(axis=1)
    ordered = sorted(segments, key=lambda s: s.start)
    clean: dict[int, list] = {}
    for s in ordered:
        if s.duration < 1.0:
            continue
        if any(o is not s and o.speaker != s.speaker and o.start < s.end and s.start < o.end
               for o in ordered):
            continue
        clean.setdefault(s.speaker or 0, []).append(s)
    out = {}
    gap = np.zeros(int(0.15 * sr), np.float32)
    for spk, segs in clean.items():
        target = _speaker_f0(segs)
        picked, secs = [], 0.0
        for s in sorted(segs, key=lambda s: -s.duration):
            if secs >= _REF_SECS * 2.5:
                break
            picked.append(s)
            secs += s.duration
        picked.sort(key=lambda s: s.start)
        key = hashlib.sha1(repr([(round(s.start, 2), round(s.end, 2)) for s in picked]
                                + [_REF_VERSION]).encode()).hexdigest()[:10]
        path = cache_dir / f"vc_ref_s{spk}_{key}.wav"
        if not path.exists():
            parts, kept = [], 0.0
            for s in picked:
                for c in _clean_chunks(voc[int(s.start * sr):int(s.end * sr)], sr, target):
                    parts += [c, gap]
                    kept += len(c) / sr
                if kept >= _REF_SECS:
                    break
            if not parts:          # nothing passed: fall back to the plain lines
                parts = [voc[int(s.start * sr):int(s.end * sr)] for s in picked]
            sf.write(str(path), np.concatenate(parts)[:int(_REF_SECS * sr)], sr)
        out[spk] = str(path)
    return out


_REF_VERSION = 2              # bump when reference selection changes
_F0_RANGE = {"male": (80, 175), "female": (160, 300)}


def _speaker_f0(segs) -> float | None:
    """Duration-weighted median line pitch, clamped to the gender's range."""
    v = sorted((s.f0, max(s.duration, 0.1)) for s in segs if s.f0)
    if not v:
        return None
    half, acc = sum(w for _, w in v) / 2, 0.0
    for f, w in v:
        acc += w
        if acc >= half:
            lo, hi = _F0_RANGE.get(segs[0].gender or "", (60, 400))
            return min(max(f, lo), hi)
    return None


def _clean_chunks(y: np.ndarray, sr: int, target: float | None) -> list[np.ndarray]:
    """
    1 s pieces of a line that sound like the speaker: mostly voiced and
    within 3 semitones of their usual pitch. Laughs, the other person's
    words left in by diarization, and shouts are left out — a reference
    that swung 108–289 Hz made the cloned voice swing with it.
    """
    import librosa
    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
    out = []
    for i in range(0, len(y16) - 8000, 16000):
        piece = y16[i:i + 16000]
        f0, voiced, _ = librosa.pyin(piece, fmin=60, fmax=400, sr=16000, frame_length=1024,
                                     hop_length=160)
        f = f0[voiced]
        if len(f) < 25:
            continue
        if target and abs(12 * np.log2(np.median(f) / target)) > 3:
            continue
        out.append(y[int(i * sr / 16000):int((i + 16000) * sr / 16000)])
    return out


_SAME_VOICE = float(os.environ.get("VC_SAME_VOICE_SIM", "0.6"))


def line_references(segments, vocals_path: str, cache_dir: Path) -> dict[int, str]:
    """
    {segment id: reference wav} built from voices that SOUND like that line.

    The speaker label can be wrong — over a loud film score pyannote merged
    several characters into one speaker and split one actor across several,
    so cloning against "the speaker's" reference gave a face the wrong voice,
    and one actor different voices. Here each line's own voice is
    fingerprinted (ECAPA), and its reference is that line plus the other
    lines that sound like the same person, keeping only clean stretches at
    that voice's pitch. With correct diarization this equals the speaker's
    own lines, as before.
    """
    import librosa
    import torch
    from speechbrain.inference.speaker import EncoderClassifier
    voc, sr = sf.read(vocals_path, dtype="float32", always_2d=True)
    voc = voc.mean(axis=1)
    ordered = sorted(segments, key=lambda s: s.start)
    usable = [s for s in ordered if s.duration >= 0.8 and not any(
        o is not s and o.start < s.end and s.start < o.end for o in ordered)]
    if len(usable) < 2:
        return {}
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    enc = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb",
                                         run_opts={"device": dev})
    embs = {}
    with torch.no_grad():
        for s in usable:
            y = voc[int(s.start * sr):int(s.end * sr)]
            y16 = librosa.resample(y, orig_sr=sr, target_sr=16000) if sr != 16000 else y
            e = enc.encode_batch(torch.tensor(y16)[None].to(dev)).squeeze()
            embs[s.id] = torch.nn.functional.normalize(e, dim=0).cpu().numpy()
    ids = list(embs)
    M = np.stack([embs[i] for i in ids])
    sim = M @ M.T
    by_id = {s.id: s for s in usable}
    gap = np.zeros(int(0.15 * sr), np.float32)
    out = {}
    for k, sid in enumerate(ids):
        order = [ids[j] for j in np.argsort(-sim[k]) if sim[k, j] >= _SAME_VOICE or ids[j] == sid]
        order = [sid] + [i for i in order if i != sid]
        own = by_id[sid]
        target = _median_f0(voc[int(own.start * sr):int(own.end * sr)], sr)
        key = hashlib.sha1(repr([(round(by_id[i].start, 2), round(by_id[i].end, 2)) for i in order[:12]]
                                + [_REF_VERSION, round(target or 0)]).encode()).hexdigest()[:10]
        path = cache_dir / f"vc_ref_l{sid}_{key}.wav"
        if not path.exists():
            parts, kept = [], 0.0
            for i in order:
                seg = by_id[i]
                for c in _clean_chunks(voc[int(seg.start * sr):int(seg.end * sr)], sr, target):
                    parts += [c, gap]
                    kept += len(c) / sr
                if kept >= _REF_SECS:
                    break
            if kept < 1.0:                 # too little clean voice: use the speaker's
                continue
            sf.write(str(path), np.concatenate(parts)[:int(_REF_SECS * sr)], sr)
        out[sid] = str(path)
    return out


def clone_lines(segments, vocals_path: str, cache_dir: Path, segs_dir: Path) -> tuple[dict, str]:
    """
    {segment id: cloned wav} for every dubbed line that converted well, plus a
    log note. Lines already converted for the same TTS clip and reference are
    reused from the cache.
    """
    refs = references(segments, vocals_path, cache_dir)
    try:
        by_line = line_references(segments, vocals_path, cache_dir)
    except Exception as exc:
        print(f"  ⚠ per-line voice references skipped ({exc}) — speaker references used")
        by_line = {}
    done, jobs = {}, []
    for s in segments:
        ref = by_line.get(s.id) or refs.get(s.speaker or 0)
        if not s.dubbed_audio or not ref or not os.path.exists(s.dubbed_audio):
            continue
        stem = os.path.splitext(os.path.basename(s.dubbed_audio))[0]
        out = segs_dir / f"{stem}_vc2_{Path(ref).stem.rsplit('_', 1)[-1]}.wav"
        if (out.parent / (out.name + ".rejected")).exists():
            continue                                  # failed the check before
        if out.exists():
            done[s.id] = str(out)
        else:
            jobs.append((s, {"source": _padded(s.dubbed_audio, segs_dir),
                             "target": str(Path(ref).resolve()), "out": str(out.resolve())}))
    failed = rejected = 0
    if jobs:
        results = _run([j for _, j in jobs], cache_dir)
        for s, j in jobs:
            r = results.get(j["out"], "worker crashed — see seedvc.log")
            if r != "ok" or not os.path.exists(j["out"]):
                failed += 1
                print(f"  ⚠ voice clone seg {s.id}: {r}")
                continue
            if not _postprocess(j["source"], j["out"], _ref_f0(j["target"])):
                rejected += 1
                os.replace(j["out"], j["out"] + ".rejected")
                continue
            done[s.id] = j["out"]
    note = (f"✓ Voices cloned to each original speaker (Seed-VC): {len(done)} line(s), "
            f"{len(refs)} speaker reference(s)")
    if failed or rejected:
        note += f"; {failed + rejected} kept the TTS voice ({rejected} failed the quality check)"
    return done, note


def _padded(tts_path: str, segs_dir: Path) -> str:
    """
    The TTS line with _PAD seconds of silence either side. Seed-VC returns
    exactly the input length but can place the speech a little later, so
    without room the last syllable fell off the end — lines stopped mid-sound
    (last 20 ms at −12…−20 dB, measured). align trims the silence again.
    """
    out = segs_dir / (Path(tts_path).stem + "_vcpad.wav")
    if not out.exists():
        y, sr = sf.read(tts_path, dtype="float32", always_2d=True)
        pad = np.zeros(int(_PAD * sr), np.float32)
        sf.write(str(out), np.concatenate([pad, y.mean(axis=1), pad]), sr)
    return str(out.resolve())


def _median_f0(y: np.ndarray, sr: int) -> float | None:
    import librosa
    if sr != 16000:
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=16000, frame_length=1024,
                                 hop_length=160)
    f = f0[voiced]
    return float(np.median(f)) if len(f) >= 10 else None


_REF_F0: dict[str, float | None] = {}


def _ref_f0(ref_path: str) -> float | None:
    """Median pitch of a speaker's reference clip (cached per run)."""
    if ref_path not in _REF_F0:
        y, sr = sf.read(ref_path, dtype="float32", always_2d=True)
        _REF_F0[ref_path] = _median_f0(y.mean(axis=1), sr)
    return _REF_F0[ref_path]


def _postprocess(tts_path: str, out_path: str, speaker_f0: float | None = None) -> bool:
    """Back to the TTS sample rate and level; False if the conversion is unusable."""
    import librosa
    y, sr = sf.read(tts_path, dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    z, zsr = sf.read(out_path, dtype="float32", always_2d=True)
    z = z.mean(axis=1)
    if zsr != sr:
        z = librosa.resample(z, orig_sr=zsr, target_sr=sr)
    if abs(len(z) - len(y)) > 0.02 * len(y) + 0.05 * sr:
        return False
    z = z[:len(y)] if len(z) >= len(y) else np.pad(z, (0, len(y) - len(z)))
    if _env_corr(y, z, sr) < _MIN_ENV_CORR:
        return False
    # Seed-VC is unstable on very short lines: a 0.8 s reply by a ~145 Hz
    # man came out at 216 Hz. A clone far from the speaker's own pitch is
    # discarded and the TTS line is kept.
    if speaker_f0:
        f = _median_f0(z, sr)
        if f and abs(12 * np.log2(f / speaker_f0)) > _MAX_F0_OFF_ST:
            print(f"  ⚠ clone off pitch ({f:.0f} Hz vs speaker {speaker_f0:.0f} Hz) — TTS kept")
            return False
    g = np.sqrt(np.mean(y ** 2) + 1e-12) / np.sqrt(np.mean(z ** 2) + 1e-12)
    z = z * min(g, 4.0)
    peak = float(np.max(np.abs(z))) if len(z) else 0.0
    if peak > 0.99:
        z = z * (0.99 / peak)
    # Back to the TTS line's own window, allowing the speech to land up to
    # _LATE_TAIL later. Seed-VC fills the silent padding with breath and
    # hiss, which made every line 0.35–0.5 s "longer" to the fitter.
    pad = int(_PAD * sr)
    a, b = max(0, pad - int(0.03 * sr)), min(len(z), len(z) - pad + int(_LATE_TAIL * sr))
    z = z[a:b].copy()
    f = min(int(0.03 * sr), len(z) // 4)
    if f > 0:
        z[-f:] *= np.linspace(1.0, 0.0, f, dtype=np.float32)
    sf.write(out_path, z.astype(np.float32), sr)
    return True


def _env_corr(a: np.ndarray, b: np.ndarray, sr: int) -> float:
    """Correlation of 20 ms loudness envelopes: same words at the same times."""
    hop = int(0.02 * sr)
    n = min(len(a), len(b)) // hop
    if n < 10:
        return 1.0
    ea = np.sqrt((a[:n * hop].reshape(n, hop) ** 2).mean(1))
    eb = np.sqrt((b[:n * hop].reshape(n, hop) ** 2).mean(1))
    return float(np.corrcoef(ea, eb)[0, 1])


def _run(jobs: list[dict], cache_dir: Path, extra_env: dict | None = None,
         tag: str = "seedvc") -> dict[str, str]:
    from .nodes.lipsync import _free_gpu_gb
    jobs_path = cache_dir / f"{tag}_jobs.json"
    results_path = cache_dir / f"{tag}_results.json"
    jobs_path.write_text(json.dumps(jobs), encoding="utf-8")
    results_path.unlink(missing_ok=True)
    env = dict(os.environ, **(extra_env or {}))
    if _free_gpu_gb() < _GPU_NEED_GB:
        env["CUDA_VISIBLE_DEVICES"] = ""          # shared GPU is full: CPU, slower
        print(f"  ⚠ voice cloning on CPU — under {_GPU_NEED_GB:g} GB GPU free")
    with open(cache_dir / f"{tag}.log", "w") as log:
        subprocess.run([str(_PY), str(_WORKER), str(jobs_path.resolve()),
                        str(results_path.resolve())],
                       cwd=str(_DIR), env=env, stdout=log, stderr=subprocess.STDOUT)
    try:
        return json.loads(results_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
