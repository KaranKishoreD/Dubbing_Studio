"""
Developer mode: what happened inside a run — stage timings, speakers and
their pitch, measured TTS speeds, cached files, a per-line inspector, the
active settings (keys hidden) and the raw log.
"""

import json
import os
from pathlib import Path

from pipeline.cache import CACHE_ROOT, get_cache_dir
from pipeline.state import Segment
from ui.startup import ENV_PATH, LOG_FILE
from ui.views import fmt_time

_DEBUG_NOISE = (
    "Warning",
    "warnings.warn",
    "allow_tf32",
    ">>> ",
    "reproducib",
    "issues/1370",
    "re-enabled",
    "std = sequences",
    "torch.backends",
)


def read_debug_log(n: int = 250) -> str:
    try:
        lines = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    except Exception:
        return ""
    lines = [l for l in lines if l.strip() and not any(k in l for k in _DEBUG_NOISE)]
    return "\n".join(lines[-n:])


# Artifacts each stage leaves in cache/<video_id>/ — present = reused next run.
_CACHE_MAP = [
    ("Download", ["video.mp4", "audio.wav", "audio_hq.wav", "meta.json"]),
    ("Separate", ["separated/*_vocals.wav", "separated/*_bed.wav"]),
    ("Transcribe", ["transcript.json", "bodhan_asr.json"]),
    ("Translate", ["translations.json"]),
    ("Voice", ["segments/s*_dubbed_*.wav"]),
    ("Fit timing", ["segments/*_fit_*.wav"]),
    ("Lip sync", ["faces.npz", "segments/s*_ls_*.mp4"]),
    ("Export", ["dubbed_*.mp4", "dubbed_*.m4a", "mixed_audio.wav"]),
]


def cache_md(video_id: str) -> str:
    if not video_id:
        return "_No project loaded._"
    d = get_cache_dir(video_id)
    rows = ["| Stage | Artifact | Files | Size |", "|---|---|---|---|"]
    for stage, pats in _CACHE_MAP:
        for pat in pats:
            hits = list(d.glob(pat))
            size = sum(f.stat().st_size for f in hits if f.is_file())
            mark = "✅" if hits else "—"
            rows.append(f"| {stage} | `{pat}` | {mark} {len(hits)} | {size / 1e6:.1f} MB |")
    total = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
    return (
        f"**cache/{video_id}/** — {total / 1e6:.0f} MB total. Stages whose "
        "artifacts exist are reused on the next run; delete a file to force "
        "that stage to run again.\n\n" + "\n".join(rows)
    )


def timings_md(timings: dict, stages) -> str:
    if not timings:
        return "_No run yet in this session._"
    rows = ["| Stage | Seconds |", "|---|---|"]
    label = dict(stages)
    rows += [f"| {label.get(k, k)} | {v:.1f} |" for k, v in timings.items()]
    rows.append(f"| **Total** | **{sum(timings.values()):.1f}** |")
    return "\n".join(rows)


def speakers_md(segs: list[dict]) -> str:
    if not segs:
        return "_No project loaded._"
    rows = [
        "| Speaker | Gender | Median F0 | Voice | Lines | Speech | Delivery mix |",
        "|---|---|---|---|---|---|---|",
    ]
    by: dict[int, list[Segment]] = {}
    for d in segs:
        s = Segment.from_dict(d)
        by.setdefault(s.speaker or 0, []).append(s)
    for spk, ss in sorted(by.items()):
        f0s = sorted(s.f0 for s in ss if s.f0)
        deliv: dict[str, int] = {}
        for s in ss:
            deliv[s.delivery or "neutral"] = deliv.get(s.delivery or "neutral", 0) + 1
        rows.append(
            f"| {spk + 1} | {ss[0].gender or '?'} | "
            f"{f0s[len(f0s) // 2] if f0s else '?'} Hz | {ss[0].voice or '—'} | "
            f"{len(ss)} | {sum(s.duration for s in ss):.0f}s | "
            + ", ".join(f"{k}×{v}" for k, v in sorted(deliv.items()))
            + " |"
        )
    return "\n".join(rows)


def rates_md() -> str:
    p = CACHE_ROOT / "_tts_rates.json"
    if not p.exists():
        return "_No measurements yet — written after the first synthesis._"
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = ["| Language · voice | chars / second |", "|---|---|"]
    rows += [f"| {k.replace('|', ' · ')} | {v} |" for k, v in sorted(d.items())]
    return (
        "Measured from real TTS output each run; sets every line's translation "
        "budget.\n\n" + "\n".join(rows)
    )


_CONFIG_KEYS = [
    "WHISPER_MODEL",
    "DIARIZE_BACKEND",
    "PYANNOTE_MODEL",
    "PYANNOTE_MERGE_SIM",
    "TRANSLATOR",
    "TRANSLATE_MODEL",
    "TRANSLATE_REGISTER",
    "EDIT_MODEL",
    "MAX_STRETCH",
    "HARD_MAX_STRETCH",
    "MIN_RATE",
    "REFIT_ROUNDS",
    "BUDGET_BORROW_SECS",
    "OVERRUN_TRIGGER",
    "LIPSYNC_BACKEND",
    "LATENTSYNC_STEPS",
    "SHOT_FACE_FRAC",
    "SEPARATE",
    "DEMUCS_MODEL",
    "ASR_TEXT",
    "LOUDNORM",
    "LUFS_TARGET",
]


_SECRET_KEYS = [
    "BODHAN_TRANSLATE_KEY",
    "BODHAN_SPEAK_KEY",
    "BODHAN_TRANSCRIBE_KEY",
    "OPENROUTER_API_KEY",
    "HF_TOKEN",
]


def config_md() -> str:
    rows = ["| Setting | Value |", "|---|---|"]
    rows += [f"| `{k}` | {os.environ.get(k) or '_(default)_'} |" for k in _CONFIG_KEYS]
    rows += [f"| `{k}` | {'🔑 set' if os.environ.get(k) else '❌ missing'} |" for k in _SECRET_KEYS]
    try:
        import torch

        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        gpu = "?"
    return f"GPU: **{gpu}** · `.env`: `{ENV_PATH}`\n\n" + "\n".join(rows)


def segment_choices(segs: list[dict]) -> list[tuple[str, int]]:
    out = []
    for d in sorted(segs, key=lambda d: d["start"]):
        s = Segment.from_dict(d)
        out.append(
            (
                f"S{s.id} · {fmt_time(s.start)} · Speaker {(s.speaker or 0) + 1} · {s.text[:50]}",
                s.id,
            )
        )
    return out


def inspect_segment(sid, segs: list[dict]):
    from pipeline.nodes.align import slot_seconds, trimmed_duration

    if sid is None or not segs:
        return "_Pick a line._", None, None, None, {}
    ordered = sorted([Segment.from_dict(d) for d in segs], key=lambda s: s.start)
    idx = next((i for i, s in enumerate(ordered) if s.id == sid), None)
    if idx is None:
        return "_Line not found._", None, None, None, {}
    s = ordered[idx]
    slot = slot_seconds(ordered, idx, None)
    facts = [
        f"**S{s.id}** · Speaker {(s.speaker or 0) + 1} ({s.gender}) · voice "
        f"**{s.voice}** · delivery {s.delivery or 'neutral'}",
        f"Original line {s.duration:.2f}s ({fmt_time(s.start)}–{fmt_time(s.end)}), slot {slot:.2f}s",
    ]
    if s.dubbed_audio and Path(s.dubbed_audio).exists():
        tts = trimmed_duration(s.dubbed_audio)
        need = tts / slot if slot else 0
        verdict = (
            "fits naturally"
            if need <= 1.0
            else "slight speed-up"
            if need <= 1.10
            else "noticeable speed-up"
            if need <= 1.25
            else "trimmed"
        )
        facts.append(
            f"TTS clip {tts:.2f}s → needs **{need:.2f}×** ({verdict}); "
            f"{len(s.dubbed_text or '')} chars"
        )
    clips = s.sync_clips or []
    facts.append(
        f"Lip-synced shots: {len(clips)}"
        + (
            " — " + ", ".join(f"{fmt_time(c['start'])}–{fmt_time(c['end'])}" for c in clips)
            if clips
            else " (speaker not on camera, or not facing it)"
        )
    )
    raw = {k: v for k, v in s.to_dict().items() if k != "words"}
    raw["words"] = f"{len(s.words or [])} word timings"
    clip = clips[0]["path"] if clips and Path(clips[0]["path"]).exists() else None
    return (
        "\n\n".join(facts),
        s.dubbed_audio if s.dubbed_audio and Path(s.dubbed_audio).exists() else None,
        s.aligned_audio if s.aligned_audio and Path(s.aligned_audio).exists() else None,
        clip,
        raw,
    )
