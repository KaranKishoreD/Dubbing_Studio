"""
Raga Dubbing Studio — Gradio UI
Run: python -m ui.app   (from the dubbing/ directory)

Studio view: what a customer sees — a project, a stage tracker, and the
workflow tabs Preview · Cast · Script · Revise · Export.
Developer mode (toggle in the header): stage timings, cache inspector,
per-line inspector, speaker profiles, measured TTS rates, config, raw logs.
"""

import sys
import os
import json
import time
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from project root (one level above dubbing/)
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# Fail fast with a clear message if keys are missing
for _key in ("BODHAN_TRANSLATE_KEY", "BODHAN_SPEAK_KEY", "OPENROUTER_API_KEY"):
    if not os.environ.get(_key):
        raise EnvironmentError(
            f"{_key} not set. Add it to your .env file at: {_env_path}"
        )

# Everything the pipeline prints (and every traceback) also goes to a log file
# the developer view shows, however the app was launched.
_LOG_FILE = Path(__file__).resolve().parent.parent.parent / "logs" / "ui_debug.log"
_LOG_FILE.parent.mkdir(exist_ok=True)


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, data):
        self.stream.write(data)
        self.fh.write(data)
        self.fh.flush()
        return len(data)

    def flush(self):
        self.stream.flush()
        self.fh.flush()

    def __getattr__(self, name):
        return getattr(self.stream, name)


_log_fh = open(_LOG_FILE, "a", encoding="utf-8", buffering=1)
sys.stdout = _Tee(sys.stdout, _log_fh)
sys.stderr = _Tee(sys.stderr, _log_fh)

import gradio as gr
from pipeline.state import PipelineState, Segment
from pipeline.cache import video_id_from_url, get_cache_dir, CACHE_ROOT, clip_id
from pipeline.graph import FULL_PIPELINE, EDIT_PIPELINE, REVOICE_PIPELINE
from pipeline.edit_planner import plan_edits, apply_plan
from pipeline.nodes.synthesize import (
    BODHAN_STYLES, DEFAULT_STYLE_LABEL, voice_catalogue, preview_voice, voice_profiles,
    voice_label, is_sarvam,
)
from pipeline.nodes.translate import LANG_NAMES


SUPPORTED_LANGS = {
    "Hindi (हिन्दी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Assamese (অসমীয়া)": "as",
    "Urdu (اردو)": "ur",
    "English": "en",
}
_LANG_LABEL = {v: k for k, v in SUPPORTED_LANGS.items()}

DOMAINS = [
    "education / lecture",
    "technology / CS",
    "general",
    "medicine",
    "law",
    "finance",
]

# ── Pipeline stages, as the customer sees them ────────────────────────────────
FULL_STAGES = [("download", "Download"), ("separate", "Separate"),
               ("transcribe", "Transcribe"), ("diarize", "Speakers"),
               ("refine", "Clean up"),
               ("translate", "Translate"), ("synthesize", "Voice"),
               ("align", "Fit timing"), ("lipsync", "Lip sync"),
               ("assemble", "Export")]
EDIT_STAGES = FULL_STAGES[5:]
REVOICE_STAGES = FULL_STAGES[6:]

# ── Voice casting ─────────────────────────────────────────────────────────────
# One row per detected speaker. Six covers interviews and panels; more than
# that and diarization itself is the weak link, not the UI.
MAX_CAST_ROWS = 12
AUTO_VOICE = "Auto"
_CATALOGUE = voice_catalogue()                     # 23 female + 22 male
_PROFILES = voice_profiles()                       # measured pitch per voice


def _choice(v: str, g: str, lang: str) -> tuple[str, str]:
    pitch = _PROFILES.get(v, {}).get("pitch")
    where = "Sarvam" if is_sarvam(v) else f"Bodhan · {LANG_NAMES.get(lang, lang)}"
    return (f"{voice_label(v).replace(' (Sarvam)', '')} — {g}"
            + (f" · {pitch}" if pitch else "") + f" · {where}", v)


_BY_PITCH = sorted(_CATALOGUE, key=lambda x: _PROFILES.get(x[0], {}).get("f0", 0))
VOICE_CHOICES = [AUTO_VOICE] + [
    _choice(v, g, l)
    for engine in ("sarvam", "bodhan") for g_order in ("female", "male")
    for v, g, l in _BY_PITCH if g == g_order and is_sarvam(v) == (engine == "sarvam")
]
_HAS_SARVAM = bool(os.environ.get("SARVAM_API_KEY"))

# A short line per target language for voice previews.
_PREVIEW_TEXT = {
    "hi": "नमस्ते, मैं इस वीडियो की आवाज़ हूँ।",
    "ta": "வணக்கம், நான் இந்த வீடியோவின் குரல்.",
    "te": "నమస్కారం, నేను ఈ వీడియో గొంతు.",
    "kn": "ನಮಸ್ಕಾರ, ನಾನು ಈ ವೀಡಿಯೊದ ಧ್ವನಿ.",
    "ml": "നമസ്കാരം, ഞാനാണ് ഈ വീഡിയോയുടെ ശബ്ദം.",
    "bn": "নমস্কার, আমি এই ভিডিওর কণ্ঠ।",
    "mr": "नमस्कार, मी या व्हिडिओचा आवाज आहे.",
    "gu": "નમસ્તે, હું આ વિડિયોનો અવાજ છું.",
    "pa": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ, ਮੈਂ ਇਸ ਵੀਡੀਓ ਦੀ ਆਵਾਜ਼ ਹਾਂ।",
    "or": "ନମସ୍କାର, ମୁଁ ଏହି ଭିଡିଓର ସ୍ୱର।",
    "as": "নমস্কাৰ, মই এই ভিডিঅ'ৰ কণ্ঠ।",
    "ur": "السلام علیکم، میں اس ویڈیو کی آواز ہوں۔",
    "en": "Hello, I am the voice of this video.",
}


# ── Pipeline state ────────────────────────────────────────────────────────────

def _base_state(video_id, target_lang, domain, voice_name, voice_style, voice_map,
                lang_mix: str = "standard", tts_backend: str = "bodhan",
                translator: str = "indic", songs: str = "keep") -> PipelineState:
    return {
        "songs": songs,
        "lang_mix": lang_mix,
        "tts_backend": tts_backend,
        "translator": translator,
        "url": "",
        "target_lang": target_lang,
        "domain": domain,
        "video_id": video_id,
        "video_path": None,
        "audio_path": None,
        "audio_hq_path": None,
        "vocals_path": None,
        "background_path": None,
        "voice_ref_path": None,
        "segments": [],
        "final_video": None,
        "final_audio": None,
        "edit_ops": [],
        "log": [],
        "error": None,
        "voice_name": (voice_name or "").strip() or None,
        "voice_style": voice_style,
        "speaker_profiles": {},
        "num_speakers": None,
        "voice_map": voice_map or {},
    }


def _cached_state(video_id, target_lang, domain, voice_name, voice_style, voice_map,
                  segments_state, edit_ops, lang_mix: str = "standard",
                  tts_backend: str = "bodhan") -> PipelineState | str:
    """State for re-running part of the pipeline on a processed video."""
    cache_dir = get_cache_dir(video_id)
    # Prefer the Demucs stems; fall back to the raw-audio copy.
    sep_dir = cache_dir / "separated"
    stems = sorted(sep_dir.glob("*_vocals.wav"))
    vocals = stems[0] if stems else sep_dir / "vocals.wav"
    bed = vocals.with_name(vocals.name.replace("_vocals.wav", "_bed.wav"))
    background = bed if stems and bed.exists() else None
    audio_hq = cache_dir / "audio_hq.wav"
    if not vocals.exists():
        return "✗ Cache missing — process the video again."
    state = _base_state(video_id, target_lang, domain, voice_name, voice_style, voice_map,
                        lang_mix, tts_backend)
    state.update({
        "video_path": str(cache_dir / "video.mp4"),
        "audio_path": str(cache_dir / "audio.wav"),
        "audio_hq_path": str(audio_hq) if audio_hq.exists() else None,
        "vocals_path": str(vocals.resolve()),
        "background_path": str(background.resolve()) if background else None,
        "voice_ref_path": str(cache_dir / "voice_ref.wav"),
        "segments": segments_state,
        "edit_ops": edit_ops,
    })
    return state


def _stream(graph, state, stages):
    """
    Run a graph on a worker thread and yield a progress dict every second:
      {"kind": "progress"|"done"|"error", "current": node or None,
       "timings": {node: seconds}, "log": text, "message": str, "final": state}
    Ticking every second keeps the elapsed time moving during long stages.
    """
    import queue
    import threading
    import traceback

    q: queue.Queue = queue.Queue()
    keys = [k for k, _ in stages]
    t0 = time.time()

    def work():
        final_state = state
        last = time.time()
        try:
            for event in graph.stream(state):
                node_name, node_output = next(iter(event.items()))
                now = time.time()
                final_state = {**final_state, **node_output}
                q.put(("node", node_name, now - last, final_state))
                last = now
                if node_output.get("error"):
                    q.put(("error", f"{node_name}: {node_output['error']}", 0, final_state))
                    return
            q.put(("done", None, 0, final_state))
        except Exception as exc:
            traceback.print_exc()
            q.put(("error", f"{type(exc).__name__}: {exc}", 0, final_state))

    threading.Thread(target=work, daemon=True).start()
    timings: dict[str, float] = {}
    final, log = state, ""
    while True:
        try:
            kind, name, secs, st = q.get(timeout=1.0)
        except queue.Empty:
            kind = "tick"
        if kind in ("node", "done", "error"):
            final = st
            log = "\n".join(st.get("log", []))
        if kind == "node":
            timings[name] = round(secs, 1)
        current = next((k for k in keys if k not in timings), None)
        info = {"kind": "progress", "current": current, "timings": dict(timings),
                "log": log, "elapsed": time.time() - t0, "final": final, "message": ""}
        if kind == "done":
            info.update(kind="done", current=None)
        elif kind == "error":
            info.update(kind="error", message=name)
            print(f"✗ Run failed: {name}", flush=True)
        yield info
        if kind in ("done", "error"):
            return


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _tracker_html(stages, timings: dict, current: str | None, error: bool = False,
                  elapsed: float | None = None) -> str:
    chips = []
    for key, label in stages:
        if key in timings:
            cls, sub = "done", f"{timings[key]:.0f}s" if timings[key] >= 1 else "<1s"
        elif key == current:
            cls, sub = ("err", "failed") if error else ("run", "working…")
        else:
            cls, sub = "todo", ""
        chips.append(f'<div class="stage {cls}"><div class="dot"></div>'
                     f'<div class="lbl">{label}</div><div class="sub">{sub}</div></div>')
    total = (f'<div class="elapsed">{elapsed:.0f}s elapsed</div>'
             if elapsed is not None else "")
    return f'<div class="tracker">{"".join(chips)}</div>{total}'


def _idle_tracker() -> str:
    return _tracker_html(FULL_STAGES, {}, None)


def _meta(video_id: str, url: str | None = None) -> dict:
    """Title etc. from download; fetched once for videos cached before that existed."""
    if not video_id:
        return {}
    p = get_cache_dir(video_id) / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    if url:
        try:
            import yt_dlp
            with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False) or {}
            meta = {"url": url, "title": info.get("title"),
                    "channel": info.get("channel") or info.get("uploader"),
                    "duration": info.get("duration"), "thumbnail": info.get("thumbnail")}
            p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return meta
        except Exception:
            pass
    return {}


def _project_html(video_id: str, lang: str | None, segs: list[dict], url: str | None = None) -> str:
    if not video_id:
        return ('<div class="project empty">No project yet — paste a YouTube link and '
                'press <b>Dub video</b>.</div>')
    m = _meta(video_id, url)
    objs = [Segment.from_dict(d) for d in segs]
    speakers = len({(s.speaker or 0) for s in objs})
    speech = sum(s.duration for s in objs)
    dur = m.get("duration")
    thumb = (f'<img src="{m["thumbnail"]}" alt="">' if m.get("thumbnail") else "")
    facts = [f"→ {_LANG_LABEL.get(lang, lang or '?')}",
             f"{speakers} speaker{'s' if speakers != 1 else ''}" if objs else None,
             f"{len(objs)} lines · {speech:.0f}s speech" if objs else None,
             f"{int(dur) // 60}:{int(dur) % 60:02d}" if dur else None]
    return (f'<div class="project">{thumb}<div><div class="ptitle">'
            f'{m.get("title") or video_id}</div><div class="pmeta">'
            f'{m.get("channel") or ""}</div><div class="pfacts">'
            + "".join(f"<span>{f}</span>" for f in facts if f) + "</div></div></div>")


def _segments_to_df(segments: list[Segment]) -> list[list]:
    return [
        [s.id, f"{_ts(s.start)}", f"Speaker {(s.speaker or 0) + 1}", voice_label(s.voice),
         ("⚠ " if s.asr_flag == "low" else "＋ " if s.asr_flag == "gap" else "") + s.text,
         "— (left undubbed)" if s.keep_original else (s.dubbed_text or "—")]
        for s in sorted(segments, key=lambda s: s.start)
    ]


def _ts(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:04.1f}"


def _cast_from_segments(segs: list[dict]) -> list[dict]:
    out: dict[int, dict] = {}
    for d in segs:
        s = Segment.from_dict(d)
        spk = s.speaker or 0
        row = out.setdefault(spk, {"speaker": spk, "gender": s.gender or "?",
                                   "voice": s.voice, "seconds": 0.0, "lines": 0})
        row["seconds"] += s.duration
        row["lines"] += 1
        row["voice"] = row["voice"] or s.voice
    return [out[k] for k in sorted(out)]


def _cast_updates(segs: list[dict]) -> list:
    """Label + dropdown + preview button per cast row, hiding unused rows."""
    cast = _cast_from_segments(segs) if segs else []
    ups = []
    for k in range(MAX_CAST_ROWS):
        if k < len(cast):
            c = cast[k]
            label = (f"**Speaker {k + 1}** · {c['gender']} · "
                     f"{c['lines']} lines · {c['seconds']:.0f}s")
            ups += [gr.update(value=label, visible=True),
                    gr.update(value=c["voice"] or AUTO_VOICE, visible=True),
                    gr.update(visible=True)]
        else:
            ups += [gr.update(visible=False)] * 3
    return ups


def _placeholder_rows(nspk) -> list:
    """Before processing: as many rows as the speaker count says (1 on Auto)."""
    n = int(nspk) if str(nspk).isdigit() else 1
    note = "" if str(nspk).isdigit() else " — more appear once speakers are detected"
    ups = []
    for k in range(MAX_CAST_ROWS):
        ups += [gr.update(value=f"**Speaker {k + 1}** · detected after processing{note}",
                          visible=k < n),
                gr.update(visible=k < n), gr.update(visible=k < n)]
    return ups


def _parse_time(text) -> float | None:
    """'49', '0:49', '1:02:03' → seconds; blank → None."""
    t = str(text or "").strip()
    if not t:
        return None
    try:
        parts = [float(x) for x in t.split(":")]
    except ValueError:
        raise ValueError(f"Can't read the time “{t}” — use m:ss, h:mm:ss or seconds")
    if len(parts) > 3 or any(p < 0 for p in parts):
        raise ValueError(f"Can't read the time “{t}” — use m:ss, h:mm:ss or seconds")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _voice_map(*voices) -> dict:
    return {str(k): v for k, v in enumerate(voices) if v and v != AUTO_VOICE}


# ── Edits ─────────────────────────────────────────────────────────────────────

_STYLES = sorted({v for v in BODHAN_STYLES.values() if v != "auto"})
_ACTION_LABEL = {
    "retranslate": "Re-translate", "set_text": "New wording", "recast": "Change voice",
    "set_gender": "Fix gender", "reassign_speaker": "Fix speaker", "set_style": "Delivery",
    "keep_original": "Keep original", "dub_again": "Dub again",
}


def plan_changes(message: str, segs: list, lang: str):
    """Revise tab: message → plan (state), plan table rows, status."""
    if not segs:
        return {}, [], "⚠ Dub a video first."
    if not message.strip():
        return {}, [], "⚠ Describe the change you want."
    code = SUPPORTED_LANGS.get(lang, "hi")
    try:
        plan = plan_edits(message.strip(), [Segment.from_dict(d) for d in segs],
                          LANG_NAMES.get(code, code), _CATALOGUE, _STYLES)
    except Exception as e:
        print(f"✗ Edit planning failed: {type(e).__name__}: {e}", flush=True)
        return {}, [], f"✗ Could not plan that: {e}"
    print(f"=== Edit plan for: {message.strip()}\n{plan['raw']}", flush=True)
    rows, unhandled = [], []
    for r in plan["requests"]:
        for act in r["actions"]:
            rows.append([r["request"], _ACTION_LABEL.get(act["type"], act["type"]),
                         _action_target(act), _action_detail(act)])
        if not r["actions"]:
            unhandled.append(f"“{r['request']}” — {r['note'] or 'no action fits'}"
                             + (f" ({'; '.join(r['rejected'])})" if r["rejected"] else ""))
    n = len(plan["requests"])
    status = (f"✓ Understood **{n} request{'s' if n != 1 else ''}** → "
              f"{len(rows)} change{'s' if len(rows) != 1 else ''}. Review, then **Apply**.")
    if unhandled:
        status += "\n\n⚠ **Not handled:**\n" + "\n".join(f"- {u}" for u in unhandled)
    return plan, rows, status


def _action_target(a: dict) -> str:
    if "lines" in a:
        return ", ".join(f"S{i}" for i in a["lines"])
    if "line" in a:
        return f"S{a['line']}"
    return f"Speaker {a.get('speaker')}"


def _action_detail(a: dict) -> str:
    t = a["type"]
    return {"retranslate": a.get("instruction", ""), "set_text": a.get("text", ""),
            "recast": a.get("voice", ""), "set_gender": a.get("gender", ""),
            "reassign_speaker": f"→ Speaker {a.get('speaker')}",
            "set_style": a.get("style", "")}.get(t, "")


def _diff_md(before: list[dict], after: list[dict]) -> str:
    """What actually changed, line by line."""
    old = {d["id"]: Segment.from_dict(d) for d in before}
    rows = []
    for d in sorted(after, key=lambda d: d["start"]):
        s, o = Segment.from_dict(d), old.get(d["id"])
        if not o:
            continue
        bits = []
        if (s.dubbed_text or "") != (o.dubbed_text or ""):
            bits.append(f"“{o.dubbed_text or '—'}” → “{s.dubbed_text or '—'}”")
        if s.voice != o.voice and not s.keep_original:
            bits.append(f"voice {o.voice} → {s.voice}")
        if s.speaker != o.speaker:
            bits.append(f"Speaker {(o.speaker or 0) + 1} → {(s.speaker or 0) + 1}")
        if s.style_override != o.style_override:
            bits.append(f"delivery → {s.style_override}")
        if s.keep_original != o.keep_original:
            bits.append("now undubbed" if s.keep_original else "dubbed again")
        if bits:
            rows.append(f"- **S{s.id}** ({_ts(s.start)}): " + "; ".join(bits))
    return "\n".join(rows) if rows else "_No visible changes._"


# ── Export ────────────────────────────────────────────────────────────────────

def _srt_time(t: float) -> str:
    ms = int(round(t * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _write_srt(segs: list[Segment], path: Path, dubbed: bool) -> Path:
    lines = []
    for n, s in enumerate(sorted(segs, key=lambda s: s.start), 1):
        text = (s.dubbed_text if dubbed else s.text) or ""
        if text.strip():
            lines += [str(n), f"{_srt_time(s.start)} --> {_srt_time(s.end)}", text.strip(), ""]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _exports(video_id: str, lang: str, segs: list[dict], final_state: dict | None) -> list[str]:
    if not video_id or not segs:
        return []
    cache_dir = get_cache_dir(video_id)
    objs = [Segment.from_dict(d) for d in segs]
    files = []
    for key in ("final_video", "final_audio"):
        p = (final_state or {}).get(key)
        if p and Path(p).exists():
            files.append(p)
    if not files:
        for name in (f"dubbed_{lang}.mp4", f"dubbed_{lang}.m4a"):
            if (cache_dir / name).exists():
                files.append(str(cache_dir / name))
    files.append(str(_write_srt(objs, cache_dir / f"subtitles_{lang}.srt", dubbed=True)))
    files.append(str(_write_srt(objs, cache_dir / "subtitles_original.srt", dubbed=False)))
    return files


# ── Developer view ────────────────────────────────────────────────────────────

_DEBUG_NOISE = ("Warning", "warnings.warn", "allow_tf32", ">>> ", "reproducib",
                "issues/1370", "re-enabled", "std = sequences", "torch.backends")


def _read_debug_log(n: int = 250) -> str:
    try:
        lines = _LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
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


def _cache_md(video_id: str) -> str:
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
    return (f"**cache/{video_id}/** — {total / 1e6:.0f} MB total. Stages whose "
            "artifacts exist are reused on the next run; delete a file to force "
            "that stage to run again.\n\n" + "\n".join(rows))


def _timings_md(timings: dict, stages) -> str:
    if not timings:
        return "_No run yet in this session._"
    rows = ["| Stage | Seconds |", "|---|---|"]
    label = dict(stages)
    rows += [f"| {label.get(k, k)} | {v:.1f} |" for k, v in timings.items()]
    rows.append(f"| **Total** | **{sum(timings.values()):.1f}** |")
    return "\n".join(rows)


def _speakers_md(segs: list[dict]) -> str:
    if not segs:
        return "_No project loaded._"
    rows = ["| Speaker | Gender | Median F0 | Voice | Lines | Speech | Delivery mix |",
            "|---|---|---|---|---|---|---|"]
    by: dict[int, list[Segment]] = {}
    for d in segs:
        s = Segment.from_dict(d)
        by.setdefault(s.speaker or 0, []).append(s)
    for spk, ss in sorted(by.items()):
        f0s = sorted(s.f0 for s in ss if s.f0)
        deliv: dict[str, int] = {}
        for s in ss:
            deliv[s.delivery or "neutral"] = deliv.get(s.delivery or "neutral", 0) + 1
        rows.append(f"| {spk + 1} | {ss[0].gender or '?'} | "
                    f"{f0s[len(f0s) // 2] if f0s else '?'} Hz | {ss[0].voice or '—'} | "
                    f"{len(ss)} | {sum(s.duration for s in ss):.0f}s | "
                    + ", ".join(f"{k}×{v}" for k, v in sorted(deliv.items())) + " |")
    return "\n".join(rows)


def _rates_md() -> str:
    p = CACHE_ROOT / "_tts_rates.json"
    if not p.exists():
        return "_No measurements yet — written after the first synthesis._"
    d = json.loads(p.read_text(encoding="utf-8"))
    rows = ["| Language · voice | chars / second |", "|---|---|"]
    rows += [f"| {k.replace('|', ' · ')} | {v} |" for k, v in sorted(d.items())]
    return ("Measured from real TTS output each run; sets every line's translation "
            "budget.\n\n" + "\n".join(rows))


_CONFIG_KEYS = [
    "WHISPER_MODEL", "DIARIZE_BACKEND", "PYANNOTE_MODEL", "PYANNOTE_MERGE_SIM",
    "TRANSLATOR", "TRANSLATE_MODEL", "TRANSLATE_REGISTER", "EDIT_MODEL",
    "MAX_STRETCH", "HARD_MAX_STRETCH", "MIN_RATE", "REFIT_ROUNDS", "BUDGET_BORROW_SECS",
    "OVERRUN_TRIGGER", "LIPSYNC_BACKEND", "LATENTSYNC_STEPS", "SHOT_FACE_FRAC",
    "SEPARATE", "DEMUCS_MODEL", "ASR_TEXT", "LOUDNORM", "LUFS_TARGET",
]
_SECRET_KEYS = ["BODHAN_TRANSLATE_KEY", "BODHAN_SPEAK_KEY", "BODHAN_TRANSCRIBE_KEY",
                "OPENROUTER_API_KEY", "HF_TOKEN"]


def _config_md() -> str:
    rows = ["| Setting | Value |", "|---|---|"]
    rows += [f"| `{k}` | {os.environ.get(k) or '_(default)_'} |" for k in _CONFIG_KEYS]
    rows += [f"| `{k}` | {'🔑 set' if os.environ.get(k) else '❌ missing'} |"
             for k in _SECRET_KEYS]
    try:
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "none"
    except Exception:
        gpu = "?"
    return f"GPU: **{gpu}** · `.env`: `{_env_path}`\n\n" + "\n".join(rows)


def _segment_choices(segs: list[dict]) -> list[tuple[str, int]]:
    out = []
    for d in sorted(segs, key=lambda d: d["start"]):
        s = Segment.from_dict(d)
        out.append((f"S{s.id} · {_ts(s.start)} · Speaker {(s.speaker or 0) + 1} · "
                    f"{s.text[:50]}", s.id))
    return out


def _inspect_segment(sid, segs: list[dict]):
    from pipeline.nodes.align import slot_seconds, trimmed_duration
    if sid is None or not segs:
        return "_Pick a line._", None, None, None, {}
    ordered = sorted([Segment.from_dict(d) for d in segs], key=lambda s: s.start)
    idx = next((i for i, s in enumerate(ordered) if s.id == sid), None)
    if idx is None:
        return "_Line not found._", None, None, None, {}
    s = ordered[idx]
    slot = slot_seconds(ordered, idx, None)
    facts = [f"**S{s.id}** · Speaker {(s.speaker or 0) + 1} ({s.gender}) · voice "
             f"**{s.voice}** · delivery {s.delivery or 'neutral'}",
             f"Original line {s.duration:.2f}s ({_ts(s.start)}–{_ts(s.end)}), slot {slot:.2f}s"]
    if s.dubbed_audio and Path(s.dubbed_audio).exists():
        tts = trimmed_duration(s.dubbed_audio)
        need = tts / slot if slot else 0
        verdict = ("fits naturally" if need <= 1.0 else
                   "slight speed-up" if need <= 1.10 else
                   "noticeable speed-up" if need <= 1.25 else "trimmed")
        facts.append(f"TTS clip {tts:.2f}s → needs **{need:.2f}×** ({verdict}); "
                     f"{len(s.dubbed_text or '')} chars")
    clips = s.sync_clips or []
    facts.append(f"Lip-synced shots: {len(clips)}" + (
        " — " + ", ".join(f"{_ts(c['start'])}–{_ts(c['end'])}" for c in clips) if clips else
        " (speaker not on camera, or not facing it)"))
    raw = {k: v for k, v in s.to_dict().items() if k != "words"}
    raw["words"] = f"{len(s.words or [])} word timings"
    clip = clips[0]["path"] if clips and Path(clips[0]["path"]).exists() else None
    return ("\n\n".join(facts),
            s.dubbed_audio if s.dubbed_audio and Path(s.dubbed_audio).exists() else None,
            s.aligned_audio if s.aligned_audio and Path(s.aligned_audio).exists() else None,
            clip, raw)


# ── Library: every finished dub, playable ─────────────────────────────────────

def _library() -> list[tuple[str, str]]:
    """
    (label, dubbed mp4) per project, newest first. A project dubbed into
    several languages, or dubbed again, shows its most recent dub.
    """
    rows = []
    for d in CACHE_ROOT.iterdir() if CACHE_ROOT.exists() else []:
        dubs = list(d.glob("dubbed_*.mp4")) if d.is_dir() else []
        if not dubs:
            continue
        latest = max(dubs, key=lambda p: p.stat().st_mtime)
        try:
            title = json.loads((d / "meta.json").read_text(encoding="utf-8")).get("title") or d.name
        except Exception:
            title = d.name
        lang = latest.stem.split("_", 1)[1]
        when = time.strftime("%d %b %H:%M", time.localtime(latest.stat().st_mtime))
        test = " (test copy)" if d.name.endswith(("_top3test", "_cpuclone")) else ""
        rows.append((latest.stat().st_mtime,
                     f"{title[:60]} — {LANG_NAMES.get(lang, lang)} — {when}{test}", str(latest)))
    return [(label, path) for _, label, path in sorted(rows, reverse=True)]


def _library_pick(path: str | None):
    if not path or not Path(path).exists():
        return None, None
    original = Path(path).parent / "video.mp4"
    return path, (str(original) if original.exists() else None)


# ── Look & feel ───────────────────────────────────────────────────────────────

THEME = gr.themes.Soft(primary_hue="indigo", secondary_hue="slate", neutral_hue="slate",
                       font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"])

CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#brand {display:flex; align-items:center; gap:14px; padding:6px 2px;}
#brand .logo {width:40px; height:40px; border-radius:10px; display:grid; place-items:center;
  background:linear-gradient(135deg,#4f46e5,#9333ea); color:#fff; font-weight:700; font-size:20px;}
#brand .name {font-size:20px; font-weight:700; letter-spacing:-0.01em;}
#brand .tag {font-size:13px; opacity:.65;}
.card {border:1px solid var(--border-color-primary); border-radius:14px; padding:14px !important;}
.project {display:flex; gap:14px; align-items:center; min-height:84px;}
.project img {width:150px; border-radius:10px; object-fit:cover;}
.project.empty {opacity:.6; font-size:15px;}
.ptitle {font-size:17px; font-weight:650; line-height:1.3;}
.pmeta {font-size:13px; opacity:.65; margin:2px 0 8px;}
.pfacts span {display:inline-block; font-size:12px; padding:3px 9px; margin:0 6px 4px 0;
  border-radius:999px; background:var(--background-fill-secondary);
  border:1px solid var(--border-color-primary);}
.tracker {display:flex; gap:6px; flex-wrap:wrap; margin-top:6px;}
.stage {flex:1 1 92px; min-width:86px; padding:8px 8px 7px; border-radius:10px;
  border:1px solid var(--border-color-primary); text-align:center;}
.stage .dot {width:10px; height:10px; border-radius:50%; margin:0 auto 5px; background:#cbd5e1;}
.stage .lbl {font-size:12.5px; font-weight:600;}
.stage .sub {font-size:11px; opacity:.65; min-height:14px;}
.stage.done {background:rgba(16,185,129,.08); border-color:rgba(16,185,129,.45);}
.stage.done .dot {background:#10b981;}
.stage.run {background:rgba(79,70,229,.08); border-color:#6366f1;}
.stage.run .dot {background:#6366f1; animation:pulse 1s infinite;}
.stage.err {background:rgba(239,68,68,.08); border-color:#ef4444;}
.stage.err .dot {background:#ef4444;}
.elapsed {font-size:12px; opacity:.6; text-align:right; margin-top:4px;}
@keyframes pulse {50% {opacity:.25}}
#status {min-height:24px;}
.castrow {align-items:center;}
"""

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Raga Dubbing Studio") as demo:

    segments_state = gr.State([])
    edit_ops_state = gr.State([])
    video_id_state = gr.State("")
    lang_state = gr.State("")
    timings_state = gr.State({})
    final_state_box = gr.State({})
    before_edit_state = gr.State([])

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        gr.HTML('<div id="brand"><div class="logo">R</div><div><div class="name">'
                'Raga Dubbing Studio</div><div class="tag">Lip-synced dubbing into '
                'Indian languages</div></div></div>')
        dev_mode = gr.Checkbox(label="Developer mode", value=False, scale=0, min_width=170)

    # ── New project + project status ──────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=4, min_width=330, elem_classes="card"):
            gr.Markdown("#### New project")
            url_input = gr.Textbox(label="YouTube link",
                                   placeholder="https://www.youtube.com/watch?v=…")
            with gr.Row():
                clip_start_in = gr.Textbox(label="Start (optional)", placeholder="0:00",
                                           info="m:ss, h:mm:ss or seconds")
                clip_end_in = gr.Textbox(label="End (optional)", placeholder="blank = to the end",
                                         info="Only this part is downloaded and dubbed")
            with gr.Row():
                target_lang = gr.Dropdown(choices=list(SUPPORTED_LANGS.keys()),
                                          value="Hindi (हिन्दी)", label="Dub into")
                num_speakers_dropdown = gr.Dropdown(
                    choices=["Auto"] + [str(n) for n in range(1, 13)], value="Auto",
                    label="Speakers", info="Set it if you know — most reliable")
            lang_mix = gr.Radio(
                choices=[("Standard (as before)", "standard"), ("Pure", "pure"),
                         ("Natural spoken", "natural"),
                         ("Heavy mix (Tanglish/Hinglish)", "heavy")],
                value="standard", label="English in the dub",
                info="How many English words stay, the way real speakers talk")
            pipeline_mode = gr.Radio(
                choices=[("LLM translation (context-aware)", "llm"),
                         ("Indic: Bodhan translation, LLM only for lines it fails", "indic")],
                value="indic", label="Pipeline")
            songs_mode = gr.Radio(
                choices=[("Keep the original singing", "keep"),
                         ("Re-sing in the dub language (experimental)", "resing")],
                value="keep", label="Songs",
                info="Re-singing composes a new tune (~$0.04 per phrase); the original melody is not kept yet")
            tts_engine = gr.Radio(
                choices=[("Bodhan Indic-Speak", "bodhan"), ("Sarvam Bulbul v3", "sarvam")],
                value="bodhan", label="Voice engine",
                info=("Sarvam: 11 languages (hi bn ta te kn ml mr gu pa or en) — other "
                      "languages use Bodhan automatically. Any voice can still be picked "
                      "per speaker in Cast" if _HAS_SARVAM else
                      "Add SARVAM_API_KEY to .env to enable Sarvam"),
                interactive=_HAS_SARVAM)
            with gr.Accordion("Advanced", open=False):
                domain = gr.Dropdown(choices=DOMAINS, value="education / lecture",
                                     label="Content domain (keeps technical terms)")
                voice_style_dropdown = gr.Dropdown(choices=list(BODHAN_STYLES.keys()),
                                                   value=DEFAULT_STYLE_LABEL,
                                                   label="Speaking style")
                voice_name_input = gr.Textbox(
                    label="One voice for everyone (optional)",
                    placeholder="Blank = cast each speaker (see Cast tab)")
            process_btn = gr.Button("▶ Dub video", variant="primary", size="lg")
            gr.Markdown("<small>Re-dubbing a video reuses every finished stage — only "
                        "what changed is recomputed.</small>")

        with gr.Column(scale=7, elem_classes="card"):
            project_html = gr.HTML(_project_html("", None, []))
            tracker_html = gr.HTML(_idle_tracker())
            status_md = gr.Markdown("", elem_id="status")

    # ── Workflow tabs ─────────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("🎬 Preview"):
            with gr.Row():
                original_video = gr.Video(label="Original", interactive=False)
                dubbed_video = gr.Video(label="Dubbed", interactive=False)
            dubbed_audio = gr.Audio(label="Dubbed soundtrack", interactive=False,
                                    type="filepath")

        with gr.Tab("📚 Library"):
            with gr.Row():
                library_pick = gr.Dropdown(label="Dubbed videos — latest dub of each",
                                           choices=_library(), scale=6)
                library_refresh = gr.Button("↻ Refresh", scale=1)
            with gr.Row():
                library_dubbed = gr.Video(label="Dubbed", interactive=False)
                library_original = gr.Video(label="Original", interactive=False)

        with gr.Tab("🎙 Cast"):
            gr.Markdown("Every speaker gets one consistent voice, cast automatically "
                        "(gender from pitch, distinct voices). Choose any of the 45 "
                        "voices, preview it, then **Re-voice** — the translation is kept.")
            cast_labels, cast_voices, cast_previews = [], [], []
            for k in range(MAX_CAST_ROWS):
                with gr.Row(equal_height=True, elem_classes="castrow"):
                    cast_labels.append(gr.Markdown(
                        f"**Speaker {k + 1}** · detected after processing — more "
                        "appear once speakers are detected", visible=k < 1))
                    cast_voices.append(gr.Dropdown(choices=VOICE_CHOICES, value=AUTO_VOICE,
                                                   show_label=False, visible=k < 1, scale=3))
                    cast_previews.append(gr.Button("▶ Preview", size="sm", visible=k < 1,
                                                   scale=0, min_width=110))
            preview_audio = gr.Audio(label="Voice preview", interactive=False, autoplay=True)
            revoice_btn = gr.Button("🔁 Re-voice with this cast", variant="primary")

        with gr.Tab("📝 Script"):
            gr.Markdown("Who says what, original and dubbed. Click a line to revise it.  \n"
                        "<small>＋ recovered from speech the recogniser first skipped · "
                        "⚠ transcript uncertain — worth a listen</small>")
            transcript_table = gr.Dataframe(
                headers=["#", "Time", "Speaker", "Voice", "Original", "Dubbed"],
                datatype=["number", "str", "str", "str", "str", "str"],
                interactive=False, wrap=True, column_widths=["5%", "8%", "10%", "9%", "34%", "34%"],
            )

        with gr.Tab("✏️ Revise"):
            gr.Markdown("Ask for any number of changes in one message — wording, tone, "
                        "voices, gender, who said a line, delivery, or leaving a line "
                        "undubbed. Each request is listed so nothing is silently skipped.  \n"
                        "*e.g. “make the intro more casual, speaker 2 should be an older man "
                        "with a deeper voice, line 5 sad, and don’t dub the song at 1:20”*")
            with gr.Row():
                edit_input = gr.Textbox(label="Change request", lines=3, scale=5,
                                        placeholder="What should change?")
                with gr.Column(scale=1, min_width=150):
                    parse_btn = gr.Button("Plan changes →")
                    apply_btn = gr.Button("Apply changes ✓", variant="primary")
            parse_status = gr.Markdown("")
            edit_queue_table = gr.Dataframe(
                headers=["Request", "Change", "Applies to", "Details"],
                datatype=["str", "str", "str", "str"], interactive=False, wrap=True,
                label="Planned changes",
            )
            diff_md = gr.Markdown("")

        with gr.Tab("⬇ Export"):
            gr.Markdown("Lip-synced video (H.264/AAC), the dubbed soundtrack alone, and "
                        "subtitles in the dubbed and original languages.")
            export_files = gr.File(label="Deliverables", file_count="multiple",
                                   interactive=False)

        # ── Developer ─────────────────────────────────────────────────────────
        with gr.Tab("🛠 Developer", visible=False) as dev_tab:
            dev_refresh = gr.Button("↻ Refresh", size="sm")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("##### Stage timings (last run)")
                    dev_timings = gr.Markdown("_No run yet in this session._")
                with gr.Column():
                    gr.Markdown("##### Speakers")
                    dev_speakers = gr.Markdown("_No project loaded._")
            with gr.Accordion("Cache inspector", open=True):
                dev_cache = gr.Markdown("_No project loaded._")
            with gr.Accordion("Line inspector", open=True):
                dev_seg = gr.Dropdown(label="Line", choices=[], interactive=True)
                dev_seg_md = gr.Markdown("_Pick a line._")
                with gr.Row():
                    dev_tts = gr.Audio(label="Raw TTS clip", interactive=False)
                    dev_fit = gr.Audio(label="Fitted to slot", interactive=False)
                    dev_clip = gr.Video(label="First lip-synced shot", interactive=False)
                dev_seg_json = gr.JSON(label="Segment record")
            with gr.Accordion("Pipeline log (last run)", open=False):
                dev_log = gr.Textbox(lines=14, max_lines=30, show_label=False,
                                     interactive=False, autoscroll=True)
            with gr.Accordion("Server output (live)", open=False):
                debug_live = gr.Checkbox(value=True, label="Auto-refresh every 2 s")
                debug_box = gr.Textbox(lines=18, max_lines=40, interactive=False,
                                       autoscroll=True, show_label=False,
                                       value=_read_debug_log)
                gr.Markdown(f"Full log on the server: `{_LOG_FILE}`")
            with gr.Accordion("Measured TTS speaking rates", open=False):
                dev_rates = gr.Markdown(_rates_md())
            with gr.Accordion("Configuration", open=False):
                dev_config = gr.Markdown(_config_md())

    # ── Event wiring ──────────────────────────────────────────────────────────

    cast_outputs = [c for k in range(MAX_CAST_ROWS)
                    for c in (cast_labels[k], cast_voices[k], cast_previews[k])]
    # Every long-running action writes this same set.
    run_outputs = [status_md, tracker_html, project_html, dev_log, timings_state,
                   final_state_box, original_video, dubbed_video, dubbed_audio,
                   transcript_table, segments_state, video_id_state, lang_state,
                   *cast_outputs]
    _N_FIXED = 6                                   # status … final_state_box

    def _queued():
        # Runs instantly, outside the queue: a click always gets an answer,
        # even while another job holds the GPU.
        return ("⏳ Queued — starts as soon as the current job finishes "
                "(position shown here).")

    def _drive(events, stages, verb, on_done):
        for info in events:
            tracker = _tracker_html(stages, info["timings"], info["current"],
                                    error=info["kind"] == "error", elapsed=info["elapsed"])
            if info["kind"] == "error":
                yield (f"**✗ {verb} failed — {info['message']}**  \n<small>Details in "
                       "Developer mode → Server output.</small>", tracker, gr.update(),
                       info["log"], info["timings"], info["final"],
                       *[gr.update()] * (len(run_outputs) - _N_FIXED))
                return
            if info["kind"] == "progress":
                label = dict(stages).get(info["current"], "Finishing")
                yield (f"⏳ {verb} — **{label}**…", tracker, gr.update(), info["log"],
                       info["timings"], gr.update(),
                       *[gr.update()] * (len(run_outputs) - _N_FIXED))
            else:
                yield on_done(info, tracker)

    def _finished(info, tracker, msg, original=None, url=None):
        st = info["final"]
        segs = st.get("segments", [])
        lang = st.get("target_lang")
        seg_objs = [Segment.from_dict(s) for s in segs]
        return (f"✅ {msg} in {info['elapsed']:.0f}s — see **Preview**, **Script** and "
                "**Export**.", tracker, _project_html(st.get("video_id"), lang, segs, url),
                info["log"], info["timings"], st,
                original if original is not None else gr.update(),
                st.get("final_video"), st.get("final_audio"), _segments_to_df(seg_objs),
                segs, st.get("video_id"), lang, *_cast_updates(segs))

    def _on_process(cur_vid, url, c_start, c_end, lang, dom, nspk, vname, vstyle, mix, engine,
                    mode, songs, *voices):
        if not url.strip():
            yield ("**⚠ Paste a YouTube link first.**", _idle_tracker(),
                   *[gr.update()] * (len(run_outputs) - 2))
            return
        code = SUPPORTED_LANGS.get(lang, "hi")
        try:
            a, b = _parse_time(c_start), _parse_time(c_end)
        except ValueError as e:
            yield (f"**⚠ {e}**", _idle_tracker(), *[gr.update()] * (len(run_outputs) - 2))
            return
        if a is not None and b is not None and b <= a:
            yield ("**⚠ End must be after Start.**", _idle_tracker(),
                   *[gr.update()] * (len(run_outputs) - 2))
            return
        vid = clip_id(video_id_from_url(url.strip()), a, b)
        # Cast picks belong to the video they were made for. Carried over to
        # a new video they overrode automatic casting — a woman got the
        # previous video's male voice although she was detected as female.
        vmap = _voice_map(*voices) if vid == cur_vid else {}
        state = _base_state(vid, code, dom, vname,
                            BODHAN_STYLES.get(vstyle, "educational lecture"), vmap,
                            mix, engine, mode, songs)
        state["url"] = url.strip()
        state["clip_start"], state["clip_end"] = a, b
        state["num_speakers"] = int(nspk) if str(nspk).isdigit() else None
        print(f"\n=== Dub {url.strip()} → {code} (speakers: {nspk})", flush=True)
        yield ("⏳ Starting…", _tracker_html(FULL_STAGES, {}, "download", elapsed=0),
               _project_html(vid, code, [], url.strip()),
               *[gr.update()] * (len(run_outputs) - 3))
        yield from _drive(_stream(FULL_PIPELINE, state, FULL_STAGES), FULL_STAGES, "Dubbing",
                          lambda info, tr: _finished(info, tr, "Dubbed",
                                                     original=info["final"].get("video_path"),
                                                     url=url.strip()))

    def _after_run(vid, lang, segs, final_st, timings):
        """Export list + developer panels, refreshed after any run."""
        return (_exports(vid, lang, segs, final_st), _timings_md(timings, FULL_STAGES),
                _speakers_md(segs), _cache_md(vid),
                gr.update(choices=_segment_choices(segs), value=None), _rates_md())

    after_outputs = [export_files, dev_timings, dev_speakers, dev_cache, dev_seg, dev_rates]
    after_inputs = [video_id_state, lang_state, segments_state, final_state_box, timings_state]

    process_btn.click(
        _queued, outputs=[status_md], queue=False, show_progress="hidden",
    ).then(
        _on_process,
        inputs=[video_id_state, url_input, clip_start_in, clip_end_in, target_lang, domain,
                num_speakers_dropdown,
                voice_name_input, voice_style_dropdown, lang_mix, tts_engine, pipeline_mode,
                songs_mode,
                *cast_voices],
        outputs=run_outputs, show_progress="minimal", show_progress_on=[status_md],
    ).then(_after_run, inputs=after_inputs, outputs=after_outputs, show_progress="hidden")

    def _on_preview(voice, lang):
        if not voice or voice == AUTO_VOICE:
            raise gr.Error("Pick a voice first — 'Auto' has no single voice to preview.")
        code = SUPPORTED_LANGS.get(lang, "hi")
        text = _PREVIEW_TEXT.get(code, _PREVIEW_TEXT["en"])
        path = CACHE_ROOT / "_voice_previews" / f"{voice}_{code}.wav"
        try:
            return preview_voice(voice, code, text, path)
        except Exception as e:
            raise gr.Error(f"Preview failed: {e}")

    for k in range(MAX_CAST_ROWS):
        cast_previews[k].click(_on_preview, inputs=[cast_voices[k], target_lang],
                               outputs=[preview_audio], show_progress="minimal")

    def _on_revoice(segs, vid_id, lang, dom, vname, vstyle, mix, engine, *voices):
        if not segs:
            yield ("**⚠ Dub a video first.**", gr.update(),
                   *[gr.update()] * (len(run_outputs) - 2))
            return
        state = _cached_state(vid_id, SUPPORTED_LANGS.get(lang, "hi"), dom, vname,
                              BODHAN_STYLES.get(vstyle, "educational lecture"),
                              _voice_map(*voices), segs, [], mix, engine)
        if isinstance(state, str):
            yield (f"**{state}**", gr.update(), *[gr.update()] * (len(run_outputs) - 2))
            return
        print(f"\n=== Re-voice {vid_id}: {_voice_map(*voices) or 'auto'}", flush=True)
        yield from _drive(_stream(REVOICE_PIPELINE, state, REVOICE_STAGES), REVOICE_STAGES,
                          "Re-voicing", lambda info, tr: _finished(info, tr, "Re-voiced"))

    revoice_btn.click(
        _queued, outputs=[status_md], queue=False, show_progress="hidden",
    ).then(
        _on_revoice,
        inputs=[segments_state, video_id_state, target_lang, domain,
                voice_name_input, voice_style_dropdown, lang_mix, tts_engine, *cast_voices],
        outputs=run_outputs, show_progress="minimal", show_progress_on=[status_md],
    ).then(_after_run, inputs=after_inputs, outputs=after_outputs, show_progress="hidden")

    def _on_nspk(nspk, segs):
        return _cast_updates(segs) if segs else _placeholder_rows(nspk)

    num_speakers_dropdown.change(_on_nspk, inputs=[num_speakers_dropdown, segments_state],
                                 outputs=cast_outputs, show_progress="hidden")

    parse_btn.click(lambda: "⏳ Reading your request…", outputs=[parse_status],
                    queue=False, show_progress="hidden").then(
        plan_changes, inputs=[edit_input, segments_state, target_lang],
        outputs=[edit_ops_state, edit_queue_table, parse_status],
        show_progress="minimal", show_progress_on=[parse_status])

    def _on_apply(plan, segs, vid_id, lang, dom, vname, vstyle, mix, engine, *voices):
        none = [gr.update()] * (len(run_outputs) - 2)
        if not plan or not any(r["actions"] for r in plan.get("requests", [])):
            yield ("**⚠ Nothing to apply — use Plan changes first.**", gr.update(), *none)
            return
        if not segs:
            yield ("**⚠ Dub a video first.**", gr.update(), *none)
            return
        new_segs, vmap, notes, needs_tr = apply_plan(segs, plan, _voice_map(*voices))
        state = _cached_state(vid_id, SUPPORTED_LANGS.get(lang, "hi"), dom, vname,
                              BODHAN_STYLES.get(vstyle, "educational lecture"),
                              vmap, new_segs, [], mix, engine)
        if isinstance(state, str):
            yield (f"**{state}**", gr.update(), *none)
            return
        graph, stages = ((EDIT_PIPELINE, EDIT_STAGES) if needs_tr
                         else (REVOICE_PIPELINE, REVOICE_STAGES))
        print(f"\n=== Apply plan {vid_id}: {notes} → {'translate' if needs_tr else 'revoice'}",
              flush=True)
        yield from _drive(_stream(graph, state, stages), stages, "Applying changes",
                          lambda info, tr: _finished(info, tr, f"Applied {len(notes)} change(s)"))

    apply_btn.click(
        lambda segs: (_queued(), segs, ""), inputs=[segments_state],
        outputs=[status_md, before_edit_state, diff_md], queue=False, show_progress="hidden",
    ).then(
        _on_apply,
        inputs=[edit_ops_state, segments_state, video_id_state,
                target_lang, domain, voice_name_input, voice_style_dropdown, lang_mix,
                tts_engine, *cast_voices],
        outputs=run_outputs, show_progress="minimal", show_progress_on=[status_md],
    ).then(_after_run, inputs=after_inputs, outputs=after_outputs, show_progress="hidden"
    ).then(lambda b, a: ("#### What changed\n" + _diff_md(b, a)) if b and a else "",
           inputs=[before_edit_state, segments_state], outputs=[diff_md],
           show_progress="hidden")

    # Script row click → pre-fill a change request for that line
    def _on_row_select(evt: gr.SelectData, segs):
        if not segs or evt.index is None:
            return gr.update()
        ordered = sorted(segs, key=lambda d: d["start"])
        if evt.index[0] >= len(ordered):
            return gr.update()
        seg = Segment.from_dict(ordered[evt.index[0]])
        return f'Line {seg.id} ("{seg.text[:60]}"): '

    transcript_table.select(_on_row_select, inputs=[segments_state], outputs=[edit_input],
                            show_progress="hidden")

    # Developer mode
    dev_mode.change(lambda on: gr.update(visible=on), inputs=[dev_mode], outputs=[dev_tab],
                    show_progress="hidden")
    dev_refresh.click(lambda vid, segs, timings: (
        _cache_md(vid), _speakers_md(segs), _timings_md(timings, FULL_STAGES), _rates_md(),
        _config_md(), _read_debug_log()),
        inputs=[video_id_state, segments_state, timings_state],
        outputs=[dev_cache, dev_speakers, dev_timings, dev_rates, dev_config, debug_box],
        show_progress="hidden")
    dev_seg.change(_inspect_segment, inputs=[dev_seg, segments_state],
                   outputs=[dev_seg_md, dev_tts, dev_fit, dev_clip, dev_seg_json],
                   show_progress="hidden")
    debug_timer = gr.Timer(2.0)
    debug_timer.tick(lambda on, live: _read_debug_log() if on and live else gr.update(),
                     inputs=[dev_mode, debug_live], outputs=[debug_box],
                     show_progress="hidden")

    library_pick.change(_library_pick, inputs=[library_pick],
                        outputs=[library_dubbed, library_original], show_progress="hidden")
    library_refresh.click(lambda: gr.update(choices=_library()), outputs=[library_pick],
                          queue=False, show_progress="hidden")
    # Fresh list on every page load, so new dubs show without a restart.
    demo.load(lambda: gr.update(choices=_library()), outputs=[library_pick])


if __name__ == "__main__":
    demo.launch(share=True, show_error=True, theme=THEME, css=CSS)
