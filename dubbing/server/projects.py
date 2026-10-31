"""
The library: every dubbed video in the cache, read straight from its folder.

There is no database. A project is a folder cache/<project id>/ holding:
  meta.json              title, channel, duration (written by the download stage)
  video.mp4              the original video
  dubbed_<lang>.mp4      one finished dub per language
  project_<lang>.json    the final line-by-line state of that dub (written by
                         this server after every run), which is what makes a
                         past dub editable without re-running anything
"""

import json
import os
import re
import subprocess
import time
from pathlib import Path

from pipeline.cache import CACHE_ROOT
from pipeline.state import Segment
from pipeline.text.languages import LANG_NAMES, script_ok
from pipeline.voice.casting import assign_voices
from pipeline.voice.catalogue import voice_label

from .avatars import speaker_faces

# A real project id is a YouTube id, optionally with a clip range ("_30-90").
# Other folders are experiment copies and are left out of the library.
_PROJECT_ID = re.compile(r"^[A-Za-z0-9_-]{11}(_[0-9.]+-([0-9.]+|end))?$")

# Settings saved with a dub and reused when it is edited or re-voiced.
SETTING_KEYS = (
    "url",
    "mode",
    "domain",
    "voice_name",
    "voice_style",
    "lang_mix",
    "tts_backend",
    "translator",
    "songs",
    "voice_map",
    "lipsync",
)


def project_dir(pid: str) -> Path:
    """The folder of a project, refusing anything outside the cache."""
    if not _PROJECT_ID.match(pid or ""):
        raise ValueError("bad project id")
    d = (CACHE_ROOT / pid).resolve()
    if d.parent != CACHE_ROOT.resolve() or not d.is_dir():
        raise FileNotFoundError(pid)
    return d


def _meta(d: Path) -> dict:
    try:
        return json.loads((d / "meta.json").read_text(encoding="utf-8"))
    except Exception:
        return {}


def _dubs(d: Path) -> dict[str, Path]:
    """{language code: dubbed video} for one project. Copies with the
    subtitles burned in (dubbed_<lang>_subtitled.mp4) are exports, not dubs."""
    return {
        p.stem.split("_", 1)[1]: p
        for p in d.glob("dubbed_*.mp4")
        if not p.stem.endswith("_subtitled")
    }


def list_projects() -> list[dict]:
    """Every project with at least one finished dub, newest first."""
    rows = []
    for d in CACHE_ROOT.iterdir() if CACHE_ROOT.exists() else []:
        if not d.is_dir() or not _PROJECT_ID.match(d.name):
            continue
        dubs = _dubs(d)
        if not dubs:
            continue
        meta = _meta(d)
        langs = sorted(
            (
                {
                    "code": code,
                    "name": LANG_NAMES.get(code, code),
                    "updated": p.stat().st_mtime,
                    "editable": load_project(d.name, code) is not None,
                }
                for code, p in dubs.items()
            ),
            key=lambda x: -x["updated"],
        )
        rows.append(
            {
                "id": d.name,
                "title": meta.get("title") or d.name,
                "channel": meta.get("channel") or "",
                "duration": meta.get("duration"),
                "url": meta.get("url") or "",
                "langs": langs,
                "updated": langs[0]["updated"],
            }
        )
    return sorted(rows, key=lambda r: -r["updated"])


def search_lines(query: str, limit: int = 40) -> list[dict]:
    """
    Lines that contain `query`, in what was said or in the dub, across every
    dub: [{id, lang, line, start, text, dub}]. Reads the saved line files
    only (no audio recovery), so it stays fast.
    """
    q = query.strip().lower()
    if len(q) < 2:
        return []
    hits = []
    for d in CACHE_ROOT.iterdir() if CACHE_ROOT.exists() else []:
        if not d.is_dir() or not _PROJECT_ID.match(d.name):
            continue
        for lang in _dubs(d):
            saved = d / f"project_{lang}.json"
            try:
                if saved.exists():
                    segs = json.loads(saved.read_text(encoding="utf-8"))["segments"]
                elif (d / "translations.json").exists():
                    segs = json.loads((d / "translations.json").read_text(encoding="utf-8"))
                    sample = " ".join(s.get("dubbed_text") or "" for s in segs[:20]).strip()
                    if not sample or not script_ok(sample, lang, latin=True):
                        continue
                else:
                    continue
            except (OSError, ValueError, KeyError):
                continue
            for s in segs:
                text, dub = s.get("text") or "", s.get("dubbed_text") or ""
                if q in text.lower() or q in dub.lower():
                    hits.append(
                        {
                            "id": d.name,
                            "lang": lang,
                            "line": s.get("id"),
                            "start": s.get("start", 0),
                            "text": text,
                            "dub": dub,
                        }
                    )
                    if len(hits) >= limit:
                        return hits
    return hits


# ── One dub: load and save its line-by-line state ────────────────────────────


def save_project(pid: str, lang: str, state: dict) -> None:
    """Called after every finished run, so the dub can be opened and edited later."""
    d = CACHE_ROOT / pid
    data = {
        "segments": state.get("segments") or [],
        "settings": {k: state.get(k) for k in SETTING_KEYS},
        "saved": time.time(),
    }
    (d / f"project_{lang}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def load_project(pid: str, lang: str) -> dict | None:
    """
    {"segments", "settings"} for one dub, or None if it cannot be edited.

    Dubs made before this server existed have no project file. Their
    translations.json (written by the translate stage) still works, but it
    only holds the language dubbed last, so it is used only when its
    translated text is in this language's script.
    """
    d = CACHE_ROOT / pid
    saved = d / f"project_{lang}.json"
    if saved.exists():
        return json.loads(saved.read_text(encoding="utf-8"))
    legacy = d / "translations.json"
    if not legacy.exists():
        return None
    segs = json.loads(legacy.read_text(encoding="utf-8"))
    sample = " ".join(s.get("dubbed_text") or "" for s in segs[:20]).strip()
    if not sample or not script_ok(sample, lang, latin=True):
        return None
    _recover_audio(d, segs, lang)
    return {"segments": segs, "settings": {"url": _meta(d).get("url")}}


def _recover_audio(d: Path, segs: list[dict], lang: str) -> None:
    """
    translations.json is written before any line is voiced, so an old dub's
    lines carry no audio paths, and its first edit would redo every line.
    Recover each line's fitted clip from disk: the newest one made in the
    run that produced dubbed_<lang>.mp4 (within the hour before it). All or
    nothing — if any line has no such clip, nothing is filled in.
    """
    video = d / f"dubbed_{lang}.mp4"
    seg_dir = d / "segments"
    if not video.exists() or not seg_dir.is_dir():
        return
    done = video.stat().st_mtime
    found = {}
    for s in segs:
        if s.get("keep_original") or not s.get("dubbed_text"):
            continue
        clips = [
            f
            for f in seg_dir.glob(f"s{s['id']:04d}_dubbed_*_fit*.wav")
            if not f.name.endswith("_va.wav") and done - 3600 <= f.stat().st_mtime <= done + 5
        ]
        if not clips:
            return
        fit = max(clips, key=lambda f: f.stat().st_mtime)
        tts = seg_dir / (
            fit.name.split("_dubbed_")[0]
            + "_dubbed_"
            + fit.name.split("_dubbed_")[1].split("_")[0]
            + ".wav"
        )
        if not tts.exists():
            return
        found[s["id"]] = (str(tts), str(fit))
    root = CACHE_ROOT.parent  # paths are stored relative to the dubbing folder, like the pipeline's
    for s in segs:
        if s["id"] in found:
            tts, fit = found[s["id"]]
            s["dubbed_audio"] = os.path.relpath(tts, root)
            s["aligned_audio"] = os.path.relpath(fit, root)


# ── What the web page shows ──────────────────────────────────────────────────


def project_view(pid: str, lang: str | None) -> dict:
    """Everything the project page needs, in one response."""
    d = project_dir(pid)
    dubs = _dubs(d)
    if not dubs:
        raise FileNotFoundError(pid)
    if lang not in dubs:
        lang = max(dubs, key=lambda c: dubs[c].stat().st_mtime)
    meta = _meta(d)
    proj = load_project(pid, lang)
    segs = sorted(proj["segments"], key=lambda s: s["start"]) if proj else []
    _fill_voices(segs, lang, (proj or {}).get("settings", {}).get("voice_map") or {})
    stamp = int(dubs[lang].stat().st_mtime)  # changes when the dub is re-rendered
    return {
        "id": pid,
        "title": meta.get("title") or pid,
        "channel": meta.get("channel") or "",
        "duration": meta.get("duration"),
        "url": (proj or {}).get("settings", {}).get("url") or meta.get("url") or "",
        "lang": lang,
        "langs": [
            {"code": c, "name": LANG_NAMES.get(c, c)}
            for c in sorted(dubs, key=lambda c: -dubs[c].stat().st_mtime)
        ],
        "editable": proj is not None,
        # Fast mode is remembered per dub; lines made in it can be synced later.
        "lipsync": ((proj or {}).get("settings") or {}).get("lipsync") is not False,
        "unsynced": sum(1 for d in segs if d.get("synced") is False and not d.get("keep_original")),
        "media": {
            "original": f"/media/{pid}/video.mp4",
            "dubbed": f"/media/{pid}/dubbed_{lang}.mp4?v={stamp}",
            "audio": f"/media/{pid}/dubbed_{lang}.m4a?v={stamp}"
            if (d / f"dubbed_{lang}.m4a").exists()
            else None,
        },
        "lines": [_line(s) for s in segs],
        "speakers": _speakers(segs, pid, speaker_faces(d, segs)),
    }


def _fill_voices(segs: list[dict], lang: str, voice_map: dict) -> None:
    """
    Older saved dubs do not record each line's voice. Casting is
    deterministic, so running it again gives the voices that were used.
    """
    if not segs or all(s.get("voice") for s in segs):
        return
    objs = [Segment.from_dict(s) for s in segs]
    try:
        cast = assign_voices(objs, lang, None, voice_map)
    except Exception:
        return
    for s in segs:
        s["voice"] = s.get("voice") or cast.get(s.get("speaker") or 0)


def _line(d: dict) -> dict:
    s = Segment.from_dict(d)
    return {
        "id": s.id,
        "start": round(s.start, 2),
        "end": round(s.end, 2),
        "speaker": s.speaker or 0,
        "text": s.text,
        "dubbed": s.dubbed_text or "",
        "voice": voice_label(s.voice) if s.voice else "",
        "kept": bool(s.keep_original),
        "sung": bool(s.sung),
        "flag": s.asr_flag or "",
        "delivery": s.delivery or "",
        "lang": s.lang_override or "",
    }


def _speakers(segs: list[dict], pid: str, faces: dict[int, str]) -> list[dict]:
    out: dict[int, dict] = {}
    for d in segs:
        s = Segment.from_dict(d)
        row = out.setdefault(
            s.speaker or 0,
            {
                "id": s.speaker or 0,
                "gender": s.gender or "",
                "voice": s.voice or "",
                "lines": 0,
                "seconds": 0.0,
            },
        )
        row["lines"] += 1
        row["seconds"] = round(row["seconds"] + s.duration, 1)
        row["voice"] = row["voice"] or s.voice or ""
    for k, row in out.items():
        row["face"] = f"/media/{pid}/{faces[k]}" if k in faces else None
    return [out[k] for k in sorted(out)]


# ── Files for the browser ─────────────────────────────────────────────────────

_SERVABLE = re.compile(
    r"^(video\.mp4|dubbed_[a-z]{2,3}(_subtitled)?\.(mp4|m4a)|subtitles_[a-z_]+\.srt|thumb\.jpg|strip\.jpg"
    r"|face_s\d+_[0-9a-f]{8}\.jpg|versions/[a-z]{2,3}/v\d+\.mp4|audition/[0-9a-f]{16}\.wav|myvoice_s\d+_[0-9a-f]{8}\.wav)$"
)


def media_file(pid: str, name: str) -> Path:
    """
    A file the browser may load. Two are prepared on first request:
      thumb.jpg   a frame from the original, for the library card
    strip.jpg   STRIP_FRAMES frames across the video side by side, for the
                timeline's filmstrip and its hover preview
      video.mp4   if its index sits at the end of the file (clip downloads),
                  a copy with the index moved to the front, so playback starts
                  at once instead of after the whole file has downloaded
    """
    if not _SERVABLE.match(name):
        raise FileNotFoundError(name)
    d = project_dir(pid)
    if name == "thumb.jpg":
        return _thumbnail(d)
    if name == "strip.jpg":
        return _filmstrip(d)
    if name == "video.mp4":
        return _streamable(d / "video.mp4")
    path = d / name
    if not path.exists():
        raise FileNotFoundError(name)
    return path


STRIP_FRAMES = 60


def _filmstrip(d: Path) -> Path:
    """STRIP_FRAMES evenly spaced frames of the original, 90 px high, in one row."""
    out = d / "strip.jpg"
    src = d / "video.mp4"
    if out.exists() or not src.exists():
        if not out.exists():
            raise FileNotFoundError("strip.jpg")
        return out
    dur = float(
        subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "csv=p=0",
                str(src),
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()
        or 0
    )
    if dur <= 0:
        raise FileNotFoundError("strip.jpg")
    tmp = d / "strip.tmp.jpg"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(src),
            "-vf",
            # A touch more than STRIP_FRAMES per video, so the last tile is
            # never left empty; the tile filter keeps the first STRIP_FRAMES.
            f"fps={(STRIP_FRAMES + 1) / dur:.5f},scale=-2:90,tile={STRIP_FRAMES}x1",
            "-frames:v",
            "1",
            "-q:v",
            "5",
            str(tmp),
            "-y",
        ],
        capture_output=True,
    )
    if not tmp.exists():
        raise FileNotFoundError("strip.jpg")
    tmp.replace(out)
    return out


def _thumbnail(d: Path) -> Path:
    out = d / "thumb.jpg"
    if not out.exists():
        src = next(iter(_dubs(d).values()), d / "video.mp4")
        subprocess.run(
            [
                "ffmpeg",
                "-ss",
                "3",
                "-i",
                str(src),
                "-frames:v",
                "1",
                "-vf",
                "scale=480:-2",
                "-q:v",
                "4",
                str(out),
                "-y",
            ],
            capture_output=True,
        )
    if not out.exists():
        raise FileNotFoundError("thumb.jpg")
    return out


def _streamable(src: Path) -> Path:
    if not src.exists():
        raise FileNotFoundError(src.name)
    with open(src, "rb") as f:
        head = f.read(4_000_000)
    moov, mdat = head.find(b"moov"), head.find(b"mdat")
    if moov != -1 and (mdat == -1 or moov < mdat):
        return src
    out = src.with_name("video_web.mp4")
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(
            ["ffmpeg", "-i", str(src), "-c", "copy", "-movflags", "+faststart", str(out), "-y"],
            capture_output=True,
            check=True,
        )
    return out
