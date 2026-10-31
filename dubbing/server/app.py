"""
The Dubbing Studio web server (FastAPI). It serves the web page in web/ and
a small JSON API over the dubbing pipeline:

  GET  /api/options                       languages, voices, styles for the forms
  GET  /api/projects                      the library
  GET  /api/search?q=                     lines across every dub that contain q
  GET  /api/projects/{id}?lang=           one dub: video links, lines, speakers
  POST /api/dub                           start a new dub
  POST /api/projects/{id}/plan            read a change request, return a plan
  POST /api/projects/{id}/apply           apply a plan (re-runs only what changed)
  POST /api/projects/{id}/lines/{line}    replace one line's dubbed wording
  POST /api/projects/{id}/revoice         new voices for some speakers
  GET  /api/projects/{id}/exports?lang=   downloadable files
  GET  /api/voices/preview?voice=&lang=   a short sample of a voice
  GET  /api/estimate?seconds=             expected time for a new dub of that length
  GET  /api/projects/{id}/versions?lang=  version history of a dub
  POST /api/projects/{id}/versions/{v}/restore   make an older version current
  GET  /api/jobs                          jobs queued or running
  GET  /api/jobs/{id}/lines               a running job's lines so far (with audio)
  GET  /api/jobs/{id}/events              live progress of one job (SSE)
  GET  /media/{id}/{file}                 video/audio/subtitles, with seeking

Set STUDIO_PASSWORD in .env to ask for a password (user name "studio").
"""

import asyncio
import base64
import json
import os
import re
import secrets
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.cache import CACHE_ROOT, clip_id, video_id_from_url
from pipeline.edits.edit_planner import apply_plan, redetect_request, validate_plan
from pipeline.graph import EDIT_PIPELINE, FULL_PIPELINE, LIPSYNC_PIPELINE, REVOICE_PIPELINE
from pipeline.state import Segment
from pipeline.text.languages import LANG_NAMES
from pipeline.video.latentsync_runner import free_gpu_gb
from pipeline.voice.catalogue import voice_label
from pipeline.voice.styles import BODHAN_STYLES, DEFAULT_STYLE_LABEL
from pipeline.voice.tts import preview_voice

from . import audition as auditions
from . import (
    comments,
    eta,
    startup,  # noqa: F401, I001 — loads .env before the pipeline
    uploads,
    versions,
)
from .exports import collect_exports
from .jobs import JOBS
from .options import (
    CATALOGUE,
    DOMAINS,
    EDIT_STAGES,
    FULL_STAGES,
    HAS_SARVAM,
    INTERNATIONAL_LANGS,
    LANG_LABEL,
    LIPSYNC_STAGES,
    PREVIEW_TEXT,
    REVOICE_STAGES,
    STYLES,
    SUPPORTED_LANGS,
    VOICE_CHOICES,
    fast_stages,
)
from .projects import (
    list_projects,
    load_project,
    media_file,
    project_dir,
    project_view,
    search_lines,
)
from .revise import plan_changes
from .runner import base_state, cached_state, parse_time

# ffmpeg is installed next to Python in the venv; make sure it is found even
# when the server is started without activating the venv.
os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
DEFAULT_DOMAIN = "education / lecture"
DEFAULT_STYLE = "educational lecture"

app = FastAPI(title="Dubbing Studio", docs_url="/api/docs")


# ── Optional password ─────────────────────────────────────────────────────────
@app.middleware("http")
async def password(request: Request, call_next):
    expected = os.environ.get("STUDIO_PASSWORD")
    if expected:
        header = request.headers.get("authorization", "")
        ok = False
        if header.startswith("Basic "):
            try:
                _, _, given = base64.b64decode(header[6:]).decode().partition(":")
                ok = secrets.compare_digest(given, expected)
            except Exception:
                ok = False
        if not ok:
            return JSONResponse(
                {"detail": "Password required"},
                status_code=401,
                headers={"WWW-Authenticate": 'Basic realm="Dubbing Studio"'},
            )
    return await call_next(request)


def _not_found(exc: Exception) -> HTTPException:
    return HTTPException(404, f"Not found: {exc}")


# ── Read-only ─────────────────────────────────────────────────────────────────
@app.get("/api/options")
def options():
    return {
        "languages": [{"code": code, "label": label} for label, code in SUPPORTED_LANGS.items()],
        "international": [
            {"code": code, "label": label} for label, code in INTERNATIONAL_LANGS.items()
        ],
        "voices": [
            {"value": value, "label": label, "gender": gender}
            for label, value in VOICE_CHOICES[1:]
            for v, gender, _ in CATALOGUE
            if v == value
        ],
        "styles": list(BODHAN_STYLES),
        "delivery_styles": STYLES,  # values for "Delivery" edits
        "default_style": DEFAULT_STYLE_LABEL,
        "domains": DOMAINS,
        "sarvam": HAS_SARVAM,
    }


@app.get("/api/search")
def search(q: str):
    """Lines across every dub that contain q, in the original or the dub."""
    return search_lines(q)


@app.get("/api/projects")
def projects():
    return list_projects()


@app.get("/api/projects/{pid}")
def project(pid: str, lang: str | None = None):
    try:
        view = project_view(pid, lang)
    except (ValueError, FileNotFoundError) as exc:
        raise _not_found(exc)
    view["busy"] = JOBS.busy_with(pid)
    return view


class LangRequest(BaseModel):
    lang: str


@app.get("/api/projects/{pid}/exports")
def exports(pid: str, lang: str):
    proj = _editable(pid, lang)
    files = collect_exports(pid, lang, proj["segments"], None)
    return [
        {
            "name": Path(f).name,
            "url": f"/media/{pid}/{Path(f).name}",
            "size": Path(f).stat().st_size,
        }
        for f in files
    ]


@app.post("/api/projects/{pid}/subtitled")
def subtitled(pid: str, req: LangRequest):
    """The dubbed video with the dub's subtitles burned in (made on request,
    kept until the dub changes)."""
    proj = _editable(pid, req.lang)
    collect_exports(pid, req.lang, proj["segments"], None)  # writes the .srt
    d = CACHE_ROOT / pid
    src, srt = d / f"dubbed_{req.lang}.mp4", d / f"subtitles_{req.lang}.srt"
    out = d / f"dubbed_{req.lang}_subtitled.mp4"
    if not src.exists():
        raise HTTPException(404, "This dub has no video yet.")
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        style = "FontName=Noto Sans,FontSize=20,Outline=2,Shadow=0,MarginV=28,BorderStyle=1"
        r = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(src),
                "-vf",
                f"subtitles={srt.resolve()}:force_style='{style}'",
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-c:a",
                "copy",
                "-movflags",
                "+faststart",
                str(out),
                "-y",
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0 or not out.exists():
            raise HTTPException(500, f"Could not burn in the subtitles: {r.stderr[-200:]}")
    return {"url": f"/media/{pid}/{out.name}", "size": out.stat().st_size}


@app.get("/api/estimate")
def estimate(seconds: float, lipsync: bool = True):
    """How long a new dub of this many seconds of video should take right now."""
    free = free_gpu_gb()
    keys = [k for k, _ in FULL_STAGES]
    modes = eta.modes(free)
    if not lipsync:
        modes["lipsync"] = "off"
    per_stage = eta.estimate(keys, seconds, seconds, modes)
    notes = eta.notes(free, set(keys) if lipsync else set(keys) - {"lipsync"})
    return {"seconds": round(sum(per_stage.values())), "notes": notes}


@app.get("/api/voices/preview")
def voice_preview(voice: str, lang: str):
    if voice not in {v for v, _, _ in CATALOGUE}:
        raise HTTPException(400, "Unknown voice")
    text = PREVIEW_TEXT.get(lang, PREVIEW_TEXT["en"])
    path = CACHE_ROOT / "_voice_previews" / f"{voice}_{lang}.wav"
    try:
        return FileResponse(preview_voice(voice, lang, text, path), media_type="audio/wav")
    except Exception as exc:
        raise HTTPException(502, f"Preview failed: {exc}")


@app.get("/media/{pid}/{name:path}")
def media(pid: str, name: str):
    try:
        path = media_file(pid, name)
    except (ValueError, FileNotFoundError, OSError) as exc:
        raise _not_found(exc)
    # FileResponse answers range requests, so the player can start and seek
    # without downloading the whole file.
    return FileResponse(path, headers={"Cache-Control": "private, max-age=60"})


@app.get("/api/projects/{pid}/versions")
def project_versions(pid: str, lang: str):
    try:
        project_dir(pid)
    except (ValueError, FileNotFoundError) as exc:
        raise _not_found(exc)
    return versions.list_versions(pid, lang)


class RestoreRequest(BaseModel):
    lang: str


@app.post("/api/projects/{pid}/versions/{v}/restore")
def restore_version(pid: str, v: int, req: RestoreRequest):
    _editable(pid, req.lang)
    _not_busy(pid)
    try:
        new_v = versions.restore(pid, req.lang, v)
    except FileNotFoundError as exc:
        raise _not_found(exc)
    return {"version": new_v}


# ── Jobs ─────────────────────────────────────────────────────────────────────
@app.get("/api/jobs")
def jobs():
    return [j.public(JOBS.position(j)) for j in JOBS.active()]


@app.get("/api/jobs/{job_id}")
def job(job_id: int):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")
    return j.public(JOBS.position(j))


@app.post("/api/jobs/{job_id}/stop")
def stop_job(job_id: int):
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")
    JOBS.cancel(j)
    return j.public(JOBS.position(j))


@app.get("/api/jobs/{job_id}/lines")
def job_lines(job_id: int):
    """
    The lines of a running job as far as it has got: the transcript, then the
    translation, then each line's new voice as soon as it exists.
    """
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")
    segs = sorted((j.latest or {}).get("segments") or [], key=lambda s: s["start"])
    return [
        {
            "id": s["id"],
            "start": round(s["start"], 2),
            "speaker": s.get("speaker") or 0,
            "text": s.get("text") or "",
            "dubbed": s.get("dubbed_text") or "",
            "kept": bool(s.get("keep_original")),
            "audio": f"/api/jobs/{job_id}/lines/{s['id']}/audio"
            if (s.get("aligned_audio") or s.get("dubbed_audio"))
            else None,
        }
        for s in segs
    ]


@app.get("/api/jobs/{job_id}/lines/{line}/audio")
def job_line_audio(job_id: int, line: int):
    j = JOBS.get(job_id)
    seg = (
        next((s for s in (j.latest or {}).get("segments") or [] if s["id"] == line), None)
        if j
        else None
    )
    path = seg and (seg.get("aligned_audio") or seg.get("dubbed_audio"))
    if not path or not Path(path).exists():
        raise HTTPException(404, "No audio yet")
    p = Path(path).resolve()
    if CACHE_ROOT.resolve() not in p.parents:
        raise HTTPException(404, "No audio")
    return FileResponse(p, media_type="audio/wav")


@app.get("/api/jobs/{job_id}/events")
async def job_events(job_id: int, request: Request):
    """Server-sent events: the job's state each time it changes, until it ends."""
    j = JOBS.get(job_id)
    if not j:
        raise HTTPException(404, "No such job")

    async def stream():
        seen, last_tick = -1, 0.0
        while not await request.is_disconnected():
            now = asyncio.get_event_loop().time()
            # Also send once a second while running, so the elapsed time moves.
            if j.version != seen or (j.status == "running" and now - last_tick >= 1):
                seen, last_tick = j.version, now
                yield f"data: {json.dumps(j.public(JOBS.position(j)), ensure_ascii=False)}\n\n"
                if j.status in ("done", "error", "stopped"):
                    return
            await asyncio.sleep(0.3)

    return StreamingResponse(stream(), media_type="text/event-stream")


# ── Starting work ─────────────────────────────────────────────────────────────
class DubRequest(BaseModel):
    url: str
    lang: str = "hi"
    start: str = ""
    end: str = ""
    speakers: int | None = None
    lang_mix: str = "standard"
    translator: str = "indic"
    songs: str = "keep"
    engine: str = "bodhan"
    domain: str = DEFAULT_DOMAIN
    style: str = DEFAULT_STYLE_LABEL
    voice_name: str = ""
    lipsync: bool = True  # False = fast mode: no lip sync (can be added later)
    duration: float | None = None  # the video's length, when the dialog already knows it
    mode: str = "indic"  # "international": LLM translation + Kokoro voices


@app.post("/api/upload")
async def upload(request: Request, name: str = "video"):
    """A video or audio file from the user's computer, sent as the raw body."""
    try:
        return await uploads.save_upload(request.stream(), name)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except Exception as exc:
        raise HTTPException(400, f"Could not read that file ({type(exc).__name__}).")


@app.get("/api/probe")
def probe(url: str):
    """Does this link work? Title, picture and length, before dubbing it."""
    return uploads.probe_link(url.strip())


@app.post("/api/dub")
def dub(req: DubRequest):
    url = req.url.strip()
    if not url:
        raise HTTPException(400, "Paste a video link or upload a file first.")
    if not (url.startswith(("http://", "https://")) or re.fullmatch(r"upload:u[0-9a-f]{10}", url)):
        raise HTTPException(400, "That doesn't look like a link — it should start with https://")
    audio_only = False
    if url.startswith("upload:"):
        meta = CACHE_ROOT / "_uploads" / f"{url[7:]}.json"
        if not meta.exists():
            raise HTTPException(400, "That upload is gone — upload the file again.")
        audio_only = bool(json.loads(meta.read_text(encoding="utf-8")).get("audio_only"))
    if req.lang not in LANG_NAMES:
        raise HTTPException(400, "Unknown language")
    try:
        a, b = parse_time(req.start), parse_time(req.end)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    if a is not None and b is not None and b <= a:
        raise HTTPException(400, "End must be after Start.")
    pid = clip_id(video_id_from_url(url), a, b)
    intl = req.mode == "international"
    if intl and req.lang not in INTERNATIONAL_LANGS.values():
        raise HTTPException(400, "International mode dubs into English or a world language.")
    state = base_state(
        pid,
        req.lang,
        req.domain,
        None if intl else req.voice_name,
        BODHAN_STYLES.get(req.style, DEFAULT_STYLE),
        {},
        req.lang_mix,
        "kokoro" if intl else req.engine,
        "llm" if intl else req.translator,
        req.songs,
    )
    state.update(
        url=url,
        clip_start=a,
        clip_end=b,
        num_speakers=req.speakers,
        lipsync=req.lipsync and not audio_only,  # an audio file has no lips
        # Known before the download, so the first time estimate is right.
        source_duration=req.duration if req.duration and req.duration > 0 else None,
        mode=req.mode if req.mode == "international" else "indic",
    )
    job = JOBS.submit(
        kind="dub",
        project=pid,
        lang=req.lang,
        title=url,
        stages=_stages(FULL_STAGES, state),
        graph=FULL_PIPELINE,
        state=state,
        request=req.model_dump(),
    )
    return job.public(JOBS.position(job))


class PlanRequest(BaseModel):
    lang: str
    message: str


@app.post("/api/projects/{pid}/plan")
def plan(pid: str, req: PlanRequest):
    proj = _editable(pid, req.lang)
    plan, rows, status = plan_changes(req.message, proj["segments"], LANG_LABEL.get(req.lang, ""))
    return {
        "plan": plan,
        "changes": [
            {"request": r[0], "change": r[1], "target": r[2], "detail": r[3]} for r in rows
        ],
        "status": status,
    }


class ApplyRequest(BaseModel):
    lang: str
    plan: dict
    lipsync: bool | None = None  # None = this dub's own setting


@app.post("/api/projects/{pid}/apply")
def apply(pid: str, req: ApplyRequest):
    if not any(r.get("actions") for r in req.plan.get("requests", [])):
        raise HTTPException(400, "Nothing to apply.")
    return _run_plan(pid, req.lang, req.plan, "Apply changes", req.lipsync)


@app.post("/api/projects/{pid}/impact")
def impact(pid: str, req: ApplyRequest):
    """What a set of changes would redo, before it runs: the lines, and about
    how long it takes. Shown under the list of waiting changes."""
    proj = _editable(pid, req.lang)
    before = proj["segments"]
    plan = validate_plan(req.plan, [Segment.from_dict(d) for d in before], CATALOGUE, STYLES)
    if redetect_request(plan):
        lines = [d["id"] for d in before]
        keys, work = [k for k, _ in FULL_STAGES], None
    elif _resync_ids(plan)[1]:
        ids = _resync_ids(plan)[0]
        lines, keys = ids, [k for k, _ in LIPSYNC_STAGES]
        work = _speech_seconds(d for d in before if d["id"] in set(ids))
        req.lipsync = True
    else:
        saved = (proj.get("settings") or {}).get("voice_map") or {}
        new_segs, _, _, needs_tr = apply_plan(before, plan, saved, req.lang)
        redo = [d for d in new_segs if not d.get("dubbed_audio") and not d.get("keep_original")]
        old = {d["id"]: d for d in before}
        changed = {
            d["id"]
            for d in new_segs
            if d.get("keep_original") != old.get(d["id"], {}).get("keep_original")
        }
        lines = sorted({d["id"] for d in redo} | changed)
        keys = [k for k, _ in (EDIT_STAGES if needs_tr else REVOICE_STAGES)]
        work = _speech_seconds(redo)
    lipsync = (
        req.lipsync
        if req.lipsync is not None
        else (proj.get("settings") or {}).get("lipsync") is not False
    )
    modes = eta.modes(free_gpu_gb())
    if not lipsync:
        modes["lipsync"] = "off"
    video_s = max((d["end"] for d in before), default=0)
    per = eta.estimate(keys, video_s, video_s if work is None else work, modes)
    return {"lines": lines, "seconds": round(sum(per.values()))}


class GlossaryRequest(BaseModel):
    lang: str
    entries: list[dict]  # [{"term", "say"}]


@app.get("/api/projects/{pid}/glossary")
def get_glossary(pid: str, lang: str):
    from pipeline.text import glossary

    proj = _editable(pid, lang)
    rows = glossary.load(pid)
    return {"entries": rows, "lines": glossary.lines_using(proj["segments"], rows)}


@app.put("/api/projects/{pid}/glossary")
def put_glossary(pid: str, req: GlossaryRequest):
    from pipeline.text import glossary

    proj = _editable(pid, req.lang)
    rows = glossary.save(pid, req.entries)
    return {"entries": rows, "lines": glossary.lines_using(proj["segments"], rows)}


class CommentRequest(BaseModel):
    lang: str
    name: str
    text: str
    t: float
    t2: float | None = None


class CommentUpdate(BaseModel):
    lang: str
    done: bool | None = None
    delete: bool = False


@app.get("/api/projects/{pid}/comments")
def list_comments(pid: str, lang: str):
    try:
        project_dir(pid)
    except (ValueError, FileNotFoundError) as exc:
        raise _not_found(exc)
    return comments.load(pid, lang)


@app.post("/api/projects/{pid}/comments")
def add_comment(pid: str, req: CommentRequest):
    try:
        project_dir(pid)
        return comments.add(pid, req.lang, req.name, req.text, req.t, req.t2)
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    except FileNotFoundError as exc:
        raise _not_found(exc)


@app.post("/api/projects/{pid}/comments/{cid}")
def update_comment(pid: str, cid: str, req: CommentUpdate):
    try:
        project_dir(pid)
    except (ValueError, FileNotFoundError) as exc:
        raise _not_found(exc)
    comments.update(pid, req.lang, cid, req.done, req.delete)
    return comments.load(pid, req.lang)


@app.post("/api/projects/{pid}/myvoice")
async def my_voice(pid: str, request: Request, speaker: int):
    """The user's own voice for one speaker (recorded in the browser, or a
    file): cleaned up and kept as that speaker's cloning reference."""
    import hashlib

    try:
        d = project_dir(pid)
    except (ValueError, FileNotFoundError) as exc:
        raise _not_found(exc)
    body = await request.body()
    if len(body) > 50 * 1024**2:
        raise HTTPException(400, "That recording is too large — 20–30 seconds is plenty.")
    name = f"myvoice_s{speaker}_{hashlib.sha1(body).hexdigest()[:8]}.wav"
    out, raw = d / name, d / (name + ".in")
    raw.write_bytes(body)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-i",
                str(raw),
                "-af",
                "silenceremove=start_periods=1:start_threshold=-45dB:stop_periods=-1:stop_threshold=-45dB:stop_duration=0.6,loudnorm=I=-20",
                "-ac",
                "1",
                "-ar",
                "44100",
                "-t",
                "30",
                str(out),
                "-y",
            ],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        raise HTTPException(400, "Could not read that recording — try again, or upload a WAV/MP3.")
    finally:
        raw.unlink(missing_ok=True)
    import soundfile as sf

    secs = sf.info(str(out)).duration
    if secs < 6:
        out.unlink(missing_ok=True)
        raise HTTPException(
            400, f"Only {secs:.0f} s of speech came through — please record at least 10 seconds."
        )
    return {"ref": name, "seconds": round(secs, 1), "url": f"/media/{pid}/{name}"}


class AuditionRequest(BaseModel):
    lang: str
    line: int
    action: dict  # one plan action, as the studio's buttons build it


@app.post("/api/projects/{pid}/audition")
def audition(pid: str, req: AuditionRequest):
    """A sample of one line after a change, to hear before applying it."""
    proj = _editable(pid, req.lang)
    plan = validate_plan(
        {"requests": [{"request": "audition", "actions": [req.action]}]},
        [Segment.from_dict(d) for d in proj["segments"]],
        CATALOGUE,
        STYLES,
    )
    acts = plan["requests"][0]["actions"]
    if not acts or acts[0]["type"] in ("redetect_speakers", "merge_speakers"):
        raise HTTPException(400, "This change can't be previewed on one line.")
    try:
        name = auditions.audition(pid, req.lang, proj, req.line, acts[0])
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    return {"audio": f"/media/{pid}/{name}"}


@app.post("/api/projects/{pid}/lipsync")
def add_lipsync(pid: str, req: LangRequest):
    """Lip-sync the lines made in fast mode; everything else is reused."""
    proj = _editable(pid, req.lang)
    _not_busy(pid)
    todo = [d for d in proj["segments"] if d.get("synced") is False and not d.get("keep_original")]
    if not todo:
        raise HTTPException(400, "Every line is already lip-synced.")
    state = _state(
        pid, req.lang, proj, proj["segments"], (proj.get("settings") or {}).get("voice_map") or {}
    )
    state.update(lipsync=True, new_audio=None)
    job = JOBS.submit(
        kind="lipsync",
        project=pid,
        lang=req.lang,
        title=f"Add lip sync ({len(todo)} line{'s' if len(todo) > 1 else ''})",
        stages=LIPSYNC_STAGES,
        graph=LIPSYNC_PIPELINE,
        state=state,
        work_s=_speech_seconds(todo),
    )
    return job.public(JOBS.position(job))


class LineRequest(BaseModel):
    lang: str
    text: str


@app.post("/api/projects/{pid}/lines/{line}")
def set_line(pid: str, line: int, req: LineRequest):
    if not req.text.strip():
        raise HTTPException(400, "The new wording is empty.")
    plan = {
        "requests": [
            {
                "request": f"New wording for line {line}",
                "actions": [{"type": "set_text", "line": line, "text": req.text}],
            }
        ]
    }
    return _run_plan(pid, req.lang, plan, f"Edit line {line}")


class RevoiceRequest(BaseModel):
    lang: str
    voices: dict[str, str]  # {"0": "Kavya", ...}; speakers left out are cast automatically


@app.post("/api/projects/{pid}/revoice")
def revoice(pid: str, req: RevoiceRequest):
    proj = _editable(pid, req.lang)
    _not_busy(pid)
    known = {v for v, _, _ in CATALOGUE}
    picked = {k: v for k, v in req.voices.items() if v in known}
    # Keep earlier choices for the speakers not changed now.
    saved = (proj.get("settings") or {}).get("voice_map") or {}
    vmap = {**saved, **picked}
    # Only the re-cast speakers' lines are voiced again.
    segs = [
        {
            **d,
            "dubbed_audio": None,
            "aligned_audio": None,
            "lip_synced_clip": None,
            "sync_clips": [],
        }
        if str(d.get("speaker") or 0) in picked
        else d
        for d in proj["segments"]
    ]
    state = _state(pid, req.lang, proj, segs, vmap)
    job = JOBS.submit(
        work_s=_speech_seconds(s for s in proj["segments"] if str(s.get("speaker") or 0) in picked),
        kind="revoice",
        project=pid,
        lang=req.lang,
        title="New voices: "
        + ", ".join(f"Speaker {int(k) + 1} → {voice_label(v)}" for k, v in picked.items()),
        stages=_stages(REVOICE_STAGES, state),
        graph=REVOICE_PIPELINE,
        state=state,
    )
    return job.public(JOBS.position(job))


# ── Helpers ───────────────────────────────────────────────────────────────────
def _editable(pid: str, lang: str) -> dict:
    try:
        project_dir(pid)
    except (ValueError, FileNotFoundError) as exc:
        raise _not_found(exc)
    proj = load_project(pid, lang)
    if not proj:
        raise HTTPException(409, "This dub has no saved line data; dub it again to edit it.")
    return proj


def _not_busy(pid: str):
    if JOBS.busy_with(pid):
        raise HTTPException(409, "This video is already being processed; wait for it to finish.")


def _state(pid: str, lang: str, proj: dict, segs: list, vmap: dict) -> dict:
    """Pipeline state for re-running part of a saved dub with its own settings."""
    s = proj.get("settings") or {}
    state = cached_state(
        pid,
        lang,
        s.get("domain") or DEFAULT_DOMAIN,
        s.get("voice_name"),
        s.get("voice_style") or DEFAULT_STYLE,
        vmap,
        segs,
        [],
        s.get("lang_mix") or "standard",
        s.get("tts_backend") or "bodhan",
    )
    if isinstance(state, str):
        raise HTTPException(409, state)
    state.update(
        url=s.get("url") or "",
        translator=s.get("translator") or "indic",
        songs=s.get("songs") or "keep",
        lipsync=s.get("lipsync") is not False,
        mode=s.get("mode") or "indic",
    )
    return state


def _stages(stages: list, state: dict) -> list:
    return stages if state.get("lipsync") is not False else fast_stages(stages)


def _speech_seconds(segs) -> float:
    return sum(s["end"] - s["start"] for s in segs if not s.get("keep_original"))


def _resync_ids(plan: dict) -> tuple[list[int], bool]:
    """Lines a plan asks to lip-sync again, and whether that is all it asks."""
    actions = [a for r in plan["requests"] for a in r["actions"]]
    ids = sorted({i for a in actions if a["type"] == "lip_sync" for i in a["lines"]})
    only = bool(actions) and all(a["type"] == "lip_sync" for a in actions)
    return ids, only


def _run_plan(pid: str, lang: str, plan: dict, title: str, lipsync: bool | None = None) -> dict:
    proj = _editable(pid, lang)
    _not_busy(pid)
    before = proj["segments"]
    saved_map = (proj.get("settings") or {}).get("voice_map") or {}
    # Plans come from the planner or straight from the studio's buttons:
    # check every action against this dub before touching anything.
    plan = validate_plan(plan, [Segment.from_dict(d) for d in before], CATALOGUE, STYLES)
    if not any(r["actions"] for r in plan["requests"]):
        problems = [why for r in plan["requests"] for why in r["rejected"]]
        raise HTTPException(
            400, "Nothing valid to apply" + (f": {problems[0]}" if problems else ".")
        )
    redo = redetect_request(plan)
    if redo:
        return _redetect(pid, lang, proj, redo.get("people"))
    new_segs, vmap, notes, needs_tr = apply_plan(before, plan, saved_map, lang)
    state = _state(pid, lang, proj, new_segs, vmap)
    if lipsync is not None:
        state["lipsync"] = lipsync
    asked = [r["request"] for r in plan["requests"] if r["actions"] and r["request"]]
    resync, only_resync = _resync_ids(plan)
    if resync:
        # Asked for lip sync: it runs, whatever the fast-mode setting says.
        state.update(lipsync=True, resync_ids=resync)
    if only_resync:
        # The audio is fine as it is: only lip sync and the final mix run.
        state["new_audio"] = []
        job = JOBS.submit(
            kind="lipsync",
            project=pid,
            lang=lang,
            title="; ".join(asked) or f"Lip sync again ({len(resync)} lines)",
            stages=LIPSYNC_STAGES,
            graph=LIPSYNC_PIPELINE,
            state=state,
            before=before,
            plan=plan,
            work_s=_speech_seconds(d for d in new_segs if d["id"] in set(resync)),
        )
        return job.public(JOBS.position(job))
    graph, stages = (EDIT_PIPELINE, EDIT_STAGES) if needs_tr else (REVOICE_PIPELINE, REVOICE_STAGES)
    stages = _stages(stages, state)
    # The title is what was asked for, in the user's (or the button's) words.
    job = JOBS.submit(
        kind="edit",
        project=pid,
        lang=lang,
        title="; ".join(asked) or title,
        stages=stages,
        graph=graph,
        state=state,
        before=before,
        plan=plan,
        # Only lines whose audio was cleared by the plan are voiced again.
        work_s=_speech_seconds(s for s in new_segs if not s.get("dubbed_audio")),
    )
    return job.public(JOBS.position(job))


def _redetect(pid: str, lang: str, proj: dict, people: int | None) -> dict:
    """Find the speakers again and dub the whole video again with its settings.
    Download, separation and transcription are reused from the cache."""
    s = proj.get("settings") or {}
    url = s.get("url") or project_view(pid, lang).get("url")
    if not url:
        raise HTTPException(409, "The video's link is unknown, so it cannot be dubbed again.")
    state = base_state(
        pid,
        lang,
        s.get("domain") or DEFAULT_DOMAIN,
        s.get("voice_name"),
        s.get("voice_style") or DEFAULT_STYLE,
        {},
        s.get("lang_mix") or "standard",
        s.get("tts_backend") or "bodhan",
        s.get("translator") or "indic",
        s.get("songs") or "keep",
    )
    m = re.search(r"_([0-9.]+)-([0-9.]+|end)$", pid)
    a, b = (float(m[1]), None if m[2] == "end" else float(m[2])) if m else (None, None)
    state.update(
        url=url,
        clip_start=a,
        clip_end=b,
        num_speakers=people,
        lipsync=s.get("lipsync") is not False,
    )
    job = JOBS.submit(
        kind="edit",
        project=pid,
        lang=lang,
        title="Find speakers again" + (f" ({people} people)" if people else ""),
        stages=_stages(FULL_STAGES, state),
        graph=FULL_PIPELINE,
        state=state,
    )
    return job.public(JOBS.position(job))


class _WebFiles(StaticFiles):
    """The page's own files (HTML, JS, CSS). The browser must check for a newer
    copy every time (a cheap 304 when unchanged), or after an update it keeps
    running the old page against the new server."""

    async def get_response(self, path, scope):
        response = await super().get_response(path, scope)
        response.headers["Cache-Control"] = "no-cache"
        return response


# The web page itself. Mounted last so the API routes above take priority.
app.mount("/", _WebFiles(directory=WEB_DIR, html=True), name="web")
