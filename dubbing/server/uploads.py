"""
Sources other than a YouTube link: an uploaded file, or a link to any site
yt-dlp supports.

An upload is streamed to cache/_uploads/<id>.mp4 (H.264 + AAC), where <id>
is "u" + 10 hex characters of its SHA-1, so the same file uploaded twice is
the same project. Audio files (podcasts, voice-overs) get a waveform video,
and are dubbed without lip sync. The download stage then takes the file
from there instead of the internet (pipeline/steps/download.py).
"""

import hashlib
import json
import subprocess
from pathlib import Path

from pipeline.steps.download import UPLOADS

MAX_BYTES = 4 * 1024**3


async def save_upload(chunks, filename: str) -> dict:
    """Stream the body to disk, convert it once, return its description."""
    UPLOADS.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS / f"incoming_{hashlib.sha1(filename.encode()).hexdigest()[:8]}"
    sha, size = hashlib.sha1(), 0
    with open(tmp, "wb") as f:
        async for chunk in chunks:
            size += len(chunk)
            if size > MAX_BYTES:
                f.close()
                tmp.unlink(missing_ok=True)
                raise ValueError("That file is over 4 GB — try a shorter part of it.")
            sha.update(chunk)
            f.write(chunk)
    uid = "u" + sha.hexdigest()[:10]
    out, meta = UPLOADS / f"{uid}.mp4", UPLOADS / f"{uid}.json"
    try:
        if not out.exists():
            info = _convert(tmp, out)
            info["title"] = Path(filename).stem.replace("_", " ").strip() or "My video"
            info["channel"] = "Your upload"
            meta.write_text(json.dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    finally:
        tmp.unlink(missing_ok=True)
    return {"source": f"upload:{uid}", **json.loads(meta.read_text(encoding="utf-8"))}


def _convert(src: Path, out: Path) -> dict:
    probe = json.loads(
        subprocess.run(
            ["ffprobe", "-v", "error", "-show_streams", "-show_format", "-of", "json", str(src)],
            capture_output=True,
            text=True,
        ).stdout
        or "{}"
    )
    streams = probe.get("streams") or []
    video = next(
        (
            s
            for s in streams
            if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic") != 1
        ),
        None,
    )
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if not audio:
        raise ValueError("That file has no sound to dub.")
    duration = float((probe.get("format") or {}).get("duration") or 0)
    cmd = ["ffmpeg", "-v", "error", "-i", str(src)]
    if video is None:
        # Audio only: a waveform to look at; the dub replaces the sound.
        cmd += [
            "-filter_complex",
            "[0:a]showwaves=s=1280x720:mode=cline:colors=0xff7a45:rate=25,format=yuv420p[v]",
            "-map",
            "[v]",
            "-map",
            "0:a",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-c:a",
            "aac",
        ]
    else:
        cmd += ["-map", "0:v:0", "-map", "0:a:0"]
        cmd += (
            ["-c:v", "copy"]
            if video.get("codec_name") == "h264"
            else ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p"]
        )
        cmd += ["-c:a", "copy"] if audio.get("codec_name") == "aac" else ["-c:a", "aac"]
    subprocess.run(cmd + ["-movflags", "+faststart", str(out), "-y"], check=True)
    return {"duration": round(duration, 1), "audio_only": video is None}


def probe_link(url: str) -> dict:
    """Title, picture and length of a link, without downloading it — or why it fails."""
    import yt_dlp

    try:
        with yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "noplaylist": True}) as ydl:
            info = ydl.extract_info(url, download=False) or {}
    except Exception as exc:
        msg = str(exc).split("ERROR: ")[-1][:200]
        return {"ok": False, "error": f"Can't open that link: {msg}"}
    if info.get("_type") == "playlist":
        return {"ok": False, "error": "That is a playlist — paste a link to one video."}
    return {
        "ok": True,
        "title": info.get("title"),
        "site": info.get("extractor_key") or info.get("extractor"),
        "thumbnail": info.get("thumbnail"),
        "duration": info.get("duration"),
    }
