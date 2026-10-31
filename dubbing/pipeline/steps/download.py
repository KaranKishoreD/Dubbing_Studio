"""
Stage 1 — Download: fetch the video and make the two audio copies.

  1. yt-dlp downloads the video (capped at 720p, H.264 preferred so it can
     be copied into the output without re-encoding) from any site it
     supports. If the user chose a clip range, only that range is
     downloaded. A file uploaded in the studio ("upload:<id>") is taken from
     cache/_uploads instead, cut to the range if one was chosen.
  2. audio.wav: 16 kHz mono, used only for analysis (speech recognition,
     voice activity, pitch).
  3. audio_hq.wav: 44.1 kHz stereo, the base of the final mix, so the
     output never inherits the low analysis quality.

Title, channel and length are saved to meta.json for the project card.
"""

import os
from pathlib import Path

import ffmpeg
import yt_dlp

from ..cache import get_cache_dir
from ..state import PipelineState


def download_node(state: PipelineState) -> dict:
    url = state["url"]
    video_id = state["video_id"]
    cache_dir = get_cache_dir(video_id)
    video_path = cache_dir / "video.mp4"
    audio_path = cache_dir / "audio.wav"

    if not video_path.exists() and url.startswith("upload:"):
        _from_upload(url.split(":", 1)[1], state, cache_dir, video_path)
    if not video_path.exists():
        # The video stream is copied into the output untouched, so resolution
        # only costs download time. Unbounded "bestvideo" pulls 1080p60/4K;
        # capping height and preferring H.264 keeps it fast and copy-friendly.
        h = os.environ.get("MAX_VIDEO_HEIGHT", "720")
        ydl_opts = {
            "format": (
                f"bestvideo[height<={h}][vcodec^=avc1]+bestaudio[ext=m4a]/"
                f"bestvideo[height<={h}][ext=mp4]+bestaudio[ext=m4a]/"
                f"best[height<={h}][ext=mp4]/best[height<={h}]/best"
            ),
            "outtmpl": str(video_path.with_suffix("")),
            "merge_output_format": "mp4",
            "concurrent_fragment_downloads": 8,
            "quiet": True,
            "noprogress": True,
        }
        # YouTube sometimes demands sign-in ("confirm you're not a bot").
        # Reuse a browser's logged-in cookies, or an exported cookies.txt.
        browser = os.environ.get("YTDLP_COOKIES_BROWSER", "").strip()
        cookie_file = os.environ.get("YTDLP_COOKIES_FILE", "").strip()
        if cookie_file:
            ydl_opts["cookiefile"] = cookie_file
        elif browser:
            ydl_opts["cookiesfrombrowser"] = (browser,)
        # Part of the video only: yt-dlp fetches just that range (cut on exact
        # frames), so a 49 s clip of a 30 min video downloads 49 s.
        start, end = state.get("clip_start"), state.get("clip_end")
        if start is not None or end is not None:
            from yt_dlp.utils import download_range_func

            ydl_opts["download_ranges"] = download_range_func(
                None, [(start or 0.0, end if end is not None else float("inf"))]
            )
            ydl_opts["force_keyframes_at_cuts"] = True
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True) or {}
        if start is not None or end is not None:
            info = dict(info)
            full = info.get("duration") or 0
            info["duration"] = (end if end is not None else full) - (start or 0)
            info["title"] = (
                f"{info.get('title') or ''} [{_ts(start or 0)}–{_ts(end) if end is not None else 'end'}]"
            )
        _save_meta(cache_dir, url, info)

    # 16 kHz mono is what ASR wants, but it is phone-call quality. The final
    # mix is built on a separate full-quality copy so the output does not
    # inherit that downgrade.
    if not audio_path.exists():
        (
            ffmpeg.input(str(video_path))
            .output(str(audio_path), ar=16000, ac=1, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )
    audio_hq = ensure_hq_audio(video_path, cache_dir)

    return {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "audio_hq_path": str(audio_hq),
        "log": state.get("log", []) + [f"✓ Download complete: {video_path.name}"],
    }


UPLOADS = Path("cache") / "_uploads"


def _from_upload(uid: str, state: PipelineState, cache_dir: Path, video_path: Path) -> None:
    """The uploaded file as video.mp4 (H.264 + AAC), cut to the clip range."""
    import json
    import subprocess

    src = UPLOADS / f"{uid}.mp4"
    if not src.exists():
        raise FileNotFoundError("The uploaded file is gone — upload it again.")
    start, end = state.get("clip_start"), state.get("clip_end")
    cmd = ["ffmpeg", "-v", "error"]
    if start:
        cmd += ["-ss", str(start)]
    if end is not None:
        cmd += ["-to", str(end)]
    cmd += ["-i", str(src)]
    if start or end is not None:  # exact cut: re-encode
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-c:a", "aac"]
    else:
        cmd += ["-c", "copy"]
    subprocess.run(cmd + ["-movflags", "+faststart", str(video_path), "-y"], check=True)
    info = json.loads((UPLOADS / f"{uid}.json").read_text(encoding="utf-8"))
    if start is not None or end is not None:
        full = info.get("duration") or 0
        info["duration"] = (end if end is not None else full) - (start or 0)
        info["title"] = (
            f"{info['title']} [{_ts(start or 0)}–{_ts(end) if end is not None else 'end'}]"
        )
    _save_meta(cache_dir, f"upload:{uid}", info)


def _ts(t: float) -> str:
    return f"{int(t) // 60}:{int(t) % 60:02d}"


def _save_meta(cache_dir: Path, url: str, info: dict) -> None:
    """Title, channel and length for the studio's project card."""
    import json

    meta = {
        "url": url,
        "title": info.get("title"),
        "channel": info.get("channel") or info.get("uploader"),
        "duration": info.get("duration"),
        "thumbnail": info.get("thumbnail"),
    }
    (cache_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"
    )


def ensure_hq_audio(video_path: Path, cache_dir: Path) -> Path:
    """44.1 kHz stereo — Demucs' native rate, so separation needs no resample."""
    out = cache_dir / "audio_hq.wav"
    if not out.exists():
        (
            ffmpeg.input(str(video_path))
            .output(str(out), ar=44100, ac=2, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )
    return out
