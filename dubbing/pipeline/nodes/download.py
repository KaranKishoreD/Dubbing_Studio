import ffmpeg
import yt_dlp
from pathlib import Path
from ..state import PipelineState
from ..cache import get_cache_dir


def download_node(state: PipelineState) -> dict:
    url = state["url"]
    video_id = state["video_id"]
    cache_dir = get_cache_dir(video_id)
    video_path = cache_dir / "video.mp4"
    audio_path = cache_dir / "audio.wav"

    if not video_path.exists():
        ydl_opts = {
            "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
            "outtmpl": str(video_path.with_suffix("")),
            "merge_output_format": "mp4",
            "quiet": True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

    if not audio_path.exists():
        (
            ffmpeg
            .input(str(video_path))
            .output(str(audio_path), ar=16000, ac=1, acodec="pcm_s16le")
            .overwrite_output()
            .run(quiet=True)
        )

    return {
        "video_path": str(video_path),
        "audio_path": str(audio_path),
        "log": state.get("log", []) + [f"✓ Download complete: {video_path.name}"],
    }
