"""
The Download tab: the dubbed video, the dubbed soundtrack, and subtitle files
(original and translated) for a finished project.
"""

from pathlib import Path

from pipeline.cache import get_cache_dir
from pipeline.state import Segment


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


def collect_exports(
    video_id: str, lang: str, segs: list[dict], final_state: dict | None
) -> list[str]:
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
