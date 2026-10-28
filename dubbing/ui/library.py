"""
The Library tab: every finished dub in the cache, newest first, playable
next to its original video.
"""

import json
import time
from pathlib import Path

from pipeline.cache import CACHE_ROOT
from pipeline.text.languages import LANG_NAMES


def library_choices() -> list[tuple[str, str]]:
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
        rows.append(
            (
                latest.stat().st_mtime,
                f"{title[:60]} — {LANG_NAMES.get(lang, lang)} — {when}{test}",
                str(latest),
            )
        )
    return [(label, path) for _, label, path in sorted(rows, reverse=True)]


def load_library_item(path: str | None):
    if not path or not Path(path).exists():
        return None, None
    original = Path(path).parent / "video.mp4"
    return path, (str(original) if original.exists() else None)
