"""
Version history of a dub: every edit can be looked at again and undone.

A version is a saved copy of a dub's three files: the video, the
soundtrack and its line-by-line state (project_<lang>.json). They live in
cache/<project>/versions/<lang>/ next to index.json, which lists
the versions in order with what each one changed.

  v1          the dub as it was before the first change
  v2, v3, …   one per finished edit, re-voice or re-dub

Restoring never deletes anything: the old version's files are copied back
and recorded as a new version ("Restored v2"), so it can be undone too.
"""

import json
import shutil
import time
from pathlib import Path

from pipeline.cache import CACHE_ROOT


def _dir(pid: str, lang: str) -> Path:
    return CACHE_ROOT / pid / "versions" / lang


def _files(pid: str, lang: str) -> dict[str, Path]:
    """The current dub's files, by the suffix they get inside a version."""
    d = CACHE_ROOT / pid
    return {
        ".mp4": d / f"dubbed_{lang}.mp4",
        ".m4a": d / f"dubbed_{lang}.m4a",
        ".json": d / f"project_{lang}.json",
    }


def _index(pid: str, lang: str) -> list[dict]:
    try:
        return json.loads((_dir(pid, lang) / "index.json").read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        return []


def list_versions(pid: str, lang: str) -> list[dict]:
    """Newest first, each with a link to its video."""
    out = []
    for e in reversed(_index(pid, lang)):
        out.append({**e, "video": f"/media/{pid}/versions/{lang}/v{e['v']}.mp4"})
    return out


def snapshot(pid: str, lang: str, title: str, kind: str, changes: str = "") -> int | None:
    """Save the dub as it is now as the next version. Returns its number."""
    files = _files(pid, lang)
    if not files[".mp4"].exists():
        return None
    d = _dir(pid, lang)
    d.mkdir(parents=True, exist_ok=True)
    index = _index(pid, lang)
    v = (index[-1]["v"] + 1) if index else 1
    for suffix, src in files.items():
        if src.exists():
            shutil.copy2(src, d / f"v{v}{suffix}")
    index.append({"v": v, "title": title, "kind": kind, "changes": changes, "at": time.time()})
    (d / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=1), encoding="utf-8")
    return v


def ensure_first(pid: str, lang: str) -> None:
    """Before the first change to a dub, keep the dub as it was as v1."""
    if not _index(pid, lang):
        snapshot(pid, lang, "Original dub", "dub")


def restore(pid: str, lang: str, v: int) -> int | None:
    """Make version v the current dub again; recorded as a new version."""
    d = _dir(pid, lang)
    if not any(e["v"] == v for e in _index(pid, lang)):
        raise FileNotFoundError(f"v{v}")
    for suffix, dst in _files(pid, lang).items():
        src = d / f"v{v}{suffix}"
        if src.exists():
            shutil.copy2(src, dst)
    return snapshot(pid, lang, f"Restored v{v}", "restore")
