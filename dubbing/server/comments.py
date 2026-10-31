"""
Review comments on a dub: anyone with the studio or a watch link can mark a
moment and leave a note with their name. Stored per video and language in
cache/<id>/comments_<lang>.json; the studio shows them on the timeline and
turns any of them into a change request (Ask tab) with one click.
"""

import json
import secrets
import threading
import time

from pipeline.cache import CACHE_ROOT

_LOCK = threading.Lock()


def _path(pid: str, lang: str):
    return CACHE_ROOT / pid / f"comments_{lang}.json"


def load(pid: str, lang: str) -> list[dict]:
    try:
        return json.loads(_path(pid, lang).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []


def _write(pid: str, lang: str, rows: list[dict]) -> None:
    _path(pid, lang).write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def add(pid: str, lang: str, name: str, text: str, t: float, t2: float | None) -> dict:
    name, text = name.strip()[:60], text.strip()[:1000]
    if not name or not text:
        raise ValueError("Please add your name and a comment.")
    row = {
        "id": secrets.token_hex(4),
        "name": name,
        "text": text,
        "t": round(max(0.0, t), 2),
        "t2": round(t2, 2) if t2 is not None and t2 > t else None,
        "at": time.time(),
        "done": False,
    }
    with _LOCK:
        rows = load(pid, lang)
        rows.append(row)
        _write(pid, lang, rows)
    return row


def update(pid: str, lang: str, cid: str, done: bool | None = None, delete: bool = False) -> None:
    with _LOCK:
        rows = load(pid, lang)
        if delete:
            rows = [r for r in rows if r["id"] != cid]
        else:
            for r in rows:
                if r["id"] == cid and done is not None:
                    r["done"] = done
        _write(pid, lang, rows)
