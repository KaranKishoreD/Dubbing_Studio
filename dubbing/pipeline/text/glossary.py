"""
Names & words: a per-video list of terms the dub must get right — a name
that must stay a name ("Muthuvel"), a brand, a technical term to keep in
English ("convex set"), or a word to say a particular way ("Rajini" →
"रजनी").

Stored in cache/<video>/glossary.json as [{"term": ..., "say": ...}], where
an empty "say" means keep the term as it is. Shared by every language of
the video. The translators, the dubbing director and the transcript
clean-up all receive it (block()).
"""

import json
import re

from ..cache import get_cache_dir


def load(video_id: str) -> list[dict]:
    path = get_cache_dir(video_id) / "glossary.json"
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [r for r in rows if isinstance(r, dict) and str(r.get("term") or "").strip()]


def save(video_id: str, rows: list[dict]) -> list[dict]:
    clean = []
    seen = set()
    for r in rows:
        term = str(r.get("term") or "").strip()[:80]
        if not term or term.lower() in seen:
            continue
        seen.add(term.lower())
        clean.append({"term": term, "say": str(r.get("say") or "").strip()[:80]})
    path = get_cache_dir(video_id) / "glossary.json"
    path.write_text(json.dumps(clean, ensure_ascii=False, indent=1), encoding="utf-8")
    return clean


def block(rows: list[dict]) -> str:
    """The list as an instruction for an LLM prompt ("" when empty)."""
    if not rows:
        return ""
    items = "; ".join(
        f'"{r["term"]}" → write exactly "{r["say"]}"' if r["say"] else f'"{r["term"]}" → keep as is'
        for r in rows
    )
    return (
        "NAMES & WORDS (the user's list — always follow it): "
        + items
        + ". Keep names as names (transliterate, never translate their meaning)."
    )


def lines_using(segments: list[dict], rows: list[dict]) -> dict[str, list[int]]:
    """{term: [line ids whose original text contains it]}."""
    out = {}
    for r in rows:
        pat = re.compile(re.escape(r["term"]), re.IGNORECASE)
        out[r["term"]] = [s["id"] for s in segments if pat.search(s.get("text") or "")]
    return out
