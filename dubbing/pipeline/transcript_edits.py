"""
Manual transcript corrections.

The source transcript is what everything downstream is built on — a
mis-heard word becomes a mistranslation. Users can fix it two ways:

  - edit the Original column in the Script tab directly (no model involved);
  - describe the fix in words ("at 0:25 it's Maruti Suzuki, not Baruti") —
    a cheap model maps that onto lines and proposes the corrected text,
    which lands in the table for review before anything is re-dubbed.

apply_source_edits() then updates only the changed lines and clears their
translation and audio, so the edit pipeline re-translates and re-voices
exactly those lines and reuses everything else.
"""

import json
import os
import re

from .state import Segment

FIX_MODEL = os.environ.get("TRANSCRIPT_FIX_MODEL", "google/gemini-3.8-flash")

# Markers the Script table puts in front of a line; never part of the text.
_MARKERS = re.compile(r"^(⚠|＋|✎)\s*")
_LABEL = re.compile(r"^\s*(\[\d+\]\s*)?([\d:.–-]+\s*)?Speaker\s*\d+\s*:\s*", re.I)

_SYSTEM = """\
You correct a video transcript. You get the numbered transcript lines (in the
language actually spoken) and a user's description of what is wrong.

Return the corrected text of ONLY the lines the user's request changes.
Rules:
1. Change only what the user asks for; keep every other word of the line
   exactly as it is, in the same language and script.
2. "at 0:25", "around 25s" → the line whose time range contains it.
   Quoted words → the line containing them.
3. Do not translate. Do not touch lines the request does not concern.
4. "text" is the spoken words only — never include the time or the
   "Speaker N:" label shown in the transcript.

Return ONLY JSON: [{"id": <int>, "text": "<corrected full line>"}]
"""


def strip_markers(text: str) -> str:
    return _MARKERS.sub("", str(text or "")).strip()


def parse_corrections(message: str, segments: list[Segment]) -> dict[int, str]:
    """{line id: corrected text} for a described correction (cheap model)."""
    from openai import OpenAI
    from .edit_resolver import extract_json_array

    ordered = sorted(segments, key=lambda s: s.start)
    transcript = "\n".join(
        f"[{s.id}] {int(s.start) // 60}:{s.start % 60:04.1f}–{int(s.end) // 60}:{s.end % 60:04.1f} "
        f"Speaker {(s.speaker or 0) + 1}: {s.text}" for s in ordered)
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    resp = client.chat.completions.create(
        model=FIX_MODEL,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": f"TRANSCRIPT\n{transcript}\n\nREQUEST\n{message}"}],
        temperature=0.0, timeout=60,
    )
    items = extract_json_array(resp.choices[0].message.content.strip())
    ids = {s.id for s in segments}
    out = {}
    for it in items:
        if not isinstance(it, dict):
            continue
        try:
            sid = int(re.search(r"-?\d+", str(it.get("id"))).group())
        except (AttributeError, ValueError):
            continue
        text = _LABEL.sub("", str(it.get("text") or "")).strip()
        if sid in ids and text:
            out[sid] = text
    return out


def apply_source_edits(segs: list[dict], edits: dict[int, str]) -> tuple[list[dict], list[int]]:
    """Set new source text on the edited lines; returns (segments, changed ids)."""
    changed = []
    out = []
    for d in segs:
        s = Segment.from_dict(d)
        new = strip_markers(edits.get(s.id, s.text))
        if new and new != s.text:
            if s.asr_raw is None:
                s.asr_raw = s.text          # keep what the recogniser heard
            s.text = new
            s.asr_flag = "edited"
            s.asr_alt = None
            s.dubbed_text = None            # → re-translated
            s.edit_instruction = None
            s.dubbed_audio = s.aligned_audio = s.lip_synced_clip = None
            s.sync_clips = []
            changed.append(s.id)
        out.append(s.to_dict())
    return out, changed


def table_edits(rows, segs: list[dict]) -> dict[int, str]:
    """{id: text} for rows whose Original cell differs from the transcript."""
    if rows is None:
        return {}
    if hasattr(rows, "values"):              # pandas DataFrame
        rows = rows.values.tolist()
    current = {d["id"]: d["text"] for d in segs}
    out = {}
    for r in rows:
        if not r or len(r) < 5:
            continue
        try:
            sid = int(r[0])
        except (TypeError, ValueError):
            continue
        text = strip_markers(r[4])
        if sid in current and text and text != current[sid]:
            out[sid] = text
    return out
