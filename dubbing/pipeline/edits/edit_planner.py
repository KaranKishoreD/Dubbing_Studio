"""
Natural-language edits → a typed, validated edit plan → applied segments.

A request is not always "re-translate these lines": "speaker 2 is a man with
a deep voice" is a casting change, and one message can hold several requests.
So:

  1. The model first splits the message into every distinct REQUEST, then
     maps each request to one or more typed ACTIONS (below).
  2. Every action is validated against the project (lines, speakers, voices,
     styles). A request with no valid action is reported as unhandled, never
     dropped.
  3. apply_plan() changes the segments directly and reports what changed, so
     only the necessary stages re-run: text changes re-translate just those
     lines; casting/style changes only re-voice.

Actions
  retranslate       lines, instruction   — tone, wording, register, simpler…
  set_text          line, text           — exact dubbed wording from the user
  recast            speaker, voice       — a specific catalogue voice
  set_gender        speaker, gender      — fixes casting AND verb agreement
  reassign_speaker  lines, speaker       — fix who said a line
  set_style         lines, style         — Bodhan delivery style
  keep_original     lines                — leave undubbed
  dub_again         lines                — undo keep_original
"""

import json
import os
import re

from openai import OpenAI

from ..state import Segment

EDIT_MODEL = os.environ.get("EDIT_MODEL", "anthropic/claude-sonnet-5.5")

_SYSTEM = """\
You are the edit planner of a video dubbing studio. A video has been dubbed
into {lang}. The user describes changes in plain language. Turn the message
into a precise plan.

STEP 1 — list EVERY distinct request in the message, in the user's words.
A message often has several ("make the intro funnier, speaker 2 should be a
man, and don't dub the song"). Missing one is the worst possible error.

STEP 2 — map each request to one or more actions:
  {{"type": "retranslate", "lines": [ids], "instruction": "<for the translator>"}}
      rewording, tone, humour, formality, simpler words, more/less English
      mixing, shorter/longer, fix a mistranslation
  {{"type": "set_text", "line": id, "text": "<exact {lang} dub text>"}}
      ONLY when the user dictates the exact new wording
  {{"type": "recast", "speaker": n, "voice": "<voice name from VOICES>"}}
      change a speaker's voice; choose the best match for any description
      from VOICES, matching gender. Use the measured pitch: "deep", "older",
      "mature", "heavy" → a deep voice; "young", "light", "bright" → a high
      voice. Prefer a voice whose native language is the dub language when
      it fits the description equally well. Never pick the voice another
      speaker already has.
  {{"type": "set_gender", "speaker": n, "gender": "male"|"female"}}
      the speaker's gender was detected wrong. Also add a recast if the user
      describes the voice.
  {{"type": "reassign_speaker", "lines": [ids], "speaker": n}}
      a line was attributed to the wrong person
  {{"type": "set_style", "lines": [ids], "style": "<one of STYLES>"}}
      delivery/emotion only (sad, happy, angry, news-like, storytelling…)
      without changing the words
  {{"type": "keep_original", "lines": [ids]}}   leave these lines undubbed
  {{"type": "dub_again", "lines": [ids]}}       undo keep_original

Resolving references:
- Line ids are the [n] numbers. "Line 3"/"segment 3" means id 3.
- Speakers are numbered from 1 as shown ("Speaker 2").
- "around 45s"/"at 1:20" → the line whose time range contains it, else the
  nearest. "The intro" → first 1-3 lines; "the outro/ending" → last 1-3.
- "Everyone"/"all speakers"/"the whole video" → every speaker / every line.
- Use the dubbed text to find content the user quotes or describes.

If a request is impossible with these actions (e.g. "change the background
music"), still list it, with "actions": [] and a short "note" saying why.

Return ONLY JSON:
{{"requests": [{{"request": "<user's words>", "actions": [...], "note": "<optional>"}}]}}
"""


def plan_edits(
    message: str,
    segments: list[Segment],
    lang_name: str,
    catalogue: list[tuple[str, str, str]],
    styles: list[str],
) -> dict:
    """
    {"requests": [{"request", "actions": [valid actions], "rejected": [reasons],
                   "note"}], "raw": model reply}
    """
    ordered = sorted(segments, key=lambda s: s.start)
    lines = "\n".join(
        f"[{s.id}] {_ts(s.start)}–{_ts(s.end)} Speaker {(s.speaker or 0) + 1} "
        f"({s.gender or '?'}, voice {s.voice or '?'})"
        + (" [UNDUBBED]" if s.keep_original else "")
        + f"\n     original: {s.text}\n     dubbed:   {s.dubbed_text or '—'}"
        for s in ordered
    )
    from ..voice.catalogue import voice_profiles

    prof = voice_profiles()
    voices = "\n".join(
        f"{v} ({g}, native {lang}"
        + (
            f", {prof[v]['pitch']} voice, {prof[v]['f0']:.0f} Hz"
            if v in prof and "pitch" in prof[v]
            else ""
        )
        + ")"
        for v, g, lang in catalogue
    )
    user = (
        f"TRANSCRIPT\n{lines}\n\nVOICES\n{voices}\n\nSTYLES\n"
        + ", ".join(styles)
        + f"\n\nUSER MESSAGE\n{message}"
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"]
    )
    resp = client.chat.completions.create(
        model=EDIT_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM.format(lang=lang_name)},
            {"role": "user", "content": user},
        ],
        temperature=0.1,
        timeout=90,
    )
    raw = (resp.choices[0].message.content or "").strip()
    data = _parse_json(raw)
    reqs = data.get("requests") if isinstance(data, dict) else None
    if not isinstance(reqs, list) or not reqs:
        raise ValueError(f"The planner returned no requests. Reply:\n{raw[:400]}")

    ids = {s.id for s in segments}
    speakers = {(s.speaker or 0) + 1 for s in segments}
    voice_gender = {v: g for v, g, _ in catalogue}
    out = []
    for r in reqs:
        if not isinstance(r, dict):
            continue
        good, bad = [], []
        for a in r.get("actions") or []:
            ok, why = _validate(a, ids, speakers, voice_gender, styles)
            (good if ok else bad).append(a if ok else why)
        out.append(
            {
                "request": str(r.get("request") or "").strip(),
                "actions": good,
                "rejected": bad,
                "note": str(r.get("note") or ""),
            }
        )
    return {"requests": out, "raw": raw}


def _validate(a, ids, speakers, voice_gender, styles) -> tuple[bool, str]:
    if not isinstance(a, dict):
        return False, "malformed action"
    t = a.get("type")

    def line_list(key="lines"):
        v = a.get(key)
        v = v if isinstance(v, list) else [v]
        got = [_as_int(x) for x in v]
        return [x for x in got if x in ids]

    if t in ("retranslate", "reassign_speaker", "set_style", "keep_original", "dub_again"):
        a["lines"] = line_list()
        if not a["lines"]:
            return False, f"{t}: no valid line ids"
    if t == "retranslate" and not str(a.get("instruction") or "").strip():
        return False, "retranslate: empty instruction"
    if t == "set_text":
        a["line"] = _as_int(a.get("line"))
        if a["line"] not in ids or not str(a.get("text") or "").strip():
            return False, "set_text: bad line id or empty text"
    if t in ("recast", "set_gender", "reassign_speaker"):
        a["speaker"] = _as_int(a.get("speaker"))
        if a["speaker"] not in speakers and t != "reassign_speaker":
            return False, f"{t}: no Speaker {a.get('speaker')}"
        if t == "reassign_speaker" and (a["speaker"] is None or a["speaker"] < 1):
            return False, "reassign_speaker: bad speaker number"
    if t == "recast" and a.get("voice") not in voice_gender:
        return False, f"recast: unknown voice {a.get('voice')!r}"
    if t == "set_gender" and a.get("gender") not in ("male", "female"):
        return False, "set_gender: gender must be male or female"
    if t == "set_style" and a.get("style") not in styles:
        return False, f"set_style: unknown style {a.get('style')!r}"
    if t not in (
        "retranslate",
        "set_text",
        "recast",
        "set_gender",
        "reassign_speaker",
        "set_style",
        "keep_original",
        "dub_again",
    ):
        return False, f"unknown action {t!r}"
    return True, ""


def apply_plan(
    segs: list[dict], plan: dict, voice_map: dict
) -> tuple[list[dict], dict, list[str], bool]:
    """
    Apply every action. Returns (segments, voice_map, change notes,
    needs_translation). Downstream outputs of touched lines are cleared, so
    the pipeline recomputes exactly those and reuses everything else.
    """
    segments = [Segment.from_dict(d) for d in segs]
    by_id = {s.id: s for s in segments}
    vmap = dict(voice_map or {})
    notes: list[str] = []

    def reset_audio(s: Segment):
        s.dubbed_audio = s.aligned_audio = s.lip_synced_clip = None
        s.sync_clips = []

    def gender_of(spk0: int) -> str | None:
        return next((s.gender for s in segments if (s.speaker or 0) == spk0 and s.gender), None)

    for req in plan.get("requests", []):
        for a in req.get("actions", []):
            t = a["type"]
            if t == "retranslate":
                for i in a["lines"]:
                    s = by_id[i]
                    s.edit_instruction, s.dubbed_text = a["instruction"], None
                    reset_audio(s)
                notes.append(f"Re-translate {_ids(a['lines'])}: {a['instruction']}")
            elif t == "set_text":
                s = by_id[a["line"]]
                s.dubbed_text = a["text"].strip()
                reset_audio(s)
                notes.append(f"New wording for S{s.id}")
            elif t == "recast":
                spk0 = a["speaker"] - 1
                vmap[str(spk0)] = a["voice"]
                for s in segments:
                    if (s.speaker or 0) == spk0:
                        reset_audio(s)
                notes.append(f"Speaker {a['speaker']} → voice {a['voice']}")
            elif t == "set_gender":
                spk0 = a["speaker"] - 1
                for s in segments:
                    if (s.speaker or 0) == spk0:
                        if s.gender != a["gender"]:
                            # Verb endings agree with the speaker's gender in
                            # Tamil/Hindi, so the words must be redone too.
                            s.gender, s.dubbed_text = a["gender"], None
                        reset_audio(s)
                if vmap.get(str(spk0)) and not _voice_matches(vmap[str(spk0)], a["gender"]):
                    vmap.pop(str(spk0))
                notes.append(f"Speaker {a['speaker']} is {a['gender']}")
            elif t == "reassign_speaker":
                spk0 = a["speaker"] - 1
                g = gender_of(spk0)
                for i in a["lines"]:
                    s = by_id[i]
                    s.speaker = spk0
                    if g and s.gender != g:
                        s.gender, s.dubbed_text = g, None
                    reset_audio(s)
                notes.append(f"{_ids(a['lines'])} → Speaker {a['speaker']}")
            elif t == "set_style":
                for i in a["lines"]:
                    by_id[i].style_override = a["style"]
                    reset_audio(by_id[i])
                notes.append(f"Style '{a['style']}' for {_ids(a['lines'])}")
            elif t in ("keep_original", "dub_again"):
                for i in a["lines"]:
                    by_id[i].keep_original = t == "keep_original"
                    reset_audio(by_id[i])
                notes.append(
                    ("Keep original audio for " if t == "keep_original" else "Dub again: ")
                    + _ids(a["lines"])
                )

    needs_translation = any(s.dubbed_text is None and not s.keep_original for s in segments)
    return [s.to_dict() for s in segments], vmap, notes, needs_translation


def _voice_matches(voice: str, gender: str) -> bool:
    from ..voice.catalogue import voice_catalogue

    return dict((v, g) for v, g, _ in voice_catalogue()).get(voice) == gender


def _ids(ids: list[int]) -> str:
    return ", ".join(f"S{i}" for i in ids)


def _ts(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:04.1f}"


def _as_int(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x)
    m = re.search(r"\d+", str(x))
    return int(m.group()) if m else None


def _parse_json(raw: str):
    text = raw.strip()
    if text.startswith("```"):
        text = "\n".join(l for l in text.split("\n") if not l.strip().startswith("```"))
    # The first complete JSON object; models sometimes add a remark after it.
    a = text.find("{")
    if a != -1:
        try:
            return json.JSONDecoder().raw_decode(text[a:])[0]
        except json.JSONDecodeError:
            pass
    raise ValueError(f"Planner reply was not JSON:\n{raw[:400]}")
