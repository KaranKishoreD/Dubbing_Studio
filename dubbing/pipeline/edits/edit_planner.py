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
  set_pace          lines, pace          — speak faster / slower / normally
  keep_original     lines                — leave undubbed
  dub_again         lines                — undo keep_original
  new_speaker       lines, gender, voice — these lines are a different person
  merge_speakers    speaker, into        — two speaker labels are one person
  steady_voice      speaker or lines     — one consistent cloned voice
  set_original      line, text           — correct the transcript, re-translate
  set_language      lines, lang          — dub these lines in another language
  own_voice         speaker, ref, gender — clone a speaker from the user's recording
                                           (studio button only, not planned from text)
  lip_sync          lines (or none = all) — redo lip sync with the audio as it is
  redetect_speakers people (optional)    — find the speakers again, re-dub
"""

import json
import os
import re

from openai import OpenAI

from ..state import Segment
from ..text.languages import LANG_NAMES

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
  {{"type": "set_text", "line": id, "text": "<exact dub text>"}}
      ONLY when the user dictates the exact new wording of the dub
  {{"type": "set_original", "line": id, "text": "<corrected original>"}}
      the ORIGINAL transcript is wrong: a mis-heard word or name, words
      nobody said, filler or unwanted words to drop. Write the full
      corrected line in the original language and script; the line is then
      translated again from it.
  {{"type": "set_language", "lines": [ids], "lang": "<code from LANGUAGES>"}}
      dub these lines in another language than {lang} ("keep lines 3-5 in
      Bengali", "the song part in English"). The rest stays in {lang}.
      Use the dub's own code to put lines back into {lang}.
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
      without changing the words. "More energetic", "sadder", "calmer",
      "more excited", "angrier" are about HOW a line is said: use set_style
      ALONE — never retranslate for them unless the user also asks for
      different words. NEVER use set_style for speed: speed is set_pace.
  {{"type": "set_pace", "lines": [ids], "pace": "fast"|"slow"|"normal"}}
      how FAST the line is spoken: "faster", "too slow", "rattling off",
      "blasting through it", "rushed", "like the original's speed" → fast;
      "slower", "too rushed", "calm and deliberate" → slow. The studio picks
      the fastest/slowest-sounding delivery measured for that voice. If the
      words are also too long to say quickly, add retranslate with an
      instruction to keep it short.
  {{"type": "keep_original", "lines": [ids]}}   leave these lines undubbed
  {{"type": "dub_again", "lines": [ids]}}       undo keep_original
  {{"type": "new_speaker", "lines": [ids], "gender": "male"|"female", "voice": "<optional, from VOICES>"}}
      these lines are spoken by a person who has no speaker of their own yet
      (the speaker detection merged two people). Also use it when the user
      says there are more people than speakers shown.
  {{"type": "merge_speakers", "speaker": n, "into": m}}
      Speaker n and Speaker m are the same person (fewer people than shown)
  {{"type": "steady_voice", "speaker": n}}   or   {{"type": "steady_voice", "lines": [ids]}}
      the dub's voice sounds inconsistent, like someone else, or the wrong
      gender on some lines. Each line is normally cloned from voices that
      sound like that line's original audio, so music, a chorus or another
      person under a line can leak into it. This re-clones the lines from
      the speaker's own clean reference: one steady voice, and never the
      wrong gender.
  {{"type": "lip_sync", "lines": [ids]}}   or   {{"type": "lip_sync"}} for the whole video
      the mouths do not move with the dub, lip sync is missing, looks off,
      or the user just asks to (re)do lip sync. The audio stays exactly as
      it is; only the faces are rendered again. Use it alone unless the
      user also asks for other changes.
  {{"type": "redetect_speakers", "people": n|null}}
      find who speaks each line again from scratch, optionally with the
      number of distinct people the user says there are, then dub the whole
      video again (earlier edits are not kept; the old version stays in
      History). Use it only when the user asks to redo speaker detection or
      the speaker labels are wrong almost everywhere. Use it alone.

Do only what was asked. Every action redoes its lines (new speech, voice
and lip sync), so never add actions the user did not ask for, and never
include lines beyond the ones the request is about.

Resolving references:
- Line ids are the [n] numbers. "Line 3"/"segment 3" means id 3.
- Speakers are numbered from 1 as shown ("Speaker 2").
- "around 45s"/"at 1:20" → the line whose time range contains it, else the
  nearest. "The intro" → first 1-3 lines; "the outro/ending" → last 1-3.
- "Everyone"/"all speakers"/"the whole video" → every speaker / every line.
- Use the dubbed text to find content the user quotes or describes.

Vague requests: the user reviews the plan before anything runs, so do the
most likely fix instead of asking a question. Use the evidence:
- Names: work out which speaker a named person is from the transcript
  (who is addressed, who speaks most, what they say) and SPEAKERS.
- "X's voice became female/changed/inconsistent", "X has several voices":
  steady_voice on X's speaker (or just the lines described). Lines whose
  original pitch is far above their speaker's usual pitch usually have
  music or another voice underneath.
- "X's lines are spoken by Y's voice": reassign_speaker or new_speaker.
- "Voices not dubbed"/"missing": lines marked [UNDUBBED] → dub_again.
State the guess in "note" (e.g. "Taking Speaker 1 as Rajini — he speaks
most and is addressed as Muthuvel"). Leave "actions" empty only when
nothing fits at all (e.g. "change the background music"), with a short
"note" saying why.

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
        f"({s.gender or '?'}, voice {s.voice or '?'}"
        + (f", original pitch {s.f0:.0f} Hz" if s.f0 else "")
        + ")"
        + (" [UNDUBBED]" if s.keep_original else "")
        + (" [STEADY VOICE]" if s.clone_ref == "speaker" else "")
        + (
            f" [IN {LANG_NAMES.get(s.lang_override, s.lang_override).upper()}]"
            if s.lang_override
            else ""
        )
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
        f"SPEAKERS\n{_speakers(ordered)}\n\nTRANSCRIPT\n{lines}\n\nVOICES\n{voices}\n\nSTYLES\n"
        + ", ".join(styles)
        + "\n\nLANGUAGES\n"
        + ", ".join(f"{k} = {v}" for k, v in LANG_NAMES.items())
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


def validate_plan(plan: dict, segments: list[Segment], catalogue, styles) -> dict:
    """
    Check a plan that did not come from plan_edits (the studio's buttons
    build plans directly). Invalid actions are dropped and reported.
    """
    ids = {s.id for s in segments}
    speakers = {(s.speaker or 0) + 1 for s in segments}
    voice_gender = {v: g for v, g, _ in catalogue}
    out = []
    for r in plan.get("requests") or []:
        good, bad = [], []
        for a in r.get("actions") or []:
            ok, why = _validate(a, ids, speakers, voice_gender, styles)
            (good if ok else bad).append(a if ok else why)
        out.append(
            {"request": str(r.get("request") or ""), "actions": good, "rejected": bad, "note": ""}
        )
    return {"requests": out}


def _validate(a, ids, speakers, voice_gender, styles) -> tuple[bool, str]:
    if not isinstance(a, dict):
        return False, "malformed action"
    t = a.get("type")

    def line_list(key="lines"):
        v = a.get(key)
        v = v if isinstance(v, list) else [v]
        got = [_as_int(x) for x in v]
        return [x for x in got if x in ids]

    if t in (
        "retranslate",
        "reassign_speaker",
        "set_style",
        "keep_original",
        "dub_again",
        "new_speaker",
    ):
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
    if t == "new_speaker":
        if a.get("gender") not in ("male", "female"):
            return False, "new_speaker: gender must be male or female"
        if a.get("voice") and a["voice"] not in voice_gender:
            return False, f"new_speaker: unknown voice {a.get('voice')!r}"
    if t == "merge_speakers":
        a["speaker"], a["into"] = _as_int(a.get("speaker")), _as_int(a.get("into"))
        if a["speaker"] not in speakers or a["into"] not in speakers or a["speaker"] == a["into"]:
            return False, "merge_speakers: needs two different existing speakers"
    if t == "steady_voice":
        if a.get("lines") is not None:
            a["lines"] = line_list()
            a.pop("speaker", None)
            if not a["lines"]:
                return False, "steady_voice: no valid line ids"
        else:
            a.pop("lines", None)
            a["speaker"] = _as_int(a.get("speaker"))
            if a["speaker"] not in speakers:
                return False, f"steady_voice: no Speaker {a.get('speaker')}"
    if t == "lip_sync":
        # No lines (or "all") means the whole video; lines kept in the
        # original audio are skipped by the lipsync stage anyway.
        if a.get("lines") in (None, "all", []):
            a["lines"] = sorted(ids)
        else:
            a["lines"] = line_list()
        if not a["lines"]:
            return False, "lip_sync: no valid line ids"
    if t == "set_original":
        a["line"] = _as_int(a.get("line"))
        if a["line"] not in ids or not str(a.get("text") or "").strip():
            return False, "set_original: bad line id or empty text"
    if t == "set_language":
        a["lines"] = line_list()
        if not a["lines"]:
            return False, "set_language: no valid line ids"
        if a.get("lang") not in LANG_NAMES:
            return False, f"set_language: unknown language {a.get('lang')!r}"
    if t == "own_voice":
        a["speaker"] = _as_int(a.get("speaker"))
        if a["speaker"] not in speakers:
            return False, f"own_voice: no Speaker {a.get('speaker')}"
        if not re.fullmatch(r"myvoice_s\d+_[0-9a-f]{8}\.wav", str(a.get("ref") or "")):
            return False, "own_voice: no recording"
        if a.get("gender") not in (None, "male", "female"):
            return False, "own_voice: gender must be male or female"
    if t == "redetect_speakers":
        n = _as_int(a.get("people"))
        a["people"] = n if n and 1 <= n <= 20 else None
    if t == "set_gender" and a.get("gender") not in ("male", "female"):
        return False, "set_gender: gender must be male or female"
    if t == "set_pace":
        a["lines"] = line_list()
        if not a["lines"]:
            return False, "set_pace: no valid line ids"
        if a.get("pace") not in ("fast", "slow", "normal"):
            return False, "set_pace: pace must be fast, slow or normal"
    if t == "set_style" and a.get("style") not in styles:
        return False, f"set_style: unknown style {a.get('style')!r}"
    if t not in (
        "retranslate",
        "set_text",
        "recast",
        "set_gender",
        "reassign_speaker",
        "set_style",
        "set_pace",
        "keep_original",
        "dub_again",
        "new_speaker",
        "merge_speakers",
        "steady_voice",
        "redetect_speakers",
        "set_original",
        "set_language",
        "own_voice",
        "lip_sync",
    ):
        return False, f"unknown action {t!r}"
    return True, ""


def apply_plan(
    segs: list[dict], plan: dict, voice_map: dict, dub_lang: str | None = None
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
                notes.append(f"New wording for line {s.id}")
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
            elif t == "set_pace":
                from ..speech.pace import style_for_speed

                for i in a["lines"]:
                    s = by_id[i]
                    lang = s.lang_override or dub_lang or "hi"
                    s.style_override = style_for_speed(lang, s.voice, a["pace"])
                    reset_audio(s)
                notes.append(f"Pace {a['pace']} for {_ids(a['lines'])}")
            elif t == "new_speaker":
                spk0 = max((s.speaker or 0) for s in segments) + 1
                for i in a["lines"]:
                    s = by_id[i]
                    s.speaker = spk0
                    if s.gender != a["gender"]:
                        # Verb endings follow the speaker's gender: reword too.
                        s.gender, s.dubbed_text = a["gender"], None
                    reset_audio(s)
                if a.get("voice"):
                    vmap[str(spk0)] = a["voice"]
                notes.append(f"{_ids(a['lines'])} → new Speaker {spk0 + 1} ({a['gender']})")
            elif t == "merge_speakers":
                src, dst = a["speaker"] - 1, a["into"] - 1
                g = gender_of(dst)
                for s in segments:
                    if (s.speaker or 0) == src:
                        s.speaker = dst
                        if g and s.gender != g:
                            s.gender, s.dubbed_text = g, None
                        reset_audio(s)
                vmap.pop(str(src), None)
                notes.append(f"Speaker {a['speaker']} merged into Speaker {a['into']}")
            elif t == "set_original":
                s = by_id[a["line"]]
                s.text = a["text"].strip()
                # The other recogniser's reading and the "check" flag were
                # about the old text.
                s.asr_alt = s.asr_flag = None
                s.dubbed_text = s.emotion = None
                reset_audio(s)
                notes.append(f"Corrected transcript of line {s.id}")
            elif t == "set_language":
                lang = None if a["lang"] == dub_lang else a["lang"]
                for i in a["lines"]:
                    s = by_id[i]
                    if s.lang_override != lang:
                        s.lang_override, s.dubbed_text = lang, None
                        reset_audio(s)
                notes.append(f"{_ids(a['lines'])} in {LANG_NAMES[a['lang']]}")
            elif t == "own_voice":
                spk0 = a["speaker"] - 1
                for s in segments:
                    if (s.speaker or 0) == spk0:
                        s.clone_ref = f"file:{a['ref']}"
                        if a.get("gender") and s.gender != a["gender"]:
                            s.gender, s.dubbed_text = a["gender"], None  # verb endings follow
                        reset_audio(s)
                vmap.pop(str(spk0), None)  # cloned from the recording, not a picked voice
                notes.append(f"Speaker {a['speaker']} speaks in your recorded voice")
            elif t == "lip_sync":
                # The audio is kept; the lipsync stage renders these lines
                # again (server/app.py passes them on as resync_ids).
                for i in a["lines"]:
                    by_id[i].synced = False
                notes.append(f"Lip sync redone on {len(a['lines'])} line(s)")
            elif t == "steady_voice":
                if a.get("lines"):
                    picked = [by_id[i] for i in a["lines"]]
                    what = _ids(a["lines"])
                else:
                    picked = [s for s in segments if (s.speaker or 0) == a["speaker"] - 1]
                    what = f"Speaker {a['speaker']}"
                for s in picked:
                    s.clone_ref = "speaker"
                    reset_audio(s)
                notes.append(f"One steady voice for {what}")
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


def redetect_request(plan: dict) -> dict | None:
    """The redetect_speakers action in a plan, if any. It replaces the dub."""
    return next(
        (
            a
            for r in plan.get("requests", [])
            for a in r.get("actions", [])
            if a.get("type") == "redetect_speakers"
        ),
        None,
    )


def _speakers(ordered: list[Segment]) -> str:
    """One row per speaker: how much they speak and their usual pitch."""
    rows = []
    for spk in sorted({s.speaker or 0 for s in ordered}):
        mine = [s for s in ordered if (s.speaker or 0) == spk]
        f0s = sorted(s.f0 for s in mine if s.f0)
        f0 = f"usual pitch {f0s[len(f0s) // 2]:.0f} Hz" if f0s else "pitch unknown"
        rows.append(
            f"Speaker {spk + 1}: {mine[0].gender or '?'}, {len(mine)} lines, "
            f"{sum(s.duration for s in mine):.0f} s, {f0}, first at {_ts(mine[0].start)}"
        )
    return "\n".join(rows)


def _voice_matches(voice: str, gender: str) -> bool:
    from ..voice.catalogue import voice_catalogue

    return dict((v, g) for v, g, _ in voice_catalogue()).get(voice) == gender


def _ids(ids: list[int]) -> str:
    return ("line " if len(ids) == 1 else "lines ") + ", ".join(str(i) for i in ids)


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
