"""
Transcript refinement: an LLM reconciles the recognisers into one clean,
faithful transcript before anything is translated.

Each line arrives with up to two transcriptions of the same audio (Bodhan
indic-transcribe and Whisper), which fail in different places: Bodhan can
return junk ("ୄୄୄୄ") on a line Whisper heard fine, Whisper drifts into other
scripts or misspells brands over music. Translating the wrong one turned a
spoken question into "உருளைக்கிழங்கு". Here the model sees both, the
neighbouring lines and the video title, and writes what was actually said —
in the source language, never translated, never invented.

Per line it returns:
  text        the corrected transcript ("[inaudible]" for words it cannot
              recover from either recogniser)
  confidence  high | low   — low lines are flagged in the Script tab
  kind        speech | singing | noise — noise is left undubbed; anything
              spoken or sung is dubbed

Results are cached against a hash of their input, so a re-dub of an
unchanged video reuses the same transcript instead of re-rolling it.
"""

import hashlib
import json
import os

from ..cache import get_cache_dir
from ..state import PipelineState, Segment

REFINE_MODEL = os.environ.get("REFINE_MODEL", os.environ.get("TRANSLATE_MODEL",
                                                             "anthropic/claude-sonnet-5.5"))
_ENABLED = os.environ.get("REFINE_TRANSCRIPT", "1") == "1"
_CHUNK = 40

_SYSTEM = """\
You are a meticulous transcript editor for a video dubbing studio. You get a
video's lines, each with one or two automatic transcriptions of the SAME
audio ("text" and "alt") from different speech recognisers, plus who speaks,
the timing, and the video title. Write what was actually said.

Rules:
1. Stay in the language(s) actually spoken, in the script the line uses.
   Do NOT translate. Code-mixed speech stays code-mixed.
2. The two recognisers make different mistakes. Compare them, use the
   neighbouring lines and the title, and keep whichever words are right.
   Fix mis-heard words, brand and name spellings, and obvious grammar slips
   the recogniser introduced. Remove recogniser junk: characters from
   unrelated scripts, repeated symbols, filler like "Thank you for watching"
   that nobody said.
3. NEVER add content neither recogniser supports. If a word cannot be
   recovered, write [inaudible] in its place and set confidence "low".
4. confidence: "high" if you are sure of the line, "low" if either
   recogniser was garbage and you had to guess, or words are [inaudible].
5. kind: "speech" (talking, announcements, voice-over), "singing" (sung
   lyrics), or "noise" (no real words — music, laughter, a cough).
6. Same ids, one output per input line. Do not merge or split lines.

Return ONLY JSON:
[{"id": <int>, "text": "<clean transcript>", "confidence": "high"|"low",
  "kind": "speech"|"singing"|"noise"}]
"""


def refine_node(state: PipelineState) -> dict:
    segments = [Segment.from_dict(d) for d in state["segments"]]
    log = state.get("log", [])
    if not _ENABLED or not segments or not os.environ.get("OPENROUTER_API_KEY"):
        return {"log": log + ["  Transcript refinement skipped"]}

    cache_dir = get_cache_dir(state["video_id"])
    title = _title(cache_dir)
    ordered = sorted(segments, key=lambda s: s.start)
    lines = [{"id": s.id, "time": f"{s.start:.1f}-{s.end:.1f}",
              "speaker": (s.speaker or 0) + 1, "text": s.text,
              **({"alt": s.asr_alt} if s.asr_alt and s.asr_alt != s.text else {})}
             for s in ordered]
    key = hashlib.sha1(json.dumps([REFINE_MODEL, title, lines], ensure_ascii=False)
                       .encode()).hexdigest()[:16]
    cache_path = cache_dir / f"refined_{key}.json"

    if cache_path.exists():
        result = {int(k): v for k, v in json.loads(cache_path.read_text(encoding="utf-8")).items()}
        how = "cached"
    else:
        result = {}
        for i in range(0, len(lines), _CHUNK):
            try:
                result.update(_call(title, lines, lines[i:i + _CHUNK]))
            except Exception as exc:
                print(f"  ⚠ transcript refinement failed for lines {i}+: {exc}")
        if result:
            cache_path.write_text(json.dumps(result, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        how = REFINE_MODEL.split("/")[-1]

    changed = low = noise = 0
    for s in segments:
        r = result.get(s.id)
        if not r or not str(r.get("text") or "").strip():
            continue
        new = str(r["text"]).strip()
        if new != s.text:
            s.asr_raw = s.text
            s.text = new
            changed += 1
        if r.get("confidence") == "low":
            s.asr_flag = "low"
            low += 1
        else:
            # Reconciled: the second opinion has done its job and would only
            # re-introduce the junk the translator then has to argue with.
            s.asr_alt = None
        if r.get("kind") == "noise":
            s.keep_original = True
            noise += 1

    note = (f"✓ Transcript refined ({how}): {changed} line(s) corrected"
            + (f", {low} uncertain (flagged)" if low else "")
            + (f", {noise} non-speech left undubbed" if noise else ""))
    if not result:
        note = "⚠ Transcript refinement failed — using the raw transcript"
    return {"segments": [s.to_dict() for s in segments], "log": log + [note]}


def _call(title: str, all_lines: list[dict], chunk: list[dict]) -> dict[int, dict]:
    from openai import OpenAI
    from ..edit_resolver import extract_json_array
    client = OpenAI(base_url="https://openrouter.ai/api/v1",
                    api_key=os.environ["OPENROUTER_API_KEY"])
    ids = {l["id"] for l in chunk}
    context = [l for l in all_lines if l["id"] not in ids]
    payload = {"video_title": title, "lines_to_edit": chunk,
               "other_lines_for_context": context[:80]}
    resp = client.chat.completions.create(
        model=REFINE_MODEL,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
        temperature=0.0, timeout=180,
    )
    items = extract_json_array(resp.choices[0].message.content.strip())
    return {int(it["id"]): it for it in items
            if isinstance(it, dict) and str(it.get("id", "")).lstrip("-").isdigit()
            and int(it["id"]) in ids}


def _title(cache_dir) -> str:
    try:
        return json.loads((cache_dir / "meta.json").read_text(encoding="utf-8")).get("title") or ""
    except Exception:
        return ""
