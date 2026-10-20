import os
import json
import re
from openai import OpenAI
from .state import Segment


RESOLVER_SYSTEM = """\
You are an edit resolver for a video dubbing pipeline.

Given a numbered transcript with segment IDs and timestamps, parse the user's
natural-language edit request into discrete, actionable operations.

Rules:
1. Each operation targets one or more COMPLETE segment IDs — never split a segment mid-sentence
2. "Around Xs" / "at Xs" → find the segment whose time window contains or is closest to time X
3. A joke / punchline may span 2 consecutive segments — include both if the setup and punchline are in adjacent segments
4. Multiple independent requests in one message → multiple separate operations (they run in parallel)
5. If two requests overlap the same segment → merge into one operation with a combined instruction
6. "The intro" → first 1-3 segments; "the outro" → last 1-3 segments; "the explanation at Xm" → segment near X minutes

The transcript labels segments [S0], [S1], … but "segment_ids" must contain
PLAIN INTEGERS ONLY — write [1, 2, 3], never ["S1", "S2"] and never [S1, S2].

Return ONLY valid JSON (no markdown, no commentary):
[
  {
    "segment_ids": [1, 2],
    "instruction": "<concise tone/style/language instruction for the translator>",
    "reason": "<one-line explanation of why these segments were chosen>"
  }
]
"""


def resolve_edits(user_message: str, segments: list[Segment]) -> list[dict]:
    """
    Parse a natural-language edit request into structured edit operations.
    Each operation targets specific segment IDs with a translation instruction.
    """
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
    )

    transcript_context = "\n".join(
        f"[S{s.id}] {s.start:.1f}s–{s.end:.1f}s [{s.lang}]: {s.text}"
        for s in segments
    )

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {"role": "system", "content": RESOLVER_SYSTEM},
            {
                "role": "user",
                "content": f"Transcript:\n{transcript_context}\n\nEdit request: {user_message}",
            },
        ],
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()
    ops = extract_json_array(raw)

    # Validate and clamp segment IDs to actual range
    valid_ids = {s.id for s in segments}
    cleaned = []
    for op in ops:
        if not isinstance(op, dict):
            continue
        ids = [sid for sid in op.get("segment_ids", []) if sid in valid_ids]
        if ids and op.get("instruction"):
            op["segment_ids"] = ids
            cleaned.append(op)
    return cleaned


def extract_json_array(raw: str) -> list:
    """
    Pull a JSON array out of an LLM response.

    Models wrap JSON in code fences, prefix it with "Here are the operations:",
    or append a closing remark. Rather than trust the whole string, locate the
    outermost bracket pair and parse that.
    """
    text = raw.strip()

    # Strip ``` fences, with or without a language tag
    if text.startswith("```"):
        lines = [l for l in text.split("\n") if not l.strip().startswith("```")]
        text = "\n".join(lines).strip()

    text = _repair(text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass

    # Scan for the outermost [...], respecting brackets inside strings
    start = text.find("[")
    if start != -1:
        depth, in_str, esc = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == '"':
                in_str = not in_str
            elif not in_str:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(_repair(text[start : i + 1]))
                        except json.JSONDecodeError:
                            break

    raise ValueError(
        f"Model did not return parseable JSON. First 300 chars:\n{raw[:300]}"
    )


def _repair(text: str) -> str:
    """
    Patch the near-JSON that models actually emit.

    The transcript is shown to the model as [S0], [S1], … so it tends to echo
    that label into segment_ids as a bare token: [S1, S2, S3]. `S1` is not a
    JSON value, so the whole document fails to parse. Quoted forms ("S1") and
    trailing commas get the same treatment.
    """
    # S1 / "S1" appearing where an array element belongs -> 1
    text = re.sub(r'(?<=[\[,])(\s*)"?[Ss](\d+)"?(\s*)(?=[,\]])', r"\1\2\3", text)
    # Trailing commas before a closing bracket or brace
    text = re.sub(r",(\s*[\]}])", r"\1", text)
    return text
