import os
import json
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

Return ONLY valid JSON (no markdown, no commentary):
[
  {
    "segment_ids": [<int>, ...],
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
    if raw.startswith("```"):
        raw = "\n".join(l for l in raw.split("\n") if not l.startswith("```")).strip()

    ops = json.loads(raw)

    # Validate and clamp segment IDs to actual range
    valid_ids = {s.id for s in segments}
    for op in ops:
        op["segment_ids"] = [sid for sid in op["segment_ids"] if sid in valid_ids]

    return [op for op in ops if op["segment_ids"]]
