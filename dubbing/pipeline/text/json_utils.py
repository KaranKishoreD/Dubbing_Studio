"""
Reading JSON out of LLM replies.

We ask the models for "JSON only", but replies still arrive wrapped in code
fences, with a sentence before or after, or with small syntax slips. These
helpers recover the JSON array instead of failing the whole step.
"""

import json
import re


def extract_json_array(raw: str) -> list:
    """
    Return the JSON array inside a model reply.

    Tries, in order: the whole reply (fences removed), then the outermost
    [...] found in the text. Raises ValueError if neither parses.
    """
    text = raw.strip()

    # Drop ``` fences, with or without a language tag.
    if text.startswith("```"):
        lines = [line for line in text.split("\n") if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()

    text = _repair(text)

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else [parsed]
    except json.JSONDecodeError:
        pass

    # Find the outermost [...] by counting brackets, ignoring any inside strings.
    start = text.find("[")
    if start != -1:
        depth, in_string, escaped = 0, False, False
        for i in range(start, len(text)):
            ch = text[i]
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = not in_string
            elif not in_string:
                if ch == "[":
                    depth += 1
                elif ch == "]":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(_repair(text[start : i + 1]))
                        except json.JSONDecodeError:
                            break

    raise ValueError(f"Model did not return parseable JSON. First 300 chars:\n{raw[:300]}")


def _repair(text: str) -> str:
    """
    Fix two slips models make often.

    1. Segment ids echoed as labels: the transcript shows lines as [S0], [S1]…
       so a reply may contain [S1, S2] (not valid JSON) — turn them into 1, 2.
    2. A trailing comma before a closing bracket or brace.
    """
    text = re.sub(r'(?<=[\[,])(\s*)"?[Ss](\d+)"?(\s*)(?=[,\]])', r"\1\2\3", text)
    text = re.sub(r",(\s*[\]}])", r"\1", text)
    return text
