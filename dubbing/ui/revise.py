"""
The Revise tab: turn a request like "make line 5 sadder" or "speaker 2 is a
woman" into a plan of edits (pipeline/edits/edit_planner.py), show it, and
describe what changed after the re-run.
"""

from pipeline.edits.edit_planner import plan_edits
from pipeline.state import Segment
from pipeline.text.languages import LANG_NAMES
from ui.options import CATALOGUE, STYLES, SUPPORTED_LANGS
from ui.views import fmt_time

_ACTION_LABEL = {
    "retranslate": "Re-translate",
    "set_text": "New wording",
    "recast": "Change voice",
    "set_gender": "Fix gender",
    "reassign_speaker": "Fix speaker",
    "set_style": "Delivery",
    "keep_original": "Keep original",
    "dub_again": "Dub again",
}


def plan_changes(message: str, segs: list, lang: str):
    """Revise tab: message → plan (state), plan table rows, status."""
    if not segs:
        return {}, [], "⚠ Dub a video first."
    if not message.strip():
        return {}, [], "⚠ Describe the change you want."
    code = SUPPORTED_LANGS.get(lang, "hi")
    try:
        plan = plan_edits(
            message.strip(),
            [Segment.from_dict(d) for d in segs],
            LANG_NAMES.get(code, code),
            CATALOGUE,
            STYLES,
        )
    except Exception as e:
        print(f"✗ Edit planning failed: {type(e).__name__}: {e}", flush=True)
        return {}, [], f"✗ Could not plan that: {e}"
    print(f"=== Edit plan for: {message.strip()}\n{plan['raw']}", flush=True)
    rows, unhandled = [], []
    for r in plan["requests"]:
        for act in r["actions"]:
            rows.append(
                [
                    r["request"],
                    _ACTION_LABEL.get(act["type"], act["type"]),
                    _action_target(act),
                    _action_detail(act),
                ]
            )
        if not r["actions"]:
            unhandled.append(
                f"“{r['request']}” — {r['note'] or 'no action fits'}"
                + (f" ({'; '.join(r['rejected'])})" if r["rejected"] else "")
            )
    n = len(plan["requests"])
    status = (
        f"✓ Understood **{n} request{'s' if n != 1 else ''}** → "
        f"{len(rows)} change{'s' if len(rows) != 1 else ''}. Review, then **Apply**."
    )
    if unhandled:
        status += "\n\n⚠ **Not handled:**\n" + "\n".join(f"- {u}" for u in unhandled)
    return plan, rows, status


def _action_target(a: dict) -> str:
    if "lines" in a:
        return ", ".join(f"S{i}" for i in a["lines"])
    if "line" in a:
        return f"S{a['line']}"
    return f"Speaker {a.get('speaker')}"


def _action_detail(a: dict) -> str:
    t = a["type"]
    return {
        "retranslate": a.get("instruction", ""),
        "set_text": a.get("text", ""),
        "recast": a.get("voice", ""),
        "set_gender": a.get("gender", ""),
        "reassign_speaker": f"→ Speaker {a.get('speaker')}",
        "set_style": a.get("style", ""),
    }.get(t, "")


def render_diff(before: list[dict], after: list[dict]) -> str:
    """What actually changed, line by line."""
    old = {d["id"]: Segment.from_dict(d) for d in before}
    rows = []
    for d in sorted(after, key=lambda d: d["start"]):
        s, o = Segment.from_dict(d), old.get(d["id"])
        if not o:
            continue
        bits = []
        if (s.dubbed_text or "") != (o.dubbed_text or ""):
            bits.append(f"“{o.dubbed_text or '—'}” → “{s.dubbed_text or '—'}”")
        if s.voice != o.voice and not s.keep_original:
            bits.append(f"voice {o.voice} → {s.voice}")
        if s.speaker != o.speaker:
            bits.append(f"Speaker {(o.speaker or 0) + 1} → {(s.speaker or 0) + 1}")
        if s.style_override != o.style_override:
            bits.append(f"delivery → {s.style_override}")
        if s.keep_original != o.keep_original:
            bits.append("now undubbed" if s.keep_original else "dubbed again")
        if bits:
            rows.append(f"- **S{s.id}** ({fmt_time(s.start)}): " + "; ".join(bits))
    return "\n".join(rows) if rows else "_No visible changes._"
