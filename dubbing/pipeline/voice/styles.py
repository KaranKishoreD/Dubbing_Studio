"""
Speaking style (Bodhan "style" parameter) for each line.

In Auto mode each speaker gets ONE style matched to how fast they talk
(speech/pace.py), so a voice never changes style mid-conversation; only
clearly energetic lines get "happy". An edit ("make line 5 sad") or a style
chosen in the UI overrides this.
"""

import os

from ..state import Segment
from .catalogue import is_sarvam
from .tts import speakable, tts_bytes

# ── Speaking styles ────────────────────────────────────────────────────────────
BODHAN_STYLES: dict[str, str] = {
    # Per-line: each line's measured delivery (prosody.py) picks its style,
    # on top of a neutral "educational lecture" base.
    "Auto (match original delivery)": "auto",
    "Educational / Lecture": "educational lecture",
    "TV News / Formal": "TV style news",
    "AIR News": "AIR style news",
    "General News": "news",
    "Advertisement": "advertisements",
    "Customer Care": "Customer Care",
    "Happy / Energetic": "happy",
    "Sad / Emotional": "sad",
    "Angry": "anger",
    "Surprised / Dramatic": "surprise",
    "Fear": "fear",
    "Disgusted": "disgust",
    "Storytelling / Audiobook": "single person narration audiobook",
    "Children's Story": "children's stories",
}


DEFAULT_STYLE_LABEL = "Auto (match original delivery)"

DEFAULT_STYLE = "auto"


AUTO_BASE_STYLE = os.environ.get("AUTO_BASE_STYLE", "educational lecture")

# Auto mode: the base style of each speaker, picked from their tempo
# (pace.py) at the start of synthesize_node.
_SPEAKER_STYLE: dict[int, str] = {}


def line_style(seg: Segment, global_style: str) -> str:
    """
    Global style, or in auto mode the speaker's own style (matched to their
    tempo). Only lines with a clear feeling get a style of their own
    (voice/emotion.py): switching styles line by line made one person's
    voice change mid-conversation.
    """
    from .emotion import STYLE_OF

    if seg.style_override:  # set by an edit ("make line 5 sad")
        return seg.style_override
    if global_style != "auto":
        return global_style
    base = _SPEAKER_STYLE.get(seg.speaker if seg.speaker is not None else 0, AUTO_BASE_STYLE)
    if seg.emotion is not None:  # labelled: the emotion pass decided
        return STYLE_OF.get(seg.emotion, base)
    return "happy" if seg.delivery == "energetic" else base


# News styles read plain talk like a bulletin: not used as the base voice of
# an interview, chat or pitch.
_NEWS_STYLES = {"news", "TV style news", "AIR style news"}


def pick_speaker_styles(
    segments: list[Segment], lang_code: str, global_style: str, genre: str = ""
) -> str:
    """Auto mode: each speaker's style = the one closest to their tempo
    (news styles only for news)."""
    from ..speech import pace

    _SPEAKER_STYLE.clear()
    if global_style != "auto":
        return ""
    factors = pace.speaker_factors(segments)
    picked = []
    for spk, factor in sorted(factors.items()):
        mine = [s for s in segments if (s.speaker or 0) == spk and s.dubbed_text and s.voice]
        if (
            not mine
            or is_sarvam(mine[0].voice)
            or str(mine[0].voice).startswith(("kokoro:", "piper:"))
        ):
            continue
        sample = max((s.dubbed_text for s in mine), key=len)[:160]
        try:
            rates = pace.measure_styles(
                mine[0].voice,
                lang_code,
                speakable(sample),
                lambda t, l, v, st: tts_bytes(t, l, v, st),
            )
        except Exception as exc:
            print(f"  ⚠ style speeds not measured: {exc}")
            rates = {}
        if genre and genre != "news":
            rates = {k: v for k, v in rates.items() if k not in _NEWS_STYLES}
        style = pace.pick_style(factor, rates, AUTO_BASE_STYLE)
        _SPEAKER_STYLE[spk] = style
        picked.append(f"S{spk + 1} {factor:.2f}× → {style}")
    return ("✓ Pace: " + ", ".join(picked)) if picked else ""


def resolve_style(edit_instruction: str | None, global_style: str) -> str:
    if not edit_instruction:
        return global_style
    low = edit_instruction.lower()
    if any(k in low for k in ("news", "formal", "serious")):
        return "TV style news"
    if any(k in low for k in ("happy", "energetic", "enthusiastic", "funny", "humor")):
        return "happy"
    if any(k in low for k in ("sad", "emotional", "grief")):
        return "sad"
    if any(k in low for k in ("angry", "anger", "aggressive")):
        return "anger"
    if any(k in low for k in ("surprised", "surprise", "dramatic", "shock")):
        return "surprise"
    if any(k in low for k in ("lecture", "explain", "educational", "slow", "clear")):
        return "educational lecture"
    if any(k in low for k in ("story", "narration", "audiobook")):
        return "single person narration audiobook"
    return global_style
