"""
Stage 6 — Translate: every line into the target language, natural and timed.

First, sung lines are marked so they are kept or re-sung, not spoken
(music/song_mode.py).

Default, "indic" (TRANSLATOR=indic):
  1. Bodhan indic-translate translates each line (text/line_by_line.py).
     Lines it fails on go to the LLM instead.
  2. The dubbing director (text/director.py): one LLM pass over the whole
     conversation fixes what line-by-line translation cannot see: the
     speaker's gender in verb endings, spoken rather than written register,
     consistent forms of address, and timing.
Option, "llm" (TRANSLATOR=llm): the LLM translates the whole conversation in
one go, with each line's speaker, gender and time budget (text/dialogue.py).

Then, either way, lines too long to speak in their time are shortened
(text/length_fit.py), except lines the director just wrote to fit.

Edit requests ("make line 5 funnier") are handled here too: only the edited
lines are translated again, with the request as an instruction.
"""

import json
import os

from ..cache import get_cache_dir
from ..music.song_mode import mark_songs
from ..state import PipelineState, Segment
from ..text.dialogue import _REGISTER, translate_dialogue
from ..text.director import direct_dub, video_context
from ..text.languages import _MIX_TEXT, LANG_MIX, latin_allowed
from ..text.length_fit import shorten_overlong
from ..text.line_by_line import needs_style_llm, translate_style_edit, translate_with_bodhan
from ..text.llm import TRANSLATE_MODEL

_TRANSLATOR = os.environ.get("TRANSLATOR", "indic").lower()


def translate_node(state: PipelineState) -> dict:
    segments = [Segment.from_dict(s) for s in state["segments"]]
    edit_ops = state.get("edit_ops", [])

    edit_map: dict[int, str] = {}
    for op in edit_ops:
        for sid in op["segment_ids"]:
            edit_map[sid] = op["instruction"]

    for seg in segments:
        if seg.id in edit_map:
            seg.edit_instruction = edit_map[seg.id]
            seg.dubbed_text = None
            # Its audio is redone too (later stages only voice lines without it).
            seg.dubbed_audio = seg.aligned_audio = seg.lip_synced_clip = None
            seg.sync_clips = []

    song_note = mark_songs(segments, state)
    pending = [s for s in segments if s.dubbed_text is None and not s.keep_original]
    if not pending:
        return {"log": state.get("log", []) + ["✓ Translation: all segments cached"]}

    seg_map = {s.id: s for s in segments}
    target_lang = state["target_lang"]
    # The user's names & words list, given to every translator below.
    from ..text import glossary

    words = glossary.block(glossary.load(state["video_id"]))

    def with_words(instruction):
        return "\n".join(x for x in (instruction, words) if x) or None

    # Lines an edit put in another language: translated on their own, and
    # left out of the passes below, which write in the dub's language.
    other_lang = {s.id for s in segments if s.lang_override}
    other_note = _translate_other_languages([s for s in pending if s.id in other_lang], seg_map)
    pending = [s for s in pending if s.id not in other_lang]
    # The lines this run translates. The passes below see the whole dialogue
    # but change only these, so an edit to one line leaves the rest alone.
    redo = {s.id for s in pending}
    translator = (state.get("translator") or _TRANSLATOR).lower()
    use_llm = translator == "llm" and bool(os.environ.get("OPENROUTER_API_KEY"))
    mix = (state.get("lang_mix") or LANG_MIX).lower()
    if mix not in _MIX_TEXT:
        mix = "standard"

    directed_ids: set[int] = set()
    if not pending:
        notes = []
    elif use_llm:
        # One group per instruction: plain lines together, each edit
        # instruction ("make it funny") as its own group.
        groups: dict[str | None, list[Segment]] = {}
        for seg in pending:
            groups.setdefault(seg.edit_instruction, []).append(seg)
        failed: list[Segment] = []
        for instruction, group in groups.items():
            failed += translate_dialogue(
                group,
                segments,
                target_lang,
                with_words(instruction),
                state.get("domain"),
                mix=mix,
                latin=latin_allowed(state),
            )
        if failed:
            translate_with_bodhan(failed, target_lang, seg_map)
        notes = [
            f"✓ Translated {len(pending)} lines as one dialogue with {TRANSLATE_MODEL} "
            f"(register: {_REGISTER}, language mix: {mix}"
            f"{', English kept in English spelling' if latin_allowed(state) else ''}, "
            "speaker gender + timing aware)"
            + (f"; {len(failed)} fell back to Bodhan" if failed else "")
        ]
    else:
        bodhan_group: list[Segment] = []
        openrouter_groups: dict[str, list[Segment]] = {}
        for seg in pending:
            instruction = seg.edit_instruction or "default"
            if instruction == "default" or not needs_style_llm(instruction):
                bodhan_group.append(seg)
            else:
                openrouter_groups.setdefault(instruction, []).append(seg)
        failed: list[Segment] = []
        if bodhan_group:
            failed = translate_with_bodhan(bodhan_group, target_lang, seg_map)
        for instruction, group in openrouter_groups.items():
            translate_style_edit(group, target_lang, with_words(instruction), seg_map)
        # Indic first: the LLM only for lines Bodhan could not translate.
        if failed and os.environ.get("OPENROUTER_API_KEY"):
            translate_dialogue(
                failed,
                segments,
                target_lang,
                with_words(None),
                state.get("domain"),
                mix=mix,
                latin=latin_allowed(state),
            )
        why = "Indic pipeline" if translator != "llm" else "no OPENROUTER_API_KEY"
        notes = [
            f"✓ Translated {len(pending)} lines line-by-line with Bodhan ({why})"
            + (f"; {len(failed)} failed there and went to {TRANSLATE_MODEL}" if failed else "")
        ]
        # Bodhan sees one line at a time: no speaker, gender or conversation.
        before = {s.id: s.dubbed_text for s in segments}
        context = "\n".join(x for x in (video_context(state), words) if x)
        # "concise": the earlier method — Bodhan's lines kept as they are and
        # shortened to fit below, without the director's rewrite (shorter,
        # plainer lines; gender agreement is not checked across lines).
        directed, split = (0, 0)
        if translator != "concise":
            directed, split = direct_dub(segments, target_lang, context, only=redo)
        directed_ids = {s.id for s in segments if before.get(s.id) != s.dubbed_text}
        if directed:
            notes.append(
                f"✓ Dubbing director ({TRANSLATE_MODEL}): {directed} line(s) made "
                "natural — gender agreement, spoken register, consistent address, "
                "timing" + (f"; {split} line(s) held two speakers and were split" if split else "")
            )

    # Shorten anything that cannot be spoken in its slot. Doing this before
    # synthesis is what keeps the dub from sounding hurried — time-stretching
    # can then stay within a range the ear does not notice.
    # Lines the director just rewrote already fit their budget, written with
    # the whole conversation in view; shortening them again without that
    # context would undo its work. So only the other lines are shortened
    # here. The synthesize stage still catches real overruns from audio.
    untouched = {s.id for s in segments} - redo
    shortened = shorten_overlong(
        segments, state["target_lang"], skip=directed_ids | other_lang | untouched
    )
    if shortened:
        notes.append(f"✓ Shortened {shortened} over-long segment(s) to fit their time slots")

    cache_dir = get_cache_dir(state["video_id"])
    with open(cache_dir / "translations.json", "w", encoding="utf-8") as f:
        json.dump([s.to_dict() for s in segments], f, ensure_ascii=False, indent=2)

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", [])
        + ([song_note] if song_note else [])
        + notes
        + ([other_note] if other_note else []),
    }


def _translate_other_languages(lines: list[Segment], seg_map: dict[int, Segment]) -> str:
    """Translate lines an edit set to another language, grouped by language."""
    from ..text.languages import LANG_NAMES

    by_lang: dict[str, list[Segment]] = {}
    for s in lines:
        by_lang.setdefault(s.lang_override, []).append(s)
    for lang, group in by_lang.items():
        plain = [s for s in group if not s.edit_instruction]
        if plain:
            translate_with_bodhan(plain, lang, seg_map)
        for s in group:
            if s.edit_instruction:
                translate_style_edit([s], lang, s.edit_instruction, seg_map)
    return (
        "✓ Lines in another language: "
        + ", ".join(f"{LANG_NAMES.get(k, k)} ×{len(v)}" for k, v in by_lang.items())
        if by_lang
        else ""
    )
