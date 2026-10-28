"""
Refit: fix line lengths using the REAL spoken audio.

Translation only estimates how long a line will take to say. After speech,
each clip is measured; lines that would need too much speed-up are
rewritten shorter, lines that end well before the speaker stops are
lengthened, and both are spoken again (at most _REFIT_ROUNDS times).
"""

import os

from ..state import Segment

_REFIT_ROUNDS = int(os.environ.get("REFIT_ROUNDS", "2"))

# Speed-up accepted before a line is rewritten shorter (see refit_lengths).
_REWRITE_ABOVE = float(os.environ.get("REWRITE_ABOVE_STRETCH", "1.15"))

_REWRITE_SHORT = float(os.environ.get("REWRITE_SHORT_ABOVE_STRETCH", "1.3"))


# A dub shorter than this share of the original speech is lengthened (only
# lines of at least _GROW_MIN_SECS), so the lips do not move in silence.
_GROW_BELOW = float(os.environ.get("GROW_BELOW_FILL", "0.8"))

_GROW_MIN_SECS = float(os.environ.get("GROW_MIN_SECS", "1.5"))


def refit_lengths(segments: list[Segment], audio_path: str | None, run, lang_code: str) -> str:
    """
    Rewrite-and-re-voice any line whose real audio would need more than
    MAX_STRETCH speed-up, up to _REFIT_ROUNDS times. Shortening the words is
    what keeps the voice at a natural pace; speeding it up is the last resort.
    """
    from ..nodes.align import MAX_STRETCH, _audio_len, slot_seconds, trimmed_duration
    from ..text.length_fit import condense_to_budget, expand_to_fill
    from ..voice import tts_rate

    ordered = sorted(segments, key=lambda s: s.start)
    total = _audio_len(audio_path)
    notes: list[str] = []
    cps_by_voice: dict[str, float] = {}

    for rnd in range(1, _REFIT_ROUNDS + 1):
        measured: list[tuple[Segment, float, float]] = []  # seg, clip secs, slot secs
        for i, seg in enumerate(ordered):
            if seg.dubbed_audio and seg.dubbed_text:
                try:
                    measured.append(
                        (seg, trimmed_duration(seg.dubbed_audio), slot_seconds(ordered, i, total))
                    )
                except Exception:
                    pass

        # Characters per second each voice actually produced, ignoring tiny
        # clips whose length is mostly ramp-up. Stored for the next translation.
        rates: dict[str, list[float]] = {}
        for s, d, _ in measured:
            if d >= 0.4 and len(s.dubbed_text) >= 4:
                rates.setdefault(s.voice or "?", []).append(len(s.dubbed_text) / d)
        if rnd == 1:
            tts_rate.record(lang_code, rates)
        flat = sorted(r for rs in rates.values() for r in rs)
        if len(flat) < 2:
            return ""
        cps_all = flat[len(flat) // 2]
        cps_by_voice = {
            v: sorted(rs)[len(rs) // 2] if len(rs) >= 2 else cps_all for v, rs in rates.items()
        }

        jobs, over = [], []
        for seg, dur, slot in measured:
            # Rewriting to save a few percent cost meaning and tone ("still"
            # dropped, a casual goodbye turned ceremonial); a slightly
            # faster line is better. A very short reply can only get terser,
            # so it is sped up more before anyone rewrites it.
            limit = max(MAX_STRETCH, _REWRITE_SHORT if slot < 1.5 else _REWRITE_ABOVE)
            if dur / slot > limit:
                cps = cps_by_voice.get(seg.voice or "?", cps_all)
                # Aim under the cap so the re-voiced clip fits without stretching.
                budget = int(slot * cps * (0.95 if rnd == 1 else 0.85))
                jobs.append((seg.id, seg.dubbed_text, max(budget, 4)))
                over.append(seg)
        # Lines that end well before the speaker stops: lips keep moving in
        # silence. Only clearly short ones, on lines long enough to matter.
        grow, under = [], []
        for seg, dur, slot in measured:
            if (
                seg not in over
                and seg.duration >= _GROW_MIN_SECS
                and dur < _GROW_BELOW * seg.duration
            ):
                cps = cps_by_voice.get(seg.voice or "?", cps_all)
                grow.append(
                    {
                        "id": seg.id,
                        "original": seg.text,
                        "dub": seg.dubbed_text,
                        "min_chars": int(0.9 * seg.duration * cps),
                        "max_chars": int(min(slot, 1.05 * seg.duration) * cps),
                    }
                )
                under.append(seg)
        if not jobs and not grow:
            notes.append(
                f"✓ All clips fit their slots at natural pace (measured {cps_all:.1f} chars/s)"
            )
            break

        new = condense_to_budget(jobs) if jobs else {}
        longer = expand_to_fill(grow, lang_code) if grow else {}
        redo = []
        for seg in over:
            if seg.id in new:
                seg.dubbed_text = new[seg.id]
                redo.append(seg)
        for seg in under:
            if seg.id in longer:
                seg.dubbed_text = longer[seg.id]
                redo.append(seg)
        if not redo:
            if jobs:
                why = (
                    "no OPENROUTER_API_KEY"
                    if not os.environ.get("OPENROUTER_API_KEY")
                    else "already as short as they can naturally be"
                )
                notes.append(
                    f"⚠ {len(jobs)} clip(s) still too long for their slot ({why}) "
                    "— they will be sped up"
                )
            if grow:
                notes.append(
                    f"⚠ {len(grow)} clip(s) end early — no natural way to lengthen "
                    "them; left as they are"
                )
            break
        run(redo)
        notes.append(
            f"✓ Round {rnd}: re-voiced {sum(s in over for s in redo)} over-long and "
            f"{sum(s in under for s in redo)} too-short line(s) "
            f"(measured {cps_all:.1f} chars/s)"
        )
    return "\n".join(notes)
