"""Timing rules: how much room a dubbed line gets and how its speed is smoothed."""

from pipeline.state import Segment
from pipeline.steps import align


def seg(i, start, end, speaker=0):
    s = Segment(id=i, start=start, end=end, text="x", lang="en")
    s.speaker = speaker
    return s


def test_slot_stops_before_the_next_line():
    lines = [seg(0, 0.0, 2.0), seg(1, 2.5, 4.0)]
    assert align.slot_seconds(lines, 0, 10.0) == 2.5 - align.GAP_GUARD


def test_slot_borrows_at_most_the_limit_of_silence():
    lines = [seg(0, 0.0, 2.0), seg(1, 30.0, 32.0)]
    assert align.slot_seconds(lines, 0, 60.0) == 2.0 + align.GAP_BORROW_MAX


def test_slot_is_never_shorter_than_the_original_line():
    lines = [seg(0, 0.0, 2.0), seg(1, 2.0, 4.0)]
    assert align.slot_seconds(lines, 0, 10.0) == 2.0


def test_smoothing_speeds_up_the_slower_neighbour_only():
    import numpy as np

    fast, slow = seg(0, 0.0, 2.0), seg(1, 2.5, 4.5)
    audio = np.zeros(16000 * 2, dtype=np.float32)
    plan = [[fast, audio, 16000, 1.4, 1.4, ""], [slow, audio, 16000, 1.0, 1.0, ""]]
    changed = align._smooth_rates(plan)
    assert changed == 1
    assert plan[0][3] == 1.4  # the fast line is never slowed
    assert plan[1][3] > 1.0  # the slow one eased toward it
    # ...but never so far that the line stops filling its own speech.
    assert plan[1][3] <= 2.0 / (align._SMOOTH_MIN_FILL * 2.0) + 1e-6
