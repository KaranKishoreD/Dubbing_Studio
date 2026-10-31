"""Voice casting: one voice per speaker, never shared while others are free."""

from pipeline.state import Segment
from pipeline.voice.casting import assign_voices


def line(i, speaker, gender, f0):
    s = Segment(id=i, start=i * 2.0, end=i * 2.0 + 1.5, text="x", lang="en")
    s.speaker, s.gender, s.f0 = speaker, gender, f0
    return s


def test_each_speaker_gets_one_voice_of_their_own():
    lines = [
        line(0, 0, "female", 220),
        line(1, 1, "male", 120),
        line(2, 0, "female", 230),
        line(3, 2, "male", 135),
    ]
    cast = assign_voices(lines, "hi", None)
    assert set(cast) == {0, 1, 2}
    assert len(set(cast.values())) == 3


def test_a_voice_chosen_in_the_ui_wins():
    lines = [line(0, 0, "female", 220), line(1, 1, "male", 120)]
    cast = assign_voices(lines, "hi", None, voice_map={"1": "Amit"})
    assert cast[1] == "Amit"


def test_override_gives_everyone_the_same_voice():
    lines = [line(0, 0, "female", 220), line(1, 1, "male", 120)]
    assert set(assign_voices(lines, "hi", "Kavya").values()) == {"Kavya"}
