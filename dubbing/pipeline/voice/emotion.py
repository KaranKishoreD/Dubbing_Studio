"""
Emotion per line: which lines are said with real feeling, so the voice can
carry it, and which are plain talk, so it is not overacted.

One LLM pass reads the whole dialogue (what is said, to whom, in what kind
of video) together with how each line was measured to be delivered
(speech/prosody.py: energetic, calm, quick... against the speaker's own
baseline). Per line it returns an emotion and how strong it is. A Bodhan
style is given to a line only when the words and the delivery agree and
the emotion is strong; everything else keeps the speaker's base style.
The model also says what kind of video it is, so plain talk in an
interview or chat is not read like the news.

Results are cached against a hash of their input, so re-runs are free.
"""

import hashlib
import json
import os

from ..cache import get_cache_dir
from ..state import Segment
from ..text.llm import TRANSLATE_MODEL, openrouter_client, parse_items

ENABLED = os.environ.get("EMOTION_STYLES", "1") == "1"

# Emotion → Bodhan style.
STYLE_OF = {
    "excited": "happy",
    "happy": "happy",
    "surprised": "surprise",
    "sad": "sad",
    "angry": "anger",
    "afraid": "fear",
    "disgusted": "disgust",
}
_MIN_STRENGTH = 0.4  # below this a line keeps the speaker's base style
_SURE = 0.8  # strong enough to use even if the delivery measured plain
_WIDE = 1.3  # pitch range this much wider than the speaker's usual = lively
_MAX_SHARE = 0.4  # at most this share of a speaker's lines get an emotion style
_LIVELY = {"energetic", "announcing", "urgent"}  # measured deliveries that carry feeling

_SYSTEM = """\
You direct the voice acting of a dubbed video. For every line, say how the
ORIGINAL speaker actually feels while saying it, so the dub voice can
carry the same feeling — and no more.

You get the video title, then per line: speaker, the original words, and
the measured delivery compared with that speaker's own usual way of
talking (neutral, calm, energetic, announcing, quick, slow...).

Rules:
- Most lines in most videos are plain: "neutral". Explaining, answering,
  asking a normal question, giving numbers — neutral, unless the words AND
  the delivery clearly show more.
- Only mark a feeling the line really has: excited (proud, thrilled,
  enthusiastic), happy (warm, amused), surprised (astonished, "oh my god"),
  sad, angry, afraid, disgusted.
- strength 0.0-1.0: how unmistakable the feeling is. 0.9 = anyone would
  hear it; 0.6 = a hint.
- Also say what kind of video this is: "conversation" (interview, chat,
  show, vlog, pitch), "news", "lecture" or "story".

Return ONLY JSON:
{"genre": "conversation"|"news"|"lecture"|"story",
 "lines": [{"id": <int>, "emotion": "neutral"|"excited"|"happy"|"surprised"|"sad"|"angry"|"afraid"|"disgusted", "strength": <0-1>}]}
"""


def label_lines(segments: list[Segment], state, only: set[int] | None = None) -> tuple[str, str]:
    """
    Set seg.emotion on dubbed lines that have none yet ("neutral" or a
    STYLE_OF key) and return (genre, log note). Lines already labelled keep
    their label, so an edit never changes how untouched lines sound.
    No-op without an OpenRouter key.
    """
    lines = [s for s in sorted(segments, key=lambda s: s.start) if s.text and not s.keep_original]
    if not ENABLED or not lines:
        return "", ""
    cache_dir = get_cache_dir(state["video_id"])
    genre_file = cache_dir / "emotion_genre.txt"
    todo = [s for s in lines if s.emotion is None and (only is None or s.id in only)]
    if not todo:
        genre = genre_file.read_text().strip() if genre_file.exists() else ""
        return genre, ""
    if not os.environ.get("OPENROUTER_API_KEY"):
        return "", ""
    payload = [
        {
            "id": s.id,
            "speaker": f"Speaker {(s.speaker or 0) + 1}",
            "original": s.text,
            "delivery": ", ".join(x for x in (s.delivery, s.pace) if x) or "neutral",
        }
        for s in lines
    ]
    from ..text.director import video_context

    title = video_context(state)
    key = hashlib.sha1(json.dumps([title, payload], ensure_ascii=False).encode()).hexdigest()[:16]
    path = cache_dir / f"emotions_{key}.json"
    if path.exists():
        data, how = json.loads(path.read_text(encoding="utf-8")), "cached"
    else:
        try:
            resp = openrouter_client().chat.completions.create(
                model=TRANSLATE_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM},
                    {
                        "role": "user",
                        "content": f"VIDEO: {title}\n\n" + json.dumps(payload, ensure_ascii=False),
                    },
                ],
                temperature=0.1,
                timeout=120,
            )
            data = _parse(resp.choices[0].message.content or "")
        except Exception as exc:
            print(f"  ⚠ emotion labels skipped: {exc}")
            return "", ""
        path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        how = TRANSLATE_MODEL

    raw = {int(it["id"]): it for it in data.get("lines", []) if isinstance(it, dict) and "id" in it}
    chosen = _choose(lines, raw, _wide_lines(lines, state.get("vocals_path")))
    for s in todo:
        s.emotion = chosen.get(s.id, "neutral")
    genre = str(data.get("genre") or "")
    if genre:
        genre_file.write_text(genre)
    used = [s for s in todo if s.emotion != "neutral"]
    tally: dict[str, int] = {}
    for s in used:
        tally[s.emotion] = tally.get(s.emotion, 0) + 1
    note = f"✓ Emotion ({how}): {genre or 'unknown'} video; " + (
        ", ".join(f"{k}×{v}" for k, v in sorted(tally.items())) + f", {len(todo) - len(used)} plain"
        if used
        else f"all {len(todo)} lines plain"
    )
    return genre, note


def _wide_lines(lines: list[Segment], vocals_path: str | None) -> set[int]:
    """Lines whose pitch moves clearly more than their speaker's usually does
    (5-95% range, original vocals): heard as lively even when the overall
    delivery measured neutral."""
    if not vocals_path or not os.path.exists(vocals_path):
        return set()
    import librosa
    import numpy as np

    y, sr = librosa.load(vocals_path, sr=16000, mono=True)
    span: dict[int, float] = {}
    for s in lines:
        x = y[int(s.start * sr) : int(s.end * sr)]
        if len(x) < sr // 2:
            continue
        f0 = librosa.yin(x, fmin=60, fmax=400, sr=sr)
        rms = librosa.feature.rms(y=x, frame_length=2048, hop_length=512)[0]
        n = min(len(f0), len(rms))
        f0 = f0[:n][rms[:n] > rms.max() * 0.1]  # voiced, audible frames only
        if len(f0) >= 10:
            st = 12 * np.log2(f0 / np.median(f0))
            span[s.id] = float(np.percentile(st, 95) - np.percentile(st, 5))
    wide = set()
    for spk in {s.speaker or 0 for s in lines}:
        mine = [span[s.id] for s in lines if (s.speaker or 0) == spk and s.id in span]
        if len(mine) < 3:
            continue
        usual = float(np.median(mine))
        wide |= {
            s.id for s in lines if (s.speaker or 0) == spk and span.get(s.id, 0) >= usual * _WIDE
        }
    return wide


def _choose(lines: list[Segment], raw: dict[int, dict], wide: set[int]) -> dict[int, str]:
    """Keep only clear feelings the delivery supports, a few per speaker."""
    picks: dict[int, list[tuple[float, int, str]]] = {}
    for s in lines:
        it = raw.get(s.id) or {}
        emo = str(it.get("emotion") or "neutral").lower()
        try:
            strength = float(it.get("strength") or 0)
        except (TypeError, ValueError):
            strength = 0.0
        if emo not in STYLE_OF or strength < _MIN_STRENGTH:
            continue
        lively = (s.delivery or "") in _LIVELY or s.id in wide
        # Excitement and anger are heard in the voice: the delivery must
        # show it unless the words leave no doubt.
        if emo in ("excited", "angry") and not lively and strength < _SURE:
            continue
        picks.setdefault(s.speaker or 0, []).append((strength, s.id, emo))
    out: dict[int, str] = {}
    for spk, cands in picks.items():
        n_lines = sum(1 for s in lines if (s.speaker or 0) == spk)
        keep = max(1, int(n_lines * _MAX_SHARE))
        for strength, sid, emo in sorted(cands, reverse=True)[:keep]:
            out[sid] = emo
    return out


def _parse(text: str) -> dict:
    text = text.strip()
    a = text.find("{")
    if a != -1:
        try:
            return json.JSONDecoder().raw_decode(text[a:])[0]
        except json.JSONDecodeError:
            pass
    items = parse_items(text)
    return {"lines": items if isinstance(items, list) else []}
