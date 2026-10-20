import json
import os
from pathlib import Path
from ..state import PipelineState, Segment
from ..cache import get_cache_dir


def transcribe_node(state: PipelineState) -> dict:
    video_id = state["video_id"]
    cache_dir = get_cache_dir(video_id)
    transcript_path = cache_dir / "transcript.json"

    # Raw ASR segments are returned; diarize_node splits them at speaker
    # changes and only then merges them into sentences.
    if transcript_path.exists():
        with open(transcript_path, encoding="utf-8") as f:
            raw = json.load(f)
        return {
            "segments": raw,
            "log": state.get("log", []) + [
                f"✓ Transcript loaded from cache ({len(raw)} ASR segments)"
            ],
        }

    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # Override with WHISPER_MODEL in .env. VRAM at float16: small ~1GB,
    # medium ~2.5GB, large-v3 ~4.7GB (needs >4GB card; use int8 to halve it).
    default_size = "small" if device == "cuda" else "tiny"
    model_size = os.environ.get("WHISPER_MODEL", default_size)
    model = WhisperModel(model_size, device=device, compute_type=compute_type)

    raw_segments, info = model.transcribe(
        state["vocals_path"],
        beam_size=5,
        language=None,          # auto-detect per segment
        word_timestamps=True,   # word-level timestamps for precise segment snapping
        vad_filter=True,        # skip silence (saves cost + avoids empty segments)
        vad_parameters={"min_silence_duration_ms": 500},
    )

    detected_lang = info.language

    segments = []
    for i, seg in enumerate(raw_segments):
        words = [
            {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
            for w in (seg.words or [])
        ]
        s = Segment(
            id=i,
            start=round(seg.start, 3),
            end=round(seg.end, 3),
            text=seg.text.strip(),
            lang=detected_lang,
            words=words,
        )
        segments.append(s.to_dict())

    # Raw ASR output is cached; sentence merging is re-applied on every load
    # so its rules can change without re-transcribing.
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return {
        "segments": segments,
        "log": state.get("log", []) + [
            f"✓ Transcribed {len(segments)} ASR segments "
            f"({model_size} on {device}, detected: {detected_lang})"
        ],
    }


# ── Sentence merging ──────────────────────────────────────────────────────────
# Whisper splits on acoustic pauses, not grammar, so one sentence often arrives
# as two or three segments. Translating each piece separately produces broken
# output — badly so for verb-final languages like Tamil, Hindi and Kannada,
# where the verb of the English first half has nowhere to go. A sub-second
# tail ("of nation.") also gets its own TTS call and sounds like a stray word.

_TERMINAL = (".", "?", "!", "।", "॥", "…")
_MERGE_GAP = float(os.environ.get("MERGE_GAP_SECS", "0.6"))
_MAX_SENTENCE = float(os.environ.get("MAX_SENTENCE_SECS", "15"))
_MIN_FRAGMENT = float(os.environ.get("MIN_FRAGMENT_SECS", "1.2"))


def merge_into_sentences(raw: list[dict]) -> list[dict]:
    """
    Join pieces of the same sentence. Runs AFTER diarization and never joins
    two different speakers — in fast conversation a 0.6 s "Hey, Monday." is
    a whole turn, not a fragment, and gluing it to the next line gives one
    person's words the other person's voice.
    """
    segs = sorted((Segment.from_dict(s) for s in raw), key=lambda s: s.start)
    if not segs:
        return []

    def join(a: Segment, b: Segment) -> Segment:
        return Segment(
            id=a.id, start=a.start, end=b.end,
            text=f"{a.text.rstrip()} {b.text.lstrip()}".strip(),
            lang=a.lang, words=(a.words or []) + (b.words or []),
            speaker=a.speaker,
        )

    def same_speaker(a: Segment, b: Segment) -> bool:
        return a.speaker == b.speaker

    # Pass 1: glue pieces of one sentence back together — same speaker, no
    # closing punctuation on the first piece, short pause between them.
    out: list[Segment] = [segs[0]]
    for s in segs[1:]:
        cur = out[-1]
        gap = s.start - cur.end
        if (same_speaker(cur, s)
                and not cur.text.rstrip().endswith(_TERMINAL)
                and gap <= _MERGE_GAP
                and s.end - cur.start <= _MAX_SENTENCE):
            out[-1] = join(cur, s)
        else:
            out.append(s)

    # Pass 2: absorb leftover fragments, but only into a neighbour that is the
    # same speaker. A short line from a different speaker stays its own line.
    i = 0
    while i < len(out) and len(out) > 1:
        s = out[i]
        if s.duration >= _MIN_FRAGMENT:
            i += 1
            continue
        prev = out[i - 1] if i > 0 else None
        nxt = out[i + 1] if i + 1 < len(out) else None
        prev_gap = s.start - prev.end if prev and same_speaker(prev, s) else float("inf")
        next_gap = nxt.start - s.end if nxt and same_speaker(nxt, s) else float("inf")
        if prev_gap <= next_gap and prev_gap <= 1.0 and s.end - prev.start <= _MAX_SENTENCE + 3:
            out[i - 1] = join(prev, s)
            del out[i]
        elif next_gap <= 1.0 and nxt.end - s.start <= _MAX_SENTENCE + 3:
            out[i] = join(s, nxt)
            del out[i + 1]
        else:
            i += 1

    for new_id, s in enumerate(out):
        s.id = new_id
    return [s.to_dict() for s in out]
