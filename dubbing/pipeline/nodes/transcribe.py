"""
Stage 3 — Transcribe: what is said, and exactly when.

  1. faster-whisper (large-v3 on a GPU) transcribes the vocals track with
     word-level timestamps. The raw result is cached in transcript.json.
  2. Gap filling: a more sensitive voice detector (Silero) looks for speech
     that Whisper skipped, such as soft voices or a voice-over on music.
     Each gap is transcribed on its own; known Whisper hallucinations
     ("thanks for watching") are dropped.

The output is raw Whisper segments. They are joined into sentences later
(merge_into_sentences, called by the diarize stage) once each word has a
speaker, so a sentence never mixes two people.
"""

import json
import os
from pathlib import Path

from ..cache import get_cache_dir
from ..state import PipelineState, Segment


def transcribe_node(state: PipelineState) -> dict:
    video_id = state["video_id"]
    cache_dir = get_cache_dir(video_id)
    transcript_path = cache_dir / "transcript.json"
    notes: list[str] = []

    # Raw ASR segments are returned; diarize_node splits them at speaker
    # changes and only then merges them into sentences.
    if transcript_path.exists():
        with open(transcript_path, encoding="utf-8") as f:
            segments = json.load(f)
        notes.append(f"✓ Transcript loaded from cache ({len(segments)} ASR segments)")
    else:
        model, model_size, device = _whisper()
        raw_segments, info = model.transcribe(
            state["vocals_path"],
            beam_size=5,
            language=None,  # auto-detect per segment
            word_timestamps=True,  # word-level timestamps for precise segment snapping
            vad_filter=True,  # skip silence (saves cost + avoids empty segments)
            vad_parameters={"min_silence_duration_ms": 500},
        )
        detected_lang = info.language
        segments = []
        for i, seg in enumerate(raw_segments):
            words = [
                {"word": w.word, "start": round(w.start, 3), "end": round(w.end, 3)}
                for w in (seg.words or [])
            ]
            segments.append(
                Segment(
                    id=i,
                    start=round(seg.start, 3),
                    end=round(seg.end, 3),
                    text=seg.text.strip(),
                    lang=detected_lang,
                    words=words,
                ).to_dict()
            )
        # Raw ASR output is cached; sentence merging is re-applied on every
        # load so its rules can change without re-transcribing.
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(segments, f, ensure_ascii=False, indent=2)
        notes.append(
            f"✓ Transcribed {len(segments)} ASR segments "
            f"({model_size} on {device}, detected: {detected_lang})"
        )

    # Speech Whisper skipped: soft voices, voice-over on music, endings.
    gaps, gap_note = _fill_gaps(state["vocals_path"], segments, cache_dir)
    if gap_note:
        notes.append(gap_note)
    segments = sorted(segments + gaps, key=lambda d: d["start"])
    for i, d in enumerate(segments):
        d["id"] = i

    return {"segments": segments, "log": state.get("log", []) + notes}


# ── Whisper ───────────────────────────────────────────────────────────────────

_model = None


def _whisper():
    global _model
    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Override with WHISPER_MODEL in .env. VRAM at float16: small ~1GB,
    # medium ~2.5GB, large-v3 ~4.7GB. Word timings drive the speaker split and
    # the lip-sync spans, so on a GPU the most accurate model is worth it.
    model_size = os.environ.get("WHISPER_MODEL", "large-v3" if device == "cuda" else "tiny")
    if _model is None:
        _model = WhisperModel(
            model_size, device=device, compute_type="float16" if device == "cuda" else "int8"
        )
    return _model, model_size, device


# ── Gap filling ───────────────────────────────────────────────────────────────
# Whisper's own VAD, and its habit of stopping early over music, drop real
# speech, such as a voice-over at the end of a promo. Nothing downstream can
# dub what was never transcribed. So a more sensitive
# VAD (Silero) is run on the vocals; speech with no Whisper words over it is
# re-transcribed on its own, without the VAD that skipped it.

_GAP_MIN = float(os.environ.get("GAP_MIN_SECS", "0.6"))
_GAP_VAD_THRESHOLD = float(os.environ.get("GAP_VAD_THRESHOLD", "0.35"))
# Classic Whisper hallucinations on music/silence.
_HALLUCINATIONS = (
    "thank you for watching",
    "thanks for watching",
    "subscribe",
    "like and subscribe",
    "please subscribe",
    "subtitles by",
    "amara.org",
    "சப்ஸ்கிரைப்",
    "நன்றி வணக்கம்",
)


def _bodhan_text(clip, lang: str | None) -> str:
    """Bodhan indic-transcribe on one gap ('' if no key or it fails)."""
    if not os.environ.get("BODHAN_TRANSCRIBE_KEY"):
        return ""
    try:
        from openai import OpenAI

        from ..speech import bodhan_asr

        client = OpenAI(
            base_url="https://api.bodhan.ai/v1",
            api_key=os.environ["BODHAN_TRANSCRIBE_KEY"],
            timeout=60,
        )
        parts, step = [], int(29 * 16000)
        for k in range(0, len(clip), step):
            if len(clip[k : k + step]) >= 16000 * 0.3:
                parts.append(bodhan_asr._call(client, clip[k : k + step], lang))
        text = " ".join(p for p in parts if p).strip()
        t = "".join(text.split())
        return "" if (len(t) > 6 and len(set(t)) <= 3) else text  # "ୄୄୄୄ" junk
    except Exception as exc:
        print(f"  ⚠ Bodhan ASR failed on a skipped-speech gap: {exc}")
        return ""


def _plausible_gap(g, known: list[str]) -> bool:
    import difflib

    text = g.text.strip()
    if not text or g.compression_ratio > 2.4:
        return False
    if g.no_speech_prob > 0.97 and g.avg_logprob < -1.0:
        return False
    low = text.lower()
    if any(h in low for h in _HALLUCINATIONS):
        return False
    # A copy of a line already transcribed is Whisper echoing, not new speech.
    return not any(difflib.SequenceMatcher(None, text, k).ratio() > 0.8 for k in known)


def _fill_gaps(vocals_path: str, segments: list[dict], cache_dir: Path):
    gaps_path = cache_dir / "transcript_gaps.json"
    if gaps_path.exists():
        gaps = json.loads(gaps_path.read_text(encoding="utf-8"))
        return gaps, (
            f"✓ {len(gaps)} line(s) recovered from speech Whisper skipped (cached)" if gaps else ""
        )

    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    sr = 16000
    audio = decode_audio(vocals_path, sampling_rate=sr)
    speech = get_speech_timestamps(
        audio,
        VadOptions(
            threshold=_GAP_VAD_THRESHOLD,
            min_speech_duration_ms=250,
            min_silence_duration_ms=300,
            speech_pad_ms=150,
        ),
    )

    # Time already covered by Whisper's words (padded a little).
    covered = []
    for d in segments:
        for w in d.get("words") or []:
            covered.append((w["start"] - 0.25, w["end"] + 0.25))
        if not d.get("words"):
            covered.append((d["start"], d["end"]))
    covered.sort()

    def uncovered(a: float, b: float) -> list[tuple[float, float]]:
        out, t = [], a
        for c0, c1 in covered:
            if c1 <= t or c0 >= b:
                continue
            if c0 > t:
                out.append((t, c0))
            t = max(t, c1)
        if t < b:
            out.append((t, b))
        return [(x, y) for x, y in out if y - x >= _GAP_MIN]

    spans = [g for s in speech for g in uncovered(s["start"] / sr, s["end"] / sr)]
    # Join pieces of one gap separated by tiny pauses.
    merged: list[list[float]] = []
    for a, b in spans:
        if merged and a - merged[-1][1] < 0.4:
            merged[-1][1] = b
        else:
            merged.append([a, b])

    lang = next((d.get("lang") for d in segments if d.get("lang")), None)
    model = _whisper()[0] if merged else None
    known = [d["text"] for d in segments]
    gaps, rejected = [], 0
    for a, b in merged:
        clip = audio[int(a * sr) : int(b * sr)]
        # Several decodings, best kept. No previous-line prompt: Whisper
        # copies it. Whisper's no-speech score is unreliable over music, and
        # the VAD has already confirmed speech, so only known hallucinations
        # are rejected.
        best = None
        for lg in (None, lang):
            segs, _ = model.transcribe(
                clip,
                beam_size=5,
                language=lg,
                word_timestamps=True,
                vad_filter=False,
                condition_on_previous_text=False,
            )
            segs = [g for g in segs if _plausible_gap(g, known)]
            score = sum(len(g.text.strip()) for g in segs)
            if segs and (best is None or score > best[0]):
                best = (score, segs)
        whisper_text = " ".join(g.text.strip() for g in best[1]) if best else ""
        # Bodhan indic-transcribe is much steadier than Whisper on speech over
        # music, so it gives the main text for gaps; Whisper's attempt is kept
        # as a second opinion for the refine stage.
        bodhan_text = _bodhan_text(clip, lang)
        if not bodhan_text and not whisper_text:
            rejected += 1
            continue
        if bodhan_text:
            gaps.append(
                Segment(
                    id=-1,
                    start=round(a, 3),
                    end=round(b, 3),
                    text=bodhan_text,
                    lang=lang or "",
                    words=[],
                    asr_flag="gap",
                    asr_alt=whisper_text or None,
                ).to_dict()
            )
            continue
        for g in best[1]:
            words = [
                {"word": w.word, "start": round(a + w.start, 3), "end": round(a + w.end, 3)}
                for w in (g.words or [])
            ]
            gaps.append(
                Segment(
                    id=-1,
                    start=round(a + g.start, 3),
                    end=round(a + g.end, 3),
                    text=g.text.strip(),
                    lang=lang or "",
                    words=words,
                    asr_flag="gap",
                ).to_dict()
            )

    gaps_path.write_text(json.dumps(gaps, ensure_ascii=False, indent=1), encoding="utf-8")
    if not merged:
        return gaps, "✓ No skipped speech found"
    return gaps, (
        f"✓ Checked {len(merged)} stretch(es) of speech Whisper skipped "
        f"({sum(b - a for a, b in merged):.1f}s): recovered {len(gaps)} line(s)"
        + (f", {rejected} rejected as noise/hallucination" if rejected else "")
    )


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
            id=a.id,
            start=a.start,
            end=b.end,
            text=f"{a.text.rstrip()} {b.text.lstrip()}".strip(),
            lang=a.lang,
            words=(a.words or []) + (b.words or []),
            speaker=a.speaker,
            asr_flag=a.asr_flag or b.asr_flag,
        )

    def same_speaker(a: Segment, b: Segment) -> bool:
        return a.speaker == b.speaker

    # Pass 1: glue pieces of one sentence back together — same speaker, no
    # closing punctuation on the first piece, short pause between them.
    out: list[Segment] = [segs[0]]
    for s in segs[1:]:
        cur = out[-1]
        gap = s.start - cur.end
        if (
            same_speaker(cur, s)
            and not cur.text.rstrip().endswith(_TERMINAL)
            and gap <= _MERGE_GAP
            and s.end - cur.start <= _MAX_SENTENCE
        ):
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
