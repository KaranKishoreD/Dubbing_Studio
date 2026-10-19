import json
from pathlib import Path
from ..state import PipelineState, Segment
from ..cache import get_cache_dir


def transcribe_node(state: PipelineState) -> dict:
    video_id = state["video_id"]
    cache_dir = get_cache_dir(video_id)
    transcript_path = cache_dir / "transcript.json"

    if transcript_path.exists():
        with open(transcript_path, encoding="utf-8") as f:
            segments = json.load(f)
        return {
            "segments": segments,
            "log": state.get("log", []) + [
                f"✓ Transcript loaded from cache ({len(segments)} segments)"
            ],
        }

    import torch
    from faster_whisper import WhisperModel

    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = "float16" if device == "cuda" else "int8"

    # "tiny" transcribes ~10s of audio in <3s on CPU. Accuracy is sufficient for
    # Indian lectures with clear speech. Upgrade to "small" for noisy recordings.
    model_size = "tiny"
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

    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(segments, f, ensure_ascii=False, indent=2)

    return {
        "segments": segments,
        "log": state.get("log", []) + [
            f"✓ Transcribed {len(segments)} segments (detected: {detected_lang})"
        ],
    }
