"""
TTS synthesis via Bodhan indic-speak — all segments synthesized in parallel.
₹6 / 10K input characters — 45 voices across 22 Indian languages.

Voice auto-selection
─────────────────────
Each segment's original audio is analyzed for fundamental frequency (F0/pitch).
  F0 > 165 Hz  → female voice   (typical female speech range)
  F0 ≤ 165 Hz  → male voice     (typical male speech range)

This handles multi-speaker videos automatically — a high-pitched question gets a
female voice, a low-pitched answer gets a male voice.

Manual override: set voice_name in state to use a specific voice for all segments.
"""

import os
import json
import numpy as np
import soundfile as sf
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ..state import PipelineState, Segment
from ..cache import get_segments_dir, instruction_hash
from openai import OpenAI

# ── Complete voice catalogue (from Bodhan API docs) ───────────────────────────
# Source: https://console.bodhan.ai/api-docs/#text-to-speech-api
# Any voice can read any supported language — voice sets persona/timbre.

_FEMALE_VOICE_MAP: dict[str, str] = {
    "as":  "Prastuti",
    "bn":  "Ishita",
    "brx": "Gwrbw",
    "doi": "Preeti",
    "gu":  "Dhara",
    "hi":  "Kavya",       # also: Suhani
    "kn":  "Deepika",
    "kok": "Anjali",
    "ks":  "Zoon",
    "mai": "Vaidehi",
    "ml":  "Lakshmi",
    "mni": "Thoibi",
    "mr":  "Anagha",
    "ne":  "Srijana",
    "or":  "Itishree",
    "pa":  "Kaur",
    "sa":  "Bharati",
    "sat": "Phulmani",
    "sd":  "Moomal",
    "ta":  "Anitha",
    "te":  "Sravani",
    "ur":  "Saba",
    "en":  "Kavya",
}

_MALE_VOICE_MAP: dict[str, str] = {
    "as":  "Ankur",
    "bn":  "Sourav",
    "brx": "Sansuma",
    "doi": "Sham",
    "gu":  "Parth",
    "hi":  "Amit",
    "kn":  "Adarsh",
    "kok": "Sandeep",
    "ks":  "Ishfaq",
    "mai": "Madhukar",
    "ml":  "Kiran",
    "mni": "Chaoba",
    "mr":  "Chinmay",
    "ne":  "Sagar",
    "or":  "Akash",
    "pa":  "Manpreet",
    "sa":  "Aryaman",
    "sat": "Sibu",
    "sd":  "Rano",
    "ta":  "Arun",
    "te":  "Vamsi",
    "ur":  "Zaid",
    "en":  "Amit",
}

_DEFAULT_FEMALE = "Kavya"
_DEFAULT_MALE   = "Amit"

# F0 threshold separating male (lower) from female (higher) speech
_FEMALE_F0_HZ = 165.0

# ── Speaking styles ────────────────────────────────────────────────────────────
BODHAN_STYLES: dict[str, str] = {
    "Educational / Lecture":    "educational lecture",
    "TV News / Formal":         "TV style news",
    "AIR News":                 "AIR style news",
    "General News":             "news",
    "Advertisement":            "advertisements",
    "Customer Care":            "Customer Care",
    "Happy / Energetic":        "happy",
    "Sad / Emotional":          "sad",
    "Angry":                    "anger",
    "Surprised / Dramatic":     "surprise",
    "Fear":                     "fear",
    "Disgusted":                "disgust",
    "Storytelling / Audiobook": "single person narration audiobook",
    "Children's Story":         "children's stories",
}
DEFAULT_STYLE_LABEL = "Educational / Lecture"
DEFAULT_STYLE       = "educational lecture"


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://api.bodhan.ai/v1",
        api_key=os.environ["BODHAN_SPEAK_KEY"],
    )


def _pick_voice(lang_code: str, gender: str) -> str:
    if gender == "male":
        return _MALE_VOICE_MAP.get(lang_code, _DEFAULT_MALE)
    return _FEMALE_VOICE_MAP.get(lang_code, _DEFAULT_FEMALE)


def _estimate_f0(audio_path: str) -> float:
    """
    Estimate mean fundamental frequency (Hz) of the audio segment using
    autocorrelation. Returns 0.0 if speech is undetected or file is silent.
    No external dependencies beyond numpy + soundfile.
    """
    try:
        data, sr = sf.read(audio_path)
        if data.ndim > 1:
            data = data.mean(axis=1)
        data = data.astype(np.float32)

        # Use 50ms frames to estimate pitch
        frame_len = int(sr * 0.05)
        pitches = []
        for start in range(0, len(data) - frame_len, frame_len // 2):
            frame = data[start: start + frame_len]
            if np.max(np.abs(frame)) < 0.01:  # silence
                continue
            # Normalized autocorrelation
            corr = np.correlate(frame, frame, mode="full")
            corr = corr[len(corr) // 2:]
            corr = corr / (corr[0] + 1e-10)
            # Search for peak in 50–500 Hz range
            lo = max(1, int(sr / 500))
            hi = int(sr / 50)
            if hi >= len(corr):
                continue
            peak_idx = lo + int(np.argmax(corr[lo:hi]))
            if corr[peak_idx] > 0.3:   # confidence threshold
                pitches.append(sr / peak_idx)

        return float(np.median(pitches)) if pitches else 0.0
    except Exception:
        return 0.0


def synthesize_node(state: PipelineState) -> dict:
    segments    = [Segment.from_dict(s) for s in state["segments"]]
    target_lang = state["target_lang"]
    edit_ops    = state.get("edit_ops", [])
    edit_ids    = {sid for op in edit_ops for sid in op["segment_ids"]}
    lang_code   = target_lang.split("-")[0].lower()

    # Global voice override from UI (if set, used for all segments)
    global_voice_name  = state.get("voice_name") or None
    global_voice_style = state.get("voice_style") or DEFAULT_STYLE

    segs_dir = get_segments_dir(state["video_id"])
    vocals_path = state.get("vocals_path")   # full audio for pitch slicing
    client = _client()

    to_synth = [seg for seg in segments if seg.dubbed_text is not None]

    def _synth_one(seg: Segment) -> tuple[int, str | None]:
        h = instruction_hash(seg.edit_instruction) if seg.edit_instruction else "default"
        audio_path = segs_dir / f"s{seg.id:04d}_dubbed_{h}.wav"

        if audio_path.exists() and seg.id not in edit_ids:
            return seg.id, str(audio_path)

        # ── Voice selection ──────────────────────────────────────────────────
        if global_voice_name:
            voice = global_voice_name
        else:
            # Auto-detect gender from segment's original audio pitch
            seg_audio = segs_dir / f"s{seg.id:04d}_original_audio.wav"
            if not seg_audio.exists() and vocals_path:
                _slice_audio(vocals_path, str(seg_audio), seg.start, seg.end)

            if seg_audio.exists():
                f0 = _estimate_f0(str(seg_audio))
                gender = "female" if (f0 == 0.0 or f0 >= _FEMALE_F0_HZ) else "male"
            else:
                gender = "female"

            voice = _pick_voice(lang_code, gender)

        # ── Style: per-segment instruction overrides global style ────────────
        seg_style = _resolve_style(seg.edit_instruction, global_voice_style)
        instructions_obj = {"lang": lang_code, "style": seg_style}

        response = client.audio.speech.create(
            model="indic-speak",
            input=seg.dubbed_text,
            voice=voice,
            extra_body={"instructions": json.dumps(instructions_obj)},
        )
        with open(audio_path, "wb") as f:
            f.write(response.content)
        return seg.id, str(audio_path)

    seg_map = {s.id: s for s in segments}
    processed = 0

    with ThreadPoolExecutor(max_workers=min(len(to_synth), 8)) as pool:
        futures = {pool.submit(_synth_one, seg): seg for seg in to_synth}
        for future in as_completed(futures):
            seg_id, audio_path = future.result()
            if audio_path:
                seg_map[seg_id].dubbed_audio = audio_path
                processed += 1

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + [
            f"✓ Synthesized {processed} segments "
            f"(pitch-matched voices, style: {global_voice_style}, lang: {lang_code})"
        ],
    }


def _slice_audio(vocals_path: str, out_path: str, start: float, end: float) -> None:
    """Extract a segment's audio slice from the full vocals track."""
    try:
        import subprocess
        subprocess.run(
            ["ffmpeg", "-i", vocals_path,
             "-ss", str(start), "-t", str(round(end - start, 3)),
             "-ar", "16000", "-ac", "1", out_path, "-y"],
            capture_output=True, check=True,
        )
    except Exception:
        pass


def _resolve_style(edit_instruction: str | None, global_style: str) -> str:
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
