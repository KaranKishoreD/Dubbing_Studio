from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, TypedDict
import hashlib


@dataclass
class Segment:
    id: int
    start: float
    end: float
    text: str
    lang: str
    words: List[dict] = field(default_factory=list)
    dubbed_text: Optional[str] = None
    dubbed_audio: Optional[str] = None
    aligned_audio: Optional[str] = None
    lip_synced_clip: Optional[str] = None
    original_clip: Optional[str] = None
    edit_instruction: Optional[str] = None
    edit_hash: str = "default"
    # Set by diarize_node
    speaker: Optional[int] = None
    gender: Optional[str] = None       # "male" | "female" | None if undetermined
    f0: Optional[float] = None         # median fundamental frequency, Hz
    voice: Optional[str] = None        # resolved Bodhan voice name
    # Set by diarize_node via prosody.py: how the line was delivered relative
    # to the speaker's own baseline ("neutral", "energetic", "calm", ...).
    delivery: Optional[str] = None
    pace: Optional[str] = None         # "quick" | "slow" | None
    # Second transcription of the same audio (Whisper, when text came from
    # Bodhan). Passed to the translator to reconcile recognition errors.
    asr_alt: Optional[str] = None

    _FIELDS = {
        "id", "start", "end", "text", "lang", "words",
        "dubbed_text", "dubbed_audio", "aligned_audio",
        "lip_synced_clip", "original_clip", "edit_instruction", "edit_hash",
        "speaker", "gender", "f0", "voice", "delivery", "pace", "asr_alt",
    }

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**{k: v for k, v in d.items() if k in cls._FIELDS})

    def clear_downstream(self, from_stage: str) -> None:
        """Clear cached outputs from a given stage onward so they get recomputed."""
        order = ["dubbed_text", "dubbed_audio", "aligned_audio", "lip_synced_clip"]
        clearing = False
        for attr in order:
            if attr == from_stage:
                clearing = True
            if clearing:
                setattr(self, attr, None)


class PipelineState(TypedDict):
    url: str
    target_lang: str
    domain: str
    video_id: str
    video_path: Optional[str]
    audio_path: Optional[str]          # 16 kHz mono — analysis only (ASR, VAD)
    audio_hq_path: Optional[str]       # 44.1 kHz stereo — the base of the final mix
    vocals_path: Optional[str]
    background_path: Optional[str]
    voice_ref_path: Optional[str]
    segments: List[dict]
    final_video: Optional[str]
    edit_ops: List[dict]
    log: List[str]
    error: Optional[str]
    # Voice selection
    voice_name: Optional[str]      # specific Bodhan voice (e.g. "Kavya", "Arjun")
    voice_style: Optional[str]     # speaking style / mood (e.g. "happy", "educational lecture")
    # Set by diarize_node: {speaker_id: {"gender", "f0", "n_segments"}}
    speaker_profiles: dict
    # Known speaker count from the UI; None = detect automatically
    num_speakers: Optional[int]
