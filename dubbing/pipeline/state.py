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

    _FIELDS = {
        "id", "start", "end", "text", "lang", "words",
        "dubbed_text", "dubbed_audio", "aligned_audio",
        "lip_synced_clip", "original_clip", "edit_instruction", "edit_hash",
    }

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
    audio_path: Optional[str]
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
