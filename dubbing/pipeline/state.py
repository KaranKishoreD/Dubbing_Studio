"""
The data passed between stages.

  Segment        one line of dialogue: its time, original text, speaker,
                 translation and every audio/video file made for it.
  PipelineState  the whole project: source video, settings chosen in the
                 studio, all segments, and the run log.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, TypedDict


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
    gender: Optional[str] = None  # "male" | "female" | None if undetermined
    f0: Optional[float] = None  # median fundamental frequency, Hz
    voice: Optional[str] = None  # resolved Bodhan voice name
    # Set by diarize_node via prosody.py: how the line was delivered relative
    # to the speaker's own baseline ("neutral", "energetic", "calm", ...).
    delivery: Optional[str] = None
    pace: Optional[str] = None  # "quick" | "slow" | None
    # Second transcription of the same audio (Whisper, when text came from
    # Bodhan). Passed to the translator to reconcile recognition errors.
    asr_alt: Optional[str] = None
    # Lip-synced pieces of this line: [{"start", "end", "path"}]. Only shots
    # that show the speaker's own face are synced; the dub may borrow silence
    # after the line, so a piece can end later than `end`.
    sync_clips: List[dict] = field(default_factory=list)
    # Set by edits: leave this line undubbed (original audio + video), and a
    # Bodhan style that overrides the measured delivery for this line.
    keep_original: bool = False
    style_override: Optional[str] = None
    # Set by edits: "speaker" = clone this line from the speaker's own
    # reference only, so all their lines share one steady voice.
    clone_ref: Optional[str] = None
    # Set by edits: dub this line in another language than the rest
    # (a language code, e.g. "bn").
    lang_override: Optional[str] = None
    # Set by voice/emotion.py: the feeling this line is voiced with
    # ("neutral", "excited", "surprised", ...). None = not labelled yet.
    emotion: Optional[str] = None
    # False = this line's current audio has not been lip-synced (it was made
    # in fast mode); True = the lip-sync stage has processed it. None = made
    # before this was recorded.
    synced: Optional[bool] = None
    # Transcript provenance: "gap" = recovered from speech Whisper skipped,
    # "low" = refine step is unsure. asr_raw = the text before refinement.
    asr_flag: Optional[str] = None
    asr_raw: Optional[str] = None
    # Song mode: this piece is sung lyrics (dubbed_text holds the sung
    # lyric, re-sung in assemble) and how the original is sung.
    sung: bool = False
    song_style: Optional[str] = None
    song_gender: Optional[str] = None

    _FIELDS = {
        "id",
        "start",
        "end",
        "text",
        "lang",
        "words",
        "dubbed_text",
        "dubbed_audio",
        "aligned_audio",
        "lip_synced_clip",
        "original_clip",
        "edit_instruction",
        "edit_hash",
        "speaker",
        "gender",
        "f0",
        "voice",
        "delivery",
        "pace",
        "asr_alt",
        "sync_clips",
        "keep_original",
        "style_override",
        "clone_ref",
        "lang_override",
        "emotion",
        "synced",
        "asr_flag",
        "asr_raw",
        "sung",
        "song_style",
        "song_gender",
    }

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict:
        return {k: getattr(self, k) for k in self._FIELDS}

    @classmethod
    def from_dict(cls, d: dict) -> "Segment":
        return cls(**{k: v for k, v in d.items() if k in cls._FIELDS})


class PipelineState(TypedDict):
    url: str
    target_lang: str
    domain: str
    video_id: str
    video_path: Optional[str]
    audio_path: Optional[str]  # 16 kHz mono — analysis only (ASR, VAD)
    audio_hq_path: Optional[str]  # 44.1 kHz stereo — the base of the final mix
    vocals_path: Optional[str]
    background_path: Optional[str]
    voice_ref_path: Optional[str]
    segments: List[dict]
    final_video: Optional[str]
    final_audio: Optional[str]  # the dubbed soundtrack alone, browser-playable
    edit_ops: List[dict]
    log: List[str]
    error: Optional[str]
    # Voice selection
    voice_name: Optional[str]  # specific Bodhan voice (e.g. "Kavya", "Arjun")
    voice_style: Optional[str]  # speaking style / mood (e.g. "happy", "educational lecture")
    # Set by diarize_node: {speaker_id: {"gender", "f0", "n_segments"}}
    speaker_profiles: dict
    # Known speaker count from the UI; None = detect automatically
    num_speakers: Optional[int]
    # Fast mode: False = do not render lip sync (only reuse finished shots).
    lipsync: Optional[bool]
    # Set by align: lines whose audio is new in this run (the only lines
    # lip sync renders). None = not known, e.g. an "Add lip sync" job.
    new_audio: Optional[List[int]]
    # lines whose lip sync is rendered again even though their audio is
    # unchanged (a "lip_sync" edit)
    resync_ids: Optional[List[int]]
    # Per-speaker voice chosen in the UI: {"0": "Kavya", ...}. Speakers not
    # listed are cast automatically.
    voice_map: dict
    # How much English stays in the translation: pure | natural | heavy
    lang_mix: Optional[str]
    # "indic": Bodhan translates every line, the LLM only the lines it fails;
    # "llm": the LLM translates the whole dialogue.
    translator: Optional[str]
    # "indic" (Bodhan/Sarvam, the 22 Indian languages) or "international"
    # (LLM translation, Kokoro voices: English and six world languages)
    mode: Optional[str]
    # Sung lyrics: "resing" (song mode, re-sung in the dub language) or
    # "keep" (left as the original singing).
    songs: Optional[str]
    # Optional part of the video to dub, in seconds of the source video.
    # Only this range is downloaded and processed.
    clip_start: Optional[float]
    clip_end: Optional[float]
