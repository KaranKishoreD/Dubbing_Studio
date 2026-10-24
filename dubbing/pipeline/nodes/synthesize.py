"""
TTS synthesis via Bodhan indic-speak — all segments synthesized in parallel.
₹6 / 10K input characters — 45 voices across 22 Indian languages.

Voice auto-selection
─────────────────────
Gender and speaker identity come from diarize_node, which clusters segments by
voice and pools F0 across each speaker's whole set of segments. One decision
per speaker, so a speaker keeps the same voice for the entire video.

Manual override: set voice_name in state to use a specific voice for all segments.
"""

import os
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from ..state import PipelineState, Segment
from ..cache import get_segments_dir, instruction_hash
from openai import OpenAI

# ── Complete voice catalogue (from Bodhan API docs) ───────────────────────────
# Source: https://console.bodhan.ai/api-docs/#text-to-speech-api
# Any voice can read any supported language — voice sets persona/timbre.

# Lists, not single names: a language may have more than one voice (Hindi has
# two female voices), and every one of the 45 must be reachable when casting.
# First entry is the language's primary voice.
_FEMALE_VOICES: dict[str, list[str]] = {
    "hi":  ["Kavya", "Suhani"],
    "as":  ["Prastuti"],
    "bn":  ["Ishita"],
    "brx": ["Gwrbw"],
    "doi": ["Preeti"],
    "gu":  ["Dhara"],
    "kn":  ["Deepika"],
    "kok": ["Anjali"],
    "ks":  ["Zoon"],
    "mai": ["Vaidehi"],
    "ml":  ["Lakshmi"],
    "mni": ["Thoibi"],
    "mr":  ["Anagha"],
    "ne":  ["Srijana"],
    "or":  ["Itishree"],
    "pa":  ["Kaur"],
    "sa":  ["Bharati"],
    "sat": ["Phulmani"],
    "sd":  ["Moomal"],
    "ta":  ["Anitha"],
    "te":  ["Sravani"],
    "ur":  ["Saba"],
}

_MALE_VOICES: dict[str, list[str]] = {
    "hi":  ["Amit"],
    "as":  ["Ankur"],
    "bn":  ["Sourav"],
    "brx": ["Sansuma"],
    "doi": ["Sham"],
    "gu":  ["Parth"],
    "kn":  ["Adarsh"],
    "kok": ["Sandeep"],
    "ks":  ["Ishfaq"],
    "mai": ["Madhukar"],
    "ml":  ["Kiran"],
    "mni": ["Chaoba"],
    "mr":  ["Chinmay"],
    "ne":  ["Sagar"],
    "or":  ["Akash"],
    "pa":  ["Manpreet"],
    "sa":  ["Aryaman"],
    "sat": ["Sibu"],
    "sd":  ["Rano"],
    "ta":  ["Arun"],
    "te":  ["Vamsi"],
    "ur":  ["Zaid"],
}

# English source/target reads best with the Hindi voices.
_LANG_ALIAS = {"en": "hi"}

ALL_FEMALE_VOICES = [v for vs in _FEMALE_VOICES.values() for v in vs]
ALL_MALE_VOICES   = [v for vs in _MALE_VOICES.values() for v in vs]
ALL_BODHAN_VOICES = ALL_FEMALE_VOICES + ALL_MALE_VOICES

# ── Sarvam Bulbul v3 ──────────────────────────────────────────────────────────
# A second voice engine (SARVAM_API_KEY). Every Bulbul voice speaks every
# supported language, so the pool is not per language. Ids carry a "sarvam:"
# prefix so a cast can mix engines and the cache key separates them. Gender
# is from the voice name, confirmed by measured pitch (voice_profiles).
SARVAM_MODEL = os.environ.get("SARVAM_TTS_MODEL", "bulbul:v3")
_SARVAM_FEMALE = ["ritu", "priya", "neha", "pooja", "simran", "kavya", "ishita", "shreya",
                  "roopa", "tanya", "shruti", "suhani", "kavitha", "rupali"]
_SARVAM_MALE = ["aditya", "ashutosh", "rahul", "rohan", "amit", "dev", "ratan", "varun",
                "manan", "sumit", "kabir", "aayan", "shubh", "advait", "anand", "tarun",
                "sunny", "mani", "gokul", "vijay", "mohit", "rehan", "soham"]
SARVAM_VOICES = [f"sarvam:{v}" for v in _SARVAM_FEMALE + _SARVAM_MALE]
# Languages Bulbul speaks (BCP-47 with -IN).
_SARVAM_LANGS = {"hi", "bn", "ta", "te", "kn", "ml", "mr", "gu", "pa", "or", "en"}

# ── OpenAI gpt-audio-mini (via OpenRouter) ───────────────────────────────────
# Speaks code-mixed lines the way bilingual people do: "relax" and "chill"
# come out as English, where Bodhan says "rulox" / "sill". It is a chat model
# that performs the text, so it can improvise (a 4 s line once came back as
# 12.9 s). Every clip is therefore checked — length plausible for the text,
# Whisper hears roughly the same words — retried, and on failure that line
# falls back to Bodhan.
GPT_AUDIO_MODEL = os.environ.get("GPT_AUDIO_MODEL", "openai/gpt-audio-mini")
_GPT_FEMALE = ["shimmer", "coral", "nova", "sage"]
_GPT_MALE = ["ash", "echo", "onyx", "ballad"]
GPT_VOICES = [f"gptaudio:{v}" for v in _GPT_FEMALE + _GPT_MALE]
_GPT_TRIES = int(os.environ.get("GPT_AUDIO_TRIES", "3"))

ALL_VOICES = ALL_BODHAN_VOICES + SARVAM_VOICES + GPT_VOICES
# "bodhan" | "sarvam" | "gptaudio" — which engine automatic casting draws from.
TTS_BACKEND = os.environ.get("TTS_BACKEND", "bodhan").lower()


def is_sarvam(voice: str | None) -> bool:
    return bool(voice) and voice.startswith("sarvam:")


def is_gpt_audio(voice: str | None) -> bool:
    return bool(voice) and voice.startswith("gptaudio:")


def voice_label(voice: str | None) -> str:
    """Display name: 'Anand (Sarvam)' / 'Shimmer (gpt-audio)' / 'Arun'."""
    if not voice:
        return "—"
    if is_sarvam(voice):
        return f"{voice.split(':', 1)[1].title()} (Sarvam)"
    if is_gpt_audio(voice):
        return f"{voice.split(':', 1)[1].title()} (gpt-audio)"
    return voice


def _sarvam_gender(voice: str) -> str:
    return "female" if voice.split(":", 1)[1] in _SARVAM_FEMALE else "male"


def voice_catalogue() -> list[tuple[str, str, str]]:
    """[(voice id, "female"|"male", native language code or "any")] — Bodhan's
    45 voices plus Sarvam's 37."""
    out = []
    for gender, table in (("female", _FEMALE_VOICES), ("male", _MALE_VOICES)):
        for lang, voices in table.items():
            out += [(v, gender, lang) for v in voices]
    out += [(v, _sarvam_gender(v), "any") for v in SARVAM_VOICES]
    out += [(v, "female" if v.split(":")[1] in _GPT_FEMALE else "male", "any")
            for v in GPT_VOICES]
    return out


def _sarvam_tts(text: str, lang_code: str, voice: str) -> bytes:
    """WAV bytes from Sarvam Bulbul. Retries rate limits with back-off."""
    import base64
    import time
    import requests
    if lang_code not in _SARVAM_LANGS:
        raise RuntimeError(f"Sarvam Bulbul does not support '{lang_code}'")
    lang = lang_code
    body = {"text": text, "target_language_code": f"{lang}-IN", "model": SARVAM_MODEL,
            "speaker": voice.split(":", 1)[1], "speech_sample_rate": 24000}
    for attempt in range(5):
        r = requests.post("https://api.sarvam.ai/text-to-speech", json=body, timeout=120,
                          headers={"api-subscription-key": os.environ["SARVAM_API_KEY"]})
        if r.status_code == 429 and attempt < 4:
            time.sleep(2 * (attempt + 1))
            continue
        if not r.ok:
            raise RuntimeError(f"Sarvam TTS {r.status_code}: {r.text[:200]}")
        return base64.b64decode(r.json()["audios"][0])
    raise RuntimeError("Sarvam TTS: rate limited")


def _bodhan_tts(client, text: str, lang_code: str, voice: str, style: str) -> bytes:
    """WAV bytes from Bodhan indic-speak. A rate limit (429) is retried with
    back-off; without it, lines that hit a busy moment came out silent."""
    import time
    for attempt in range(5):
        try:
            return client.audio.speech.create(
                model="indic-speak", input=text, voice=voice,
                extra_body={"instructions": json.dumps({"lang": lang_code, "style": style})},
            ).content
        except Exception as exc:
            if "429" not in str(exc) or attempt == 4:
                raise
            time.sleep(2 * (attempt + 1))


_GPT_SYSTEM = (
    "You are a voice actor dubbing a film. Read the user's line aloud EXACTLY as "
    "written — every word, nothing added, nothing removed, nothing translated, no "
    "greeting, no comment. Speak it the way a fluent {lang} speaker talks in "
    "everyday conversation: relaxed and natural. Words written in English are "
    "said as natural English, the way bilingual {lang} speakers say them. "
    # Tested on the lines that failed: quoting the line plus a duration hint
    # passed 8/8 (framing alone 6/8, duration alone 0/8, full gpt-audio 5/8).
    "The user's message contains ONLY the line to read, between « and ». Say "
    "exactly the text between the marks and nothing else — no other words, no "
    "repetition, no pause filler. This line should take about {secs:.1f} "
    "seconds to say; never pad it.")


def _gpt_audio_once(text: str, lang_code: str, voice: str) -> tuple[bytes, float]:
    """(wav bytes, seconds) from gpt-audio via OpenRouter."""
    import base64
    import io
    import requests
    import wave
    from .translate import LANG_NAMES
    from .. import tts_rate
    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions", timeout=120, stream=True,
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        json={"model": GPT_AUDIO_MODEL, "modalities": ["text", "audio"], "stream": True,
              "audio": {"voice": voice.split(":", 1)[1], "format": "pcm16"},
              "messages": [{"role": "system", "content": _GPT_SYSTEM.format(
                                lang=LANG_NAMES.get(lang_code, lang_code),
                                secs=max(0.6, len(text) / tts_rate.chars_per_sec(lang_code)))},
                           {"role": "user", "content": f"«{text}»"}]})
    if r.status_code != 200:
        raise RuntimeError(f"gpt-audio {r.status_code}: {r.text[:200]}")
    pcm = bytearray()
    for raw in r.iter_lines():
        if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
            continue
        for ch in json.loads(raw[6:]).get("choices", []):
            a = (ch.get("delta") or {}).get("audio") or {}
            if a.get("data"):
                pcm += base64.b64decode(a["data"])
    if not pcm:
        raise RuntimeError("gpt-audio returned no audio")
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(24000)
        w.writeframes(bytes(pcm))
    return buf.getvalue(), len(pcm) / 48000


def _gpt_audio_ok(wav: bytes, secs: float, text: str, lang_code: str) -> str:
    """'' if the clip plausibly reads `text`, else the reason it failed."""
    import io
    import difflib
    import re
    from .. import tts_rate
    expected = len(text) / tts_rate.chars_per_sec(lang_code)
    if secs > 2.0 * expected + 1.0:
        return f"too long ({secs:.1f}s for ~{expected:.1f}s of text)"
    if secs < 0.4 * expected:
        return f"too short ({secs:.1f}s for ~{expected:.1f}s of text)"
    try:
        import librosa
        from .transcribe import _whisper
        y, sr = sf_read(io.BytesIO(wav))
        y = librosa.resample(y, orig_sr=sr, target_sr=16000)
        segs, _ = _whisper()[0].transcribe(y, language=lang_code, beam_size=5,
                                           vad_filter=False, condition_on_previous_text=False)
        heard = " ".join(g.text.strip() for g in segs)
    except Exception:
        return ""                          # cannot verify → length check only
    norm = lambda t: re.sub(r"[\W_]+", "", t.lower())
    a, b = norm(text), norm(heard)
    if not b:
        return "nothing intelligible heard"
    ratio = len(b) / max(1, len(a))
    if ratio > 1.8:
        return f"says more than the text (heard {len(b)} vs {len(a)} chars)"
    if ratio < 0.45:
        return f"says much less than the text (heard {len(b)} vs {len(a)} chars)"
    # Script-independent overlap: English words may come back in Tamil script,
    # so only flag clips that share almost nothing with the text.
    if difflib.SequenceMatcher(None, a, b).ratio() < 0.15 and ratio > 1.3:
        return "words do not match the text"
    return ""


def sf_read(f):
    import soundfile as sf
    y, sr = sf.read(f, dtype="float32", always_2d=True)
    return y.mean(axis=1), sr


def _gpt_audio_tts(text: str, lang_code: str, voice: str) -> bytes:
    last = ""
    for _ in range(_GPT_TRIES):
        wav, secs = _gpt_audio_once(text, lang_code, voice)
        last = _gpt_audio_ok(wav, secs, text, lang_code)
        if not last:
            return wav
    raise GptAudioRejected(f"gpt-audio failed checks {_GPT_TRIES}× ({last})")


class GptAudioRejected(RuntimeError):
    pass


def tts_bytes(text: str, lang_code: str, voice: str, style: str, client=None) -> bytes:
    if is_sarvam(voice):
        return _sarvam_tts(text, lang_code, voice)
    if is_gpt_audio(voice):
        return _gpt_audio_tts(text, lang_code, voice)
    return _bodhan_tts(client or _client(), text, lang_code, voice, style)


def preview_voice(voice: str, lang_code: str, text: str, out_path: Path) -> str:
    """Synthesize a short sample of `voice` reading `text` (cached on disk)."""
    if not out_path.exists():
        audio = tts_bytes(text, lang_code, voice, _AUTO_BASE_STYLE)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(audio)
    return str(out_path)

_DEFAULT_FEMALE = "Kavya"
_DEFAULT_MALE   = "Amit"

# Used only when diarize could not measure pitch at all. Male is the safer
# default for lecture/talk content, which skews male; override with
# UNKNOWN_GENDER_FALLBACK=female or set a voice explicitly in the UI.
_UNKNOWN_GENDER_FALLBACK = os.environ.get("UNKNOWN_GENDER_FALLBACK", "male")

# ── Speaking styles ────────────────────────────────────────────────────────────
BODHAN_STYLES: dict[str, str] = {
    # Per-line: each line's measured delivery (prosody.py) picks its style,
    # on top of a neutral "educational lecture" base.
    "Auto (match original delivery)": "auto",
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
DEFAULT_STYLE_LABEL = "Auto (match original delivery)"
DEFAULT_STYLE       = "auto"
_AUTO_BASE_STYLE    = os.environ.get("AUTO_BASE_STYLE", "educational lecture")


def _line_style(seg: Segment, global_style: str) -> str:
    """Global style, or in auto mode the style matching this line's delivery."""
    if seg.style_override:             # set by an edit ("make line 5 sad")
        return seg.style_override
    if global_style != "auto":
        return global_style
    from ..prosody import BODHAN_FOR_DELIVERY
    return BODHAN_FOR_DELIVERY.get(seg.delivery or "neutral", _AUTO_BASE_STYLE)


def _client() -> OpenAI:
    return OpenAI(
        base_url="https://api.bodhan.ai/v1",
        api_key=os.environ["BODHAN_SPEAK_KEY"],
    )


def _pick_voice(lang_code: str, gender: str) -> str:
    return _voice_pool(lang_code, gender)[0]


# ── Per-speaker casting ───────────────────────────────────────────────────────
# Bodhan ships roughly one female and one male voice per language, so a second
# same-gender speaker has to borrow a voice from another language. Any voice can
# read any supported language — the voice only sets timbre — but a voice from a
# closely related language carries the nearest phonetic habits, so prefer those.
_LANG_GROUPS: list[list[str]] = [
    ["ta", "ml", "te", "kn"],                                  # Dravidian
    ["hi", "mr", "gu", "pa", "ur", "mai", "doi", "sa", "ne"],   # Indo-Aryan, west/central
    ["bn", "as", "or"],                                        # Indo-Aryan, east
    ["ks", "sd", "kok"],                                       # other Indo-Aryan
    ["brx", "mni", "sat"],                                     # Tibeto-Burman / Munda
]


def _voice_pool(lang_code: str, gender: str, backend: str = "bodhan") -> list[str]:
    """
    Ordered voice candidates for a language: its own voice first, then voices
    from languages in the same family, then everything else. Distinct speakers
    take successive entries, so each gets an audibly different voice.
    Sarvam voices are not tied to a language; they are ordered by measured
    pitch from the middle out, so the first picks sound typical, not extreme.
    """
    if backend == "gptaudio":
        return [v for v in GPT_VOICES if (v.split(":")[1] in _GPT_FEMALE) == (gender == "female")]
    if backend == "sarvam" and lang_code in _SARVAM_LANGS:
        prof = voice_profiles()
        vs = [v for v in SARVAM_VOICES if _sarvam_gender(v) == gender]
        f0 = [prof.get(v, {}).get("f0") for v in vs]
        known = sorted(x for x in f0 if x)
        mid = known[len(known) // 2] if known else 0
        return sorted(vs, key=lambda v: abs((prof.get(v, {}).get("f0") or mid) - mid))
    table = _MALE_VOICES if gender == "male" else _FEMALE_VOICES
    lang_code = _LANG_ALIAS.get(lang_code, lang_code)
    group = next((g for g in _LANG_GROUPS if lang_code in g), [])

    ordered: list[str] = []
    for key in [lang_code, *group, *table.keys()]:
        for v in table.get(key, []):
            if v not in ordered:
                ordered.append(v)
    return ordered or [_DEFAULT_MALE if gender == "male" else _DEFAULT_FEMALE]


def _assign_voices(
    segments: list[Segment], lang_code: str, override: str | None,
    voice_map: dict | None = None, backend: str = "bodhan",
) -> dict[int, str]:
    """
    Map each speaker to a voice. One decision per speaker, so a speaker keeps
    the same voice throughout, and two speakers never share one unless the
    catalogue runs out.
    """
    speakers: dict[int, str] = {}      # speaker id -> gender
    talk: dict[int, float] = {}        # speaker id -> seconds spoken
    for seg in segments:
        spk = seg.speaker if seg.speaker is not None else 0
        gender = seg.gender if seg.gender in ("male", "female") else _UNKNOWN_GENDER_FALLBACK
        # First gender seen for a speaker wins; diarize already made it uniform.
        speakers.setdefault(spk, gender)
        talk[spk] = talk.get(spk, 0.0) + seg.duration

    if override:
        return {spk: override for spk in speakers}

    # Voices picked in the UI win; everyone else is cast around them.
    chosen = {int(k): v for k, v in (voice_map or {}).items()
              if v and v in ALL_VOICES and str(k).lstrip("-").isdigit()}
    used: set[str] = set(chosen.values())
    assignment: dict[int, str] = {spk: v for spk, v in chosen.items() if spk in speakers}

    # Deal voices out per gender so same-gender speakers get different ones.
    # Whoever talks most gets the target language's own voice — they carry
    # the video, so they should sound the most native.
    for gender in ("male", "female"):
        pool = _voice_pool(lang_code, gender, backend)
        i = 0
        same = [s for s, g in speakers.items() if g == gender and s not in assignment]
        for spk in sorted(same, key=lambda s: -talk[s]):
            while i < len(pool) and pool[i] in used:
                i += 1
            voice = pool[i] if i < len(pool) else pool[-1]
            used.add(voice)
            assignment[spk] = voice
            i += 1
    return assignment


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
    client = _client()

    to_synth = [seg for seg in segments
                if seg.dubbed_text is not None and not seg.keep_original]
    for seg in segments:
        if seg.keep_original:          # edit: "leave this line as it was"
            seg.dubbed_audio = seg.aligned_audio = None
            seg.sync_clips = []

    # One voice per speaker, decided before any synthesis.
    backend = (state.get("tts_backend") or TTS_BACKEND).lower()
    if backend == "gptaudio" and not os.environ.get("OPENROUTER_API_KEY"):
        backend = "bodhan"
    if backend == "sarvam" and not os.environ.get("SARVAM_API_KEY"):
        backend = "bodhan"
    voice_map = dict(state.get("voice_map") or {})
    swapped = []
    if lang_code not in _SARVAM_LANGS:
        # Bulbul speaks 11 languages; for the rest every Sarvam pick falls
        # back to Bodhan, which covers all 22.
        backend = "bodhan"
        for k, v in list(voice_map.items()):
            if is_sarvam(v):
                swapped.append(voice_label(v))
                voice_map.pop(k)
        if is_sarvam(global_voice_name):
            swapped.append(voice_label(global_voice_name))
            global_voice_name = None
    voice_by_speaker = _assign_voices(segments, lang_code, global_voice_name,
                                      voice_map, backend)
    for seg in segments:
        seg.voice = voice_by_speaker.get(
            seg.speaker if seg.speaker is not None else 0
        )

    # Lines with English kept in English spelling (Sarvam translation) that a
    # Bodhan voice will read: Bodhan mangles Latin, so it gets a native-script
    # copy. The stored dubbed_text keeps the English spelling.
    from .translate import transliterate_latin, has_latin, BODHAN_LATIN
    speak_text: dict[int, str] = {}
    need = [(s.id, s.dubbed_text) for s in to_synth
            if not BODHAN_LATIN and not is_sarvam(s.voice)
            and has_latin(s.dubbed_text) and lang_code != "en"]
    if need:
        speak_text = transliterate_latin(need, lang_code)

    def _synth_one(seg: Segment) -> tuple[int, str | None]:
        # An explicit edit instruction ("make it sad") beats the measured delivery.
        seg_style = (seg.style_override or
                     _resolve_style(seg.edit_instruction, _line_style(seg, global_voice_style)))
        voice = seg.voice or _pick_voice(lang_code, _UNKNOWN_GENDER_FALLBACK)

        # The cache key must include the voice and style. Keying on segment id
        # alone meant a re-run with a different voice silently returned the
        # previous take — the audio never changed however the casting did.
        text = speak_text.get(seg.id, seg.dubbed_text)
        key = instruction_hash(
            f"{seg.edit_instruction or ''}|{voice}|{seg_style}|{text}"
        )
        audio_path = segs_dir / f"s{seg.id:04d}_dubbed_{key}.wav"

        if audio_path.exists() and seg.id not in edit_ids:
            return seg.id, str(audio_path)

        # Male voice names are taken from the published catalogue and not all
        # are confirmed against the API, so fall back rather than fail the run.
        fallbacks = [voice]
        primary = _pick_voice(lang_code, seg.gender or _UNKNOWN_GENDER_FALLBACK)
        for alt in (primary, _DEFAULT_MALE if (seg.gender or "") == "male" else _DEFAULT_FEMALE):
            if alt not in fallbacks:
                fallbacks.append(alt)

        last_exc: Exception | None = None
        for attempt, candidate in enumerate(fallbacks):
            # A fallback take is cached under ITS voice. Saving it under the
            # intended voice's name made every later run reuse the fallback
            # audio while reporting the intended voice.
            out_path = audio_path if not attempt else segs_dir / (
                f"s{seg.id:04d}_dubbed_"
                f"{instruction_hash(f'{seg.edit_instruction or chr(0)}|{candidate}|{seg_style}|{text}')}.wav")
            try:
                audio = tts_bytes(text, lang_code, candidate, seg_style, client)
                with open(out_path, "wb") as f:
                    f.write(audio)
                if attempt:
                    print(f"  ⚠ seg {seg.id}: voice '{voice}' rejected; used '{candidate}'")
                    seg.voice = candidate
                return seg.id, str(out_path)
            except Exception as exc:
                last_exc = exc
                if is_gpt_audio(candidate):
                    gpt_fallbacks.append((seg.id, str(exc)))

        print(f"  ✗ seg {seg.id}: all voices failed ({last_exc})")
        return seg.id, None

    seg_map = {s.id: s for s in segments}
    gpt_fallbacks: list[tuple[int, str]] = []

    def _run(batch: list[Segment]) -> int:
        done = 0
        if not batch:
            return 0
        with ThreadPoolExecutor(max_workers=min(len(batch), 8)) as pool:
            for future in as_completed([pool.submit(_synth_one, s) for s in batch]):
                seg_id, audio_path = future.result()
                if audio_path:
                    seg_map[seg_id].dubbed_audio = audio_path
                    done += 1
        return done

    processed = _run(to_synth)

    # ── Refit: shorten lines whose real audio cannot fit their slot ──────────
    # The translate step only guessed lengths. Now the real clip lengths are
    # known, so measure this voice's actual speaking rate, give each
    # overrunning line a precise character budget, rewrite it shorter and
    # re-synthesize once. This is what keeps the dub from sounding rushed.
    refit_note = _refit_overruns(segments, state.get("audio_path"), _run, lang_code)

    styles: dict[str, int] = {}
    for s in to_synth:
        st = _resolve_style(s.edit_instruction, _line_style(s, global_voice_style))
        styles[st] = styles.get(st, 0) + 1
    notes = [
        f"✓ Synthesized {processed} segments (lang: {lang_code}) — cast: "
        + _cast_summary(segments)
        + " — styles: " + ", ".join(f"{k}×{v}" for k, v in sorted(styles.items()))
    ]
    if swapped:
        notes.append(f"⚠ Sarvam does not speak {lang_code}; used Bodhan instead of "
                     + ", ".join(sorted(set(swapped))))
    gpt_lines = [x for x in to_synth if is_gpt_audio(x.voice)]
    if gpt_lines or gpt_fallbacks:
        why = "; ".join(f"S{i}: {r.split('(')[-1].rstrip(')')}" for i, r in gpt_fallbacks[:6])
        notes.append(f"✓ gpt-audio voiced {len(gpt_lines)} line(s)"
                     + (f"; {len(gpt_fallbacks)} failed its checks and used Bodhan — {why}"
                        if gpt_fallbacks else ", all passed the length and word checks"))
    if refit_note:
        notes.append(refit_note)

    return {
        "segments": [s.to_dict() for s in segments],
        "log": state.get("log", []) + notes,
    }


_REFIT_ROUNDS = int(os.environ.get("REFIT_ROUNDS", "2"))


def _refit_overruns(segments: list[Segment], audio_path: str | None, run,
                    lang_code: str) -> str:
    """
    Rewrite-and-re-voice any line whose real audio would need more than
    MAX_STRETCH speed-up, up to _REFIT_ROUNDS times. Shortening the words is
    what keeps the voice at a natural pace; speeding it up is the last resort.
    """
    from .align import MAX_STRETCH, slot_seconds, trimmed_duration, _audio_len
    from .translate import condense_to_budget
    from .. import tts_rate

    ordered = sorted(segments, key=lambda s: s.start)
    total = _audio_len(audio_path)
    notes: list[str] = []
    cps_by_voice: dict[str, float] = {}

    for rnd in range(1, _REFIT_ROUNDS + 1):
        measured: list[tuple[Segment, float, float]] = []   # seg, clip secs, slot secs
        for i, seg in enumerate(ordered):
            if seg.dubbed_audio and seg.dubbed_text:
                try:
                    measured.append((seg, trimmed_duration(seg.dubbed_audio),
                                     slot_seconds(ordered, i, total)))
                except Exception:
                    pass

        # Characters per second each voice actually produced, ignoring tiny
        # clips whose length is mostly ramp-up. Stored for the next translation.
        rates: dict[str, list[float]] = {}
        for s, d, _ in measured:
            if d >= 0.4 and len(s.dubbed_text) >= 4:
                rates.setdefault(s.voice or "?", []).append(len(s.dubbed_text) / d)
        if rnd == 1:
            tts_rate.record(lang_code, rates)
        flat = sorted(r for rs in rates.values() for r in rs)
        if len(flat) < 2:
            return ""
        cps_all = flat[len(flat) // 2]
        cps_by_voice = {v: sorted(rs)[len(rs) // 2] if len(rs) >= 2 else cps_all
                        for v, rs in rates.items()}

        jobs, over = [], []
        for seg, dur, slot in measured:
            if dur / slot > MAX_STRETCH:
                cps = cps_by_voice.get(seg.voice or "?", cps_all)
                # Aim under the cap so the re-voiced clip fits without stretching.
                budget = int(slot * cps * (0.95 if rnd == 1 else 0.85))
                jobs.append((seg.id, seg.dubbed_text, max(budget, 4)))
                over.append(seg)
        if not jobs:
            notes.append(f"✓ All clips fit their slots at natural pace "
                         f"(measured {cps_all:.1f} chars/s)")
            break

        new = condense_to_budget(jobs)
        if not new:
            why = ("no OPENROUTER_API_KEY" if not os.environ.get("OPENROUTER_API_KEY")
                   else "already as short as they can naturally be")
            notes.append(f"⚠ {len(jobs)} clip(s) still too long for their slot ({why}) "
                         "— they will be sped up")
            break
        redo = []
        for seg in over:
            if seg.id in new:
                seg.dubbed_text = new[seg.id]
                redo.append(seg)
        run(redo)
        notes.append(f"✓ Round {rnd}: shortened and re-voiced {len(redo)}/{len(jobs)} "
                     f"over-long line(s) (measured {cps_all:.1f} chars/s)")
    return "\n".join(notes)


def _cast_summary(segments: list[Segment]) -> str:
    by_spk: dict[int, tuple[str, str, int]] = {}
    for s in segments:
        spk = s.speaker if s.speaker is not None else 0
        g, v, n = by_spk.get(spk, (s.gender or "?", s.voice or "?", 0))
        by_spk[spk] = (g, v, n + 1)
    return ", ".join(f"S{k} {g}→{v} ×{n}" for k, (g, v, n) in sorted(by_spk.items()))


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


# ── Voice profiles: how each catalogue voice actually sounds ─────────────────
# The catalogue only gives name and gender, so "a deeper, older man" was a
# guess. Each voice's pitch is measured once from a short sample and stored;
# the edit planner and casting see deep / medium / high per voice.
_PROFILES_PATH = Path("cache") / "_voice_profiles.json"


def voice_profiles() -> dict:
    try:
        return json.loads(_PROFILES_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def build_voice_profiles(lang_code: str, text: str, sample_dir: Path) -> dict:
    """Synthesize `text` with every voice, measure median pitch, store tiers."""
    import time
    import librosa
    import numpy as np
    profiles = voice_profiles()
    for voice, gender, native in voice_catalogue():
        if voice in profiles:
            continue
        path = sample_dir / f"{voice}_{lang_code}.wav"
        for attempt in range(4):
            try:
                preview_voice(voice, lang_code, text, path)
                break
            except Exception as exc:
                if "429" not in str(exc) or attempt == 3:
                    print(f"  ✗ {voice}: {exc}")
                    break
                time.sleep(4 * (attempt + 1))
        if not path.exists():
            continue
        y, sr = librosa.load(str(path), sr=16000)
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=400, sr=sr)
        v = f0[voiced & ~np.isnan(f0)]
        if v.size < 10:
            continue
        profiles[voice] = {"gender": gender, "native": native,
                           "f0": round(float(np.median(v)), 1)}
        time.sleep(1.5)
    # Tiers within each gender: lowest third deep, highest third high.
    for g in ("male", "female"):
        vs = sorted((p["f0"], v) for v, p in profiles.items() if p["gender"] == g)
        for k, (_, v) in enumerate(vs):
            q = k / max(1, len(vs) - 1)
            profiles[v]["pitch"] = "deep" if q < 1 / 3 else "high" if q > 2 / 3 else "medium"
    _PROFILES_PATH.write_text(json.dumps(profiles, indent=1, ensure_ascii=False),
                              encoding="utf-8")
    return profiles
