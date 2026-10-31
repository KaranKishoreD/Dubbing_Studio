"""
Song mode: sung lyrics are re-sung in the target language and blended in.

  translate step   plan(): for each line the song detector flagged, one LLM
                   call decides which of its words are actually SUNG (audio
                   alone mistook dramatic dialogue for singing), writes a
                   singable lyric with the same syllable count and a short
                   style note. The line is split at the word timings: the
                   sung part becomes a `sung` segment (kept out of speech
                   dubbing), the spoken rest is dubbed as usual.
  assemble step    render(): Lyria 3 (OpenRouter) sings the lyric in the
                   measured style/key/tempo/voice; its vocal is separated
                   (BS-RoFormer), long held notes are shortened inside the
                   vowel, the rest gently fitted to the original's length,
                   and laid over the original score at the original vocal's
                   level. A result Whisper cannot follow keeps the original.

SONG_MODE=1 enables it; at most _MAX_PHRASES sung phrases per video.
Cost per phrase: one small LLM call (shared) + one Lyria clip ($0.04).
"""

import base64
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import numpy as np

from ..cache import get_cache_dir
from ..state import Segment

ENABLED = os.environ.get("SONG_MODE", "1") == "1"
# A whole song must be re-sung, not just some of its lines: one line left in
# the original language in the middle of a song sounds broken.
_MAX_PHRASES = int(os.environ.get("SONG_MAX_PHRASES", "12"))
_LYRIA = os.environ.get("SONG_MODEL", "google/lyria-3-clip-preview")
_MIN_MATCH = 0.6
_SEP_DIR = Path(__file__).resolve().parents[2] / "Separator"
_SEP_PY = _SEP_DIR / ".venv-sep" / "bin" / "python"
_SEP_MODEL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"


def available() -> bool:
    return ENABLED and bool(os.environ.get("OPENROUTER_API_KEY"))


# ── Planning (translate step) ────────────────────────────────────────────────

_PLAN_SYSTEM = """\
You are a music editor for a {lang} film dub. Each item is a line that an
audio detector thinks may contain SINGING, with its words, their times and
a per-word "held" score (share of sustained notes; dramatic speech can be
high too). Using the words' meaning and the video context, decide which
consecutive words are actually SUNG lyrics. Sung parts are lyrical phrases
(repetition, rhyme, poetic or emotive words). A character's name or title,
a threat, a question or any other dialogue is SPOKEN even when shouted or
stretched dramatically. In a song performance (a singing show, a music
video) the lyric lines are all sung; only a host's or judge's comments are
spoken — in film trailers a sung phrase is often followed
by the hero's name spoken over the music; that name is NOT part of the
song. When unsure about a word, it is spoken: leave it out of the range.

Return one item for EVERY line given. Lines of a song performance are sung
unless they are clearly a host's or judge's spoken comment.

For each item return:
- "sung": [first_word_index, last_word_index] of the sung words, or null
  if nothing in the line is sung.
- "lyric": those sung words TRANSLATED into {lang} — written in {lang}
  script, never in the original language — as a singable lyric: same
  meaning, natural sung {lang}, about the same number of syllables, long
  open vowels where the original holds a note. Keep it TASTEFUL: romance
  and longing are fine, but nothing sexual, suggestive or violent (the
  singing model refuses such lyrics) — soften the imagery, keep the feeling.
- "style": a short description of how the original is sung — musical
  terms only (mood, energy, tempo feel, e.g. "gentle, melodious, soulful"),
  no names of people, characters or films.

Return ONLY JSON: [{{"id": <int>, "sung": [a, b] | null, "lyric": "...", "style": "..."}}]
"""


def plan(segments, sung_ids: dict, state, lang_code: str) -> tuple[list, str]:
    """
    Split flagged lines into sung pieces (sung=True, keep_original=True,
    dubbed_text=lyric) and spoken pieces (to be dubbed). Returns the new
    segments added and a log note. Lines left unsplit keep the original.
    """
    from ..text.languages import LANG_NAMES
    from ..text.llm import TRANSLATE_MODEL, openrouter_client, parse_items

    for s in segments:
        # Lines recovered by gap filling have text but no word timings, and
        # would otherwise be skipped: spread their words evenly over the line.
        if s.id in sung_ids and not s.words and s.text.split():
            ws, n = s.text.split(), len(s.text.split())
            s.words = [
                {
                    "word": " " + w,
                    "start": s.start + s.duration * k / n,
                    "end": s.start + s.duration * (k + 1) / n,
                }
                for k, w in enumerate(ws)
            ]
    flagged = [s for s in segments if s.id in sung_ids and s.words and not s.edit_instruction]
    flagged = sorted(flagged, key=lambda s: s.start)[:_MAX_PHRASES]
    if not flagged:
        return [], ""
    held = _word_held(flagged, state["vocals_path"])
    items = [
        {
            "id": s.id,
            "text": s.text,
            "words": [
                {
                    "i": k,
                    "w": w.get("word", "").strip(),
                    "t": round(w["start"], 2),
                    "held": held.get((s.id, k), 0.0),
                }
                for k, w in enumerate(s.words)
            ],
        }
        for s in flagged
    ]
    lang = LANG_NAMES.get(lang_code, lang_code)
    try:
        r = openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            temperature=0.3,
            messages=[
                {"role": "system", "content": _PLAN_SYSTEM.format(lang=lang)},
                {
                    "role": "user",
                    "content": json.dumps(
                        {"video": _title(state), "lines": items}, ensure_ascii=False
                    ),
                },
            ],
        )
        out = {
            it["id"]: it
            for it in parse_items(r.choices[0].message.content.strip())
            if isinstance(it, dict) and "id" in it
        }
    except Exception as exc:
        print(f"  ⚠ song planning skipped: {exc}")
        return [], ""
    added, sung_n, spoken_n = [], 0, 0
    next_id = max(s.id for s in segments) + 1
    from ..text.languages import script_ok

    for s in flagged:
        it = out.get(s.id) or {}
        rng, lyric = it.get("sung"), str(it.get("lyric") or "").strip()
        # A lyric in the wrong language would be sung wrong and wasted.
        if rng and lyric and not script_ok(lyric, lang_code, latin=True):
            print(f"  ⚠ song mode #{s.id}: lyric not in {lang} — original singing kept")
            continue
        if not rng or not lyric:
            s.keep_original = False  # nothing sung: dub as speech
            spoken_n += 1
            continue
        words = list(s.words)  # the first piece overwrites s.words
        a, b = max(0, int(rng[0])), min(len(words) - 1, int(rng[1]))
        pieces = [(0, a, False), (a, b + 1, True), (b + 1, len(words), False)]
        pieces = [(x, y, sung) for x, y, sung in pieces if y > x]
        for n, (x, y, sung) in enumerate(pieces):
            ws = words[x:y]
            seg = s if n == 0 else s.__class__.from_dict(s.to_dict())
            if n:
                seg.id, next_id = next_id, next_id + 1
                added.append(seg)
            seg.start, seg.end = round(ws[0]["start"], 3), round(ws[-1]["end"], 3)
            seg.words, seg.text = ws, "".join(w.get("word", "") for w in ws).strip()
            seg.sung, seg.keep_original = sung, sung
            seg.dubbed_text = lyric if sung else None
            seg.song_style = str(it.get("style") or "")[:120] if sung else None
            seg.song_gender = _line_gender(seg, state) if sung else None
            sung_n += sung
    note = f"✓ Song mode: {sung_n} sung phrase(s) will be re-sung in {lang}" + (
        f"; {spoken_n} flagged line(s) were dialogue, dubbed as speech" if spoken_n else ""
    )
    return added, note


def _line_gender(seg, state) -> str | None:
    """Gender of the face clearly singing this line (lines of two singers can
    share a speaker label in a duet); the speaker's gender otherwise."""
    try:
        from ..cache import get_cache_dir
        from ..video import faces as F

        p = get_cache_dir(state["video_id"]) / "faces.npz"
        if p.exists():
            a = dict(np.load(p))
            if "male" in a:
                f = F.talking_faces([seg], a).get(seg.id)
                if f is not None:
                    v = a["male"][F._identities(a) == f]
                    v = v[np.isfinite(v)]
                    if len(v) >= 3 and abs(v.mean() - 0.5) >= 0.25:
                        return "male" if v.mean() > 0.5 else "female"
    except Exception:
        pass
    return seg.gender


def _word_held(segs, vocals_path: str) -> dict:
    import librosa

    y, _ = librosa.load(vocals_path, sr=16000, mono=True)
    out = {}
    for s in segs:
        for k, w in enumerate(s.words):
            clip = y[int(w["start"] * 16000) : int(w["end"] * 16000)]
            if len(clip) < 3200:
                continue
            f0, v, _ = librosa.pyin(
                clip, fmin=70, fmax=700, sr=16000, frame_length=1024, hop_length=160
            )
            st = 12 * np.log2(np.where(v, f0, np.nan) / 100)
            held = sum(
                1
                for i in range(len(st) - 15)
                if np.all(np.isfinite(st[i : i + 15]))
                and np.nanmax(st[i : i + 15]) - np.nanmin(st[i : i + 15]) < 0.6
            )
            out[(s.id, k)] = round(held / max(1, int(np.sum(v))), 2)
    return out


def _title(state) -> str:
    from ..cache import get_cache_dir

    try:
        m = json.loads((get_cache_dir(state["video_id"]) / "meta.json").read_text(encoding="utf-8"))
        return " — ".join(x for x in (m.get("title"), m.get("channel")) if x)
    except Exception:
        return ""


# ── Rendering (assemble step) ────────────────────────────────────────────────


def render(seg, state, cache_dir: Path, lang_code: str, sr: int) -> np.ndarray | None:
    """The sung lyric as a mono vocal at `sr`, fitted to seg's length, or None."""
    import librosa
    import soundfile as sf

    key = hashlib.sha1(
        f"{seg.dubbed_text}|{seg.song_style}|{seg.start:.2f}|{seg.end:.2f}|{lang_code}".encode()
    ).hexdigest()[:12]
    work = cache_dir / "song" / key
    work.mkdir(parents=True, exist_ok=True)
    done = work / "fitted.wav"
    if done.exists():
        y, s = sf.read(str(done), dtype="float32")
        return librosa.resample(y, orig_sr=s, target_sr=sr) if s != sr else y
    if (work / "failed").exists():
        return None
    try:
        voc = _sing(seg, state, work, lang_code)
        fitted = _fit(voc, seg, work, lang_code)
        if fitted is not None:
            fitted = _as_singer(fitted, seg, state, work, lang_code)
    except Exception as exc:
        print(f"  ⚠ song mode seg {seg.id}: {exc} — original singing kept")
        (work / "failed").write_text(str(exc))
        return None
    if fitted is None:
        (work / "failed").write_text("not intelligible")
        return None
    sf.write(str(done), fitted, 44100)
    return librosa.resample(fitted, orig_sr=44100, target_sr=sr) if sr != 44100 else fitted


def _as_singer(line: np.ndarray, seg, state, work: Path, lang_code: str) -> np.ndarray:
    """
    Re-voice Lyria's singer as the ORIGINAL singer of this line (Seed-VC's
    singing model keeps Lyria's melody exactly), so every line sounds like
    the same performers instead of a different AI voice per clip. Kept only
    if the words stay clear; otherwise Lyria's own voice is used.
    """
    import librosa
    import soundfile as sf

    from ..voice import voice_clone as vcl

    if not vcl.available():
        return line
    SR = 44100
    voc, vsr = sf.read(state["vocals_path"], dtype="float32", always_2d=True)
    voc = voc.mean(axis=1)
    # Reference: this line's own singing, then the same singer's other sung lines.
    parts, kept = [], 0.0
    others = [
        s
        for s in (state.get("segments") or [])
        if s.get("sung") and s.get("speaker") == seg.speaker and s.get("id") != seg.id
    ]
    for a, b in [(seg.start, seg.end)] + [(s["start"], s["end"]) for s in others]:
        parts.append(voc[int(a * vsr) : int(b * vsr)])
        kept += b - a
        if kept >= 25:
            break
    ref = work / "singer_ref.wav"
    sf.write(str(ref), np.concatenate(parts)[: int(25 * vsr)], vsr)
    src = work / "lyria_line.wav"
    sf.write(str(src), line, SR)
    out = work / "as_singer.wav"
    res = vcl._run(
        [{"source": str(src.resolve()), "target": str(ref.resolve()), "out": str(out.resolve())}],
        work,
        {"SEEDVC_SINGING": "1"},
        "singer",
    )
    if res.get(str(out.resolve())) != "ok" or not out.exists():
        print(f"  ⚠ song mode seg {seg.id}: singer conversion failed — Lyria's voice kept")
        return line
    y, ysr = sf.read(str(out), dtype="float32", always_2d=True)
    y = y.mean(axis=1)
    if ysr != SR:
        y = librosa.resample(y, orig_sr=ysr, target_sr=SR)
    y = np.pad(y, (0, max(0, len(line) - len(y))))[: len(line)]
    if _clarity(y, SR, seg.dubbed_text, lang_code) < _MIN_MATCH:
        print(
            f"  ⚠ song mode seg {seg.id}: words blurred in the singer's voice — Lyria's voice kept"
        )
        return line
    print(f"  ♪ seg {seg.id}: sung in the original singer's voice")
    return y.astype(np.float32)


def _clarity(y: np.ndarray, sr: int, lyric: str, lang_code: str) -> float:
    import difflib

    import librosa

    from ..steps.transcribe import _whisper

    y16 = librosa.resample(y, orig_sr=sr, target_sr=16000)
    heard = "".join(
        s.text for s in _whisper()[0].transcribe(y16, language=lang_code, beam_size=5)[0]
    )
    norm = lambda t: re.sub(r"[\W_]+", "", t)
    return difflib.SequenceMatcher(None, norm(lyric), norm(heard)).ratio()


def _voice(seg, vocals_path: str) -> tuple[str, str, str]:
    """(gender, low note, high note) of the original singer in this span."""
    import librosa

    y, _ = librosa.load(vocals_path, sr=16000, offset=seg.start, duration=max(0.5, seg.duration))
    f0, v, _ = librosa.pyin(y, fmin=70, fmax=800, sr=16000, frame_length=1024, hop_length=160)
    f = f0[v] if v.any() else np.array([200.0])
    gender = seg.gender or ("female" if np.median(f) > 200 else "male")
    return (
        gender,
        librosa.hz_to_note(np.percentile(f, 10)),
        librosa.hz_to_note(np.percentile(f, 90)),
    )


def _music(audio_path: str, a: float, b: float) -> tuple[int, str]:
    import librosa

    y, sr = librosa.load(audio_path, sr=22050, offset=max(0, a - 4), duration=(b - a) + 8)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    ch = librosa.feature.chroma_cqt(y=y, sr=sr).mean(1)
    names = ["C", "C-sharp", "D", "E-flat", "E", "F", "F-sharp", "G", "A-flat", "A", "B-flat", "B"]
    maj = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    mnr = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
    best = max(
        [(np.corrcoef(np.roll(maj, k), ch)[0, 1], f"{names[k]} major") for k in range(12)]
        + [(np.corrcoef(np.roll(mnr, k), ch)[0, 1], f"{names[k]} minor") for k in range(12)]
    )
    return int(round(float(np.atleast_1d(tempo)[0]))), best[1]


def _sing(seg, state, work: Path, lang_code: str) -> Path:
    """Lyria clip → separated vocal (wav path)."""
    from ..text.languages import LANG_NAMES

    lang = LANG_NAMES.get(lang_code, lang_code)
    gender, lo, hi = _voice(seg, state["vocals_path"])
    gender = getattr(seg, "song_gender", None) or gender
    tempo, key = _music(state.get("audio_hq_path") or state["audio_path"], seg.start, seg.end)
    secs = max(1.5, seg.duration)
    plain = (
        f"A short {lang} song phrase in {key}. A single {gender} voice sings this "
        f"line once, clearly, starting right away; soft accompaniment.\n\n{seg.dubbed_text}"
    )
    prompt = (
        f"A {lang} film song phrase in {key}, around {tempo} BPM. "
        f"Mood and delivery: {seg.song_style or 'melodious'}. "
        f"A single {gender} voice in the range {lo} to {hi} sings it clearly, every word "
        f"well pronounced; no chorus, no shouting. Gentle accompaniment underneath. "
        f"The singer sings this one line once, taking about {secs:.0f} seconds, starting "
        f"right away; then the music continues without vocals.\n\n{seg.dubbed_text}"
    )
    # A content-policy refusal (e.g. a style note naming a film) is retried
    # once with a plain prompt: lyric, voice and key only.
    try:
        chunks = _lyria(prompt)
    except RuntimeError as exc:
        if "PROHIBITED" not in str(exc) and "policy" not in str(exc):
            raise
        # Usually the lyric's imagery, not the style: soften the lyric
        # (same meaning and length) and try once more.
        softer = _soften(seg.dubbed_text, lang)
        print(f"  ⚠ song mode seg {seg.id}: Lyria refused — retrying with a softer lyric")
        if softer:
            seg.dubbed_text = softer
        chunks = _lyria(plain.rsplit("\n\n", 1)[0] + "\n\n" + seg.dubbed_text)
    mp3 = work / "lyria.mp3"
    mp3.write_bytes(b"".join(base64.b64decode(c) for c in chunks))
    wav = work / "lyria.wav"
    subprocess.run(
        ["ffmpeg", "-loglevel", "error", "-y", "-i", str(mp3), "-ar", "44100", str(wav)], check=True
    )
    return _separate(wav, work)


def _soften(lyric: str, lang: str) -> str | None:
    """Same lyric, family-friendly imagery, same length and meaning."""
    from ..text.llm import TRANSLATE_MODEL, openrouter_client

    try:
        r = openrouter_client().chat.completions.create(
            model=TRANSLATE_MODEL,
            temperature=0.3,
            messages=[
                {
                    "role": "user",
                    "content": f"Rewrite this {lang} song lyric so it is clean and family-friendly — "
                    "romantic but with no sexual, suggestive or violent imagery — keeping "
                    "the feeling, the language and about the same number of syllables. "
                    f"Reply with the lyric only.\n\n{lyric}",
                }
            ],
        )
        out = (r.choices[0].message.content or "").strip().strip('"')
        return out or None
    except Exception:
        return None


def _lyria(prompt: str) -> list[str]:
    """Base64 audio chunks for one Lyria clip; raises on refusal/no audio."""
    import requests

    r = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        timeout=600,
        stream=True,
        headers={"Authorization": "Bearer " + os.environ["OPENROUTER_API_KEY"]},
        json={
            "model": _LYRIA,
            "modalities": ["text", "audio"],
            "stream": True,
            "messages": [{"role": "user", "content": prompt}],
        },
    )
    if r.status_code != 200:
        raise RuntimeError(f"Lyria {r.status_code}: {r.text[:200]}")
    chunks = []
    for raw in r.iter_lines():
        if not raw.startswith(b"data: ") or raw == b"data: [DONE]":
            continue
        d = json.loads(raw[6:])
        if "error" in d:
            raise RuntimeError(f"Lyria: {d['error']}")
        for ch in d.get("choices", []):
            a = (ch.get("delta") or {}).get("audio") or {}
            if a.get("data"):
                chunks.append(a["data"])
    if not chunks:
        raise RuntimeError("Lyria returned no audio")
    return chunks


def _separate(wav: Path, work: Path) -> Path:
    """Vocal stem of `wav` (BS-RoFormer in its own venv; Demucs if unavailable)."""
    work, wav = work.resolve(), wav.resolve()  # the separator runs in its own folder
    if _SEP_PY.exists():
        code = (
            "from audio_separator.separator import Separator;"
            f"s=Separator(output_dir={str(work)!r},model_file_dir='./models',output_format='WAV');"
            f"s.load_model({_SEP_MODEL!r});print(s.separate({str(wav)!r}))"
        )
        subprocess.run(
            [str(_SEP_PY), "-c", code], cwd=str(_SEP_DIR), check=True, capture_output=True
        )
        found = list(work.glob("*(Vocals)*.wav"))
        if found:
            return found[0]
    subprocess.run(
        [
            "python",
            "-m",
            "demucs.separate",
            "-n",
            "htdemucs",
            "--two-stems",
            "vocals",
            "-o",
            str(work),
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    return next(work.glob("htdemucs/*/vocals.wav"))


def _fit(voc: Path, seg, work: Path, lang_code: str) -> np.ndarray | None:
    """
    The sung lyric, fitted to the original's length: long held notes are
    shortened inside the vowel (crossfade), then a gentle overall stretch.
    None if Whisper cannot follow the words.
    """
    import difflib

    import librosa
    import pyrubberband as pyrb

    from ..steps.transcribe import _whisper

    SR = 44100
    y, _ = librosa.load(str(voc), sr=SR, mono=True)
    y16 = librosa.resample(y, orig_sr=SR, target_sr=16000)
    segs, _ = _whisper()[0].transcribe(y16, language=lang_code, beam_size=5, word_timestamps=True)
    words = [(w.start, w.end, w.word) for s in segs for w in s.words]
    if not words:
        return None
    norm = lambda t: re.sub(r"[\W_]+", "", t)
    target = norm(seg.dubbed_text)
    # The first stretch of words that matches the lyric best (Lyria may sing it twice).
    best, span = 0.0, (0, len(words))
    for i in range(len(words)):
        for j in range(i + 1, min(len(words), i + 25) + 1):
            r = difflib.SequenceMatcher(
                None, target, norm("".join(w[2] for w in words[i:j]))
            ).ratio()
            if r > best + 1e-6:
                best, span = r, (i, j)
    if best < _MIN_MATCH:
        print(f"  ⚠ song mode seg {seg.id}: sung lyric unclear ({best:.0%}) — original kept")
        return None
    ws = words[span[0] : span[1]]
    a0, a1 = max(0.0, ws[0][0] - 0.06), ws[-1][1] + 0.15
    parts = [[max(a0, w[0]), w[1]] for w in ws]
    parts[0][0], parts[-1][1] = a0, a1
    goal = max(0.8, seg.duration)
    # Shorten the longest held words (keep 0.35 s of each end) until within 25%.
    pieces = [y[int(a * SR) : int(b * SR)] for a, b in parts]
    x = int(0.08 * SR)
    while sum(len(p) for p in pieces) / SR > goal * 1.25:
        k = int(np.argmax([len(p) for p in pieces]))
        p, keep = pieces[k], int(0.35 * SR)
        excess = int(sum(len(q) for q in pieces) - goal * 1.25 * SR)
        cut = min(excess, len(p) - 2 * keep - x)
        if cut <= int(0.05 * SR):
            break
        head, tail = (
            p[: keep + (len(p) - 2 * keep - cut) // 2],
            p[-(keep + (len(p) - 2 * keep - cut) // 2) :],
        )
        pieces[k] = np.concatenate(
            [
                head[:-x],
                head[-x:] * np.linspace(1, 0, x) + tail[:x] * np.linspace(0, 1, x),
                tail[x:],
            ]
        )
    line = np.concatenate(pieces)
    rate = len(line) / SR / goal
    if abs(rate - 1) > 0.02:
        line = pyrb.time_stretch(line, SR, rate).astype(np.float32)
    f = int(0.06 * SR)
    line[:f] *= np.linspace(0, 1, f)
    line[-f:] *= np.linspace(1, 0, f)
    print(f"  ♪ seg {seg.id}: sung lyric {best:.0%} clear, fitted ×{rate:.2f}")
    return line.astype(np.float32)


def mark_songs(segments: list[Segment], state) -> str:
    """Sung lines keep the original audio (song.py); edits can still dub them."""
    from ..music import song_detect as song

    vocals = state.get("vocals_path")
    if not song.ENABLED or not vocals or not os.path.exists(vocals):
        return ""
    try:
        sung = song.sung_lines(segments, vocals, get_cache_dir(state["video_id"]))
    except Exception as exc:
        print(f"  ⚠ song detection skipped: {exc}")
        return ""
    for s in segments:
        if s.id in sung and not s.edit_instruction:
            s.keep_original = True
    if not sung:
        return ""
    note = f"✓ {len(sung)} line(s) sound sung: " + ", ".join(f"#{i}" for i in sorted(sung))
    # Song mode: split sung lyrics from dialogue and plan the target-language
    # lyric; anything not planned keeps the original singing.
    from ..music import song_mode

    if (state.get("songs") or "keep") == "keep":
        return note + " — kept as the original singing (Songs: keep)"
    if song_mode.available() and not any(s.sung for s in segments):
        added, plan_note = song_mode.plan(segments, sung, state, state["target_lang"].split("-")[0])
        segments.extend(added)
        segments.sort(key=lambda s: s.start)
        if plan_note:
            note += "\n" + plan_note
    return note
