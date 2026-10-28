"""
HTML and table pieces of the studio: the project card, the stage tracker,
the script table and the cast rows.
"""

import json

import gradio as gr

from pipeline.cache import get_cache_dir
from pipeline.state import Segment
from pipeline.voice.catalogue import voice_label
from ui.options import AUTO_VOICE, FULL_STAGES, LANG_LABEL, MAX_CAST_ROWS


def render_tracker(
    stages, timings: dict, current: str | None, error: bool = False, elapsed: float | None = None
) -> str:
    chips = []
    for key, label in stages:
        if key in timings:
            cls, sub = "done", f"{timings[key]:.0f}s" if timings[key] >= 1 else "<1s"
        elif key == current:
            cls, sub = ("err", "failed") if error else ("run", "working…")
        else:
            cls, sub = "todo", ""
        chips.append(
            f'<div class="stage {cls}"><div class="dot"></div>'
            f'<div class="lbl">{label}</div><div class="sub">{sub}</div></div>'
        )
    total = f'<div class="elapsed">{elapsed:.0f}s elapsed</div>' if elapsed is not None else ""
    return f'<div class="tracker">{"".join(chips)}</div>{total}'


def idle_tracker() -> str:
    return render_tracker(FULL_STAGES, {}, None)


def _meta(video_id: str, url: str | None = None) -> dict:
    """Title etc. from download; fetched once for videos cached before that existed."""
    if not video_id:
        return {}
    p = get_cache_dir(video_id) / "meta.json"
    if p.exists():
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return {}
    if url:
        try:
            import yt_dlp

            with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
                info = ydl.extract_info(url, download=False) or {}
            meta = {
                "url": url,
                "title": info.get("title"),
                "channel": info.get("channel") or info.get("uploader"),
                "duration": info.get("duration"),
                "thumbnail": info.get("thumbnail"),
            }
            p.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
            return meta
        except Exception:
            pass
    return {}


def render_project_card(
    video_id: str, lang: str | None, segs: list[dict], url: str | None = None
) -> str:
    if not video_id:
        return (
            '<div class="project empty">No project yet — paste a YouTube link and '
            "press <b>Dub video</b>.</div>"
        )
    m = _meta(video_id, url)
    objs = [Segment.from_dict(d) for d in segs]
    speakers = len({(s.speaker or 0) for s in objs})
    speech = sum(s.duration for s in objs)
    dur = m.get("duration")
    thumb = f'<img src="{m["thumbnail"]}" alt="">' if m.get("thumbnail") else ""
    facts = [
        f"→ {LANG_LABEL.get(lang, lang or '?')}",
        f"{speakers} speaker{'s' if speakers != 1 else ''}" if objs else None,
        f"{len(objs)} lines · {speech:.0f}s speech" if objs else None,
        f"{int(dur) // 60}:{int(dur) % 60:02d}" if dur else None,
    ]
    return (
        f'<div class="project">{thumb}<div><div class="ptitle">'
        f'{m.get("title") or video_id}</div><div class="pmeta">'
        f'{m.get("channel") or ""}</div><div class="pfacts">'
        + "".join(f"<span>{f}</span>" for f in facts if f)
        + "</div></div></div>"
    )


def segments_to_df(segments: list[Segment]) -> list[list]:
    return [
        [
            s.id,
            f"{fmt_time(s.start)}",
            f"Speaker {(s.speaker or 0) + 1}",
            voice_label(s.voice),
            ("⚠ " if s.asr_flag == "low" else "＋ " if s.asr_flag == "gap" else "") + s.text,
            "— (left undubbed)" if s.keep_original else (s.dubbed_text or "—"),
        ]
        for s in sorted(segments, key=lambda s: s.start)
    ]


def fmt_time(t: float) -> str:
    return f"{int(t) // 60}:{t % 60:04.1f}"


def cast_from_segments(segs: list[dict]) -> list[dict]:
    out: dict[int, dict] = {}
    for d in segs:
        s = Segment.from_dict(d)
        spk = s.speaker or 0
        row = out.setdefault(
            spk,
            {
                "speaker": spk,
                "gender": s.gender or "?",
                "voice": s.voice,
                "seconds": 0.0,
                "lines": 0,
            },
        )
        row["seconds"] += s.duration
        row["lines"] += 1
        row["voice"] = row["voice"] or s.voice
    return [out[k] for k in sorted(out)]


def cast_updates(segs: list[dict]) -> list:
    """Label + dropdown + preview button per cast row, hiding unused rows."""
    cast = cast_from_segments(segs) if segs else []
    ups = []
    for k in range(MAX_CAST_ROWS):
        if k < len(cast):
            c = cast[k]
            label = (
                f"**Speaker {k + 1}** · {c['gender']} · {c['lines']} lines · {c['seconds']:.0f}s"
            )
            ups += [
                gr.update(value=label, visible=True),
                gr.update(value=c["voice"] or AUTO_VOICE, visible=True),
                gr.update(visible=True),
            ]
        else:
            ups += [gr.update(visible=False)] * 3
    return ups


def placeholder_rows(nspk) -> list:
    """Before processing: as many rows as the speaker count says (1 on Auto)."""
    n = int(nspk) if str(nspk).isdigit() else 1
    note = "" if str(nspk).isdigit() else " — more appear once speakers are detected"
    ups = []
    for k in range(MAX_CAST_ROWS):
        ups += [
            gr.update(
                value=f"**Speaker {k + 1}** · detected after processing{note}", visible=k < n
            ),
            gr.update(visible=k < n),
            gr.update(visible=k < n),
        ]
    return ups
