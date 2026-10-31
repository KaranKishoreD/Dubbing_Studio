"""
Running the pipeline from the studio.

  base_state()    a fresh state for a new project
  cached_state()  a state for re-running part of an existing project
                  (edits, re-voicing) from its cached files
  stream_run()    runs a graph stage by stage, yielding progress for the UI
  parse_time()    "1:02" / "62" → seconds, for the clip range fields
"""

import time

from pipeline.cache import get_cache_dir
from pipeline.state import PipelineState


def base_state(
    video_id,
    target_lang,
    domain,
    voice_name,
    voice_style,
    voice_map,
    lang_mix: str = "standard",
    tts_backend: str = "bodhan",
    translator: str = "indic",
    songs: str = "keep",
) -> PipelineState:
    return {
        "songs": songs,
        "lang_mix": lang_mix,
        "tts_backend": tts_backend,
        "translator": translator,
        "url": "",
        "target_lang": target_lang,
        "domain": domain,
        "video_id": video_id,
        "video_path": None,
        "audio_path": None,
        "audio_hq_path": None,
        "vocals_path": None,
        "background_path": None,
        "voice_ref_path": None,
        "segments": [],
        "final_video": None,
        "final_audio": None,
        "edit_ops": [],
        "log": [],
        "error": None,
        "voice_name": (voice_name or "").strip() or None,
        "voice_style": voice_style,
        "speaker_profiles": {},
        "num_speakers": None,
        "voice_map": voice_map or {},
    }


def cached_state(
    video_id,
    target_lang,
    domain,
    voice_name,
    voice_style,
    voice_map,
    segments_state,
    edit_ops,
    lang_mix: str = "standard",
    tts_backend: str = "bodhan",
) -> PipelineState | str:
    """State for re-running part of the pipeline on a processed video."""
    cache_dir = get_cache_dir(video_id)
    # Prefer the Demucs stems; fall back to the raw-audio copy.
    sep_dir = cache_dir / "separated"
    stems = sorted(sep_dir.glob("*_vocals.wav"))
    vocals = stems[0] if stems else sep_dir / "vocals.wav"
    bed = vocals.with_name(vocals.name.replace("_vocals.wav", "_bed.wav"))
    background = bed if stems and bed.exists() else None
    audio_hq = cache_dir / "audio_hq.wav"
    if not vocals.exists():
        return "✗ Cache missing — process the video again."
    state = base_state(
        video_id, target_lang, domain, voice_name, voice_style, voice_map, lang_mix, tts_backend
    )
    state.update(
        {
            "video_path": str(cache_dir / "video.mp4"),
            "audio_path": str(cache_dir / "audio.wav"),
            "audio_hq_path": str(audio_hq) if audio_hq.exists() else None,
            "vocals_path": str(vocals.resolve()),
            "background_path": str(background.resolve()) if background else None,
            "voice_ref_path": str(cache_dir / "voice_ref.wav"),
            "segments": segments_state,
            "edit_ops": edit_ops,
        }
    )
    return state


def stream_run(graph, state, stages, stop=None):
    """
    Run a graph on a worker thread and yield a progress dict every second:
      {"kind": "progress"|"done"|"error"|"stopped", "current": node or None,
       "timings": {node: seconds}, "log": text, "message": str, "final": state}
    Ticking every second keeps the elapsed time moving during long stages.

    `stop` (a threading.Event) ends the run once the current stage finishes;
    a stage is never cut off halfway, so no half-written file is left to be
    reused later. The last stage always finishes: by then the dub's files
    are being written.
    """
    import queue
    import threading
    import traceback

    q: queue.Queue = queue.Queue()
    keys = [k for k, _ in stages]
    t0 = time.time()

    def work():
        final_state = state
        last = time.time()
        try:
            for event in graph.stream(state):
                node_name, node_output = next(iter(event.items()))
                now = time.time()
                final_state = {**final_state, **node_output}
                q.put(("node", node_name, now - last, final_state))
                last = now
                if node_output.get("error"):
                    q.put(("error", f"{node_name}: {node_output['error']}", 0, final_state))
                    return
                if stop is not None and stop.is_set() and node_name != keys[-1]:
                    q.put(("stopped", node_name, 0, final_state))
                    return
            q.put(("done", None, 0, final_state))
        except Exception as exc:
            traceback.print_exc()
            q.put(("error", f"{type(exc).__name__}: {exc}", 0, final_state))

    threading.Thread(target=work, daemon=True).start()
    timings: dict[str, float] = {}
    final, log = state, ""
    while True:
        try:
            kind, name, secs, st = q.get(timeout=1.0)
        except queue.Empty:
            kind = "tick"
        if kind in ("node", "done", "error", "stopped"):
            final = st
            log = "\n".join(st.get("log", []))
        if kind == "node":
            timings[name] = round(secs, 1)
        current = next((k for k in keys if k not in timings), None)
        info = {
            "kind": "progress",
            "current": current,
            "timings": dict(timings),
            "log": log,
            "elapsed": time.time() - t0,
            "final": final,
            "message": "",
        }
        if kind == "done":
            info.update(kind="done", current=None)
        elif kind == "stopped":
            info.update(kind="stopped", current=None, message=name)
        elif kind == "error":
            info.update(kind="error", message=name)
            print(f"✗ Run failed: {name}", flush=True)
        yield info
        if kind in ("done", "error", "stopped"):
            return


def parse_time(text) -> float | None:
    """'49', '0:49', '1:02:03' → seconds; blank → None."""
    t = str(text or "").strip()
    if not t:
        return None
    try:
        parts = [float(x) for x in t.split(":")]
    except ValueError:
        raise ValueError(f"Can't read the time “{t}” — use m:ss, h:mm:ss or seconds")
    if len(parts) > 3 or any(p < 0 for p in parts):
        raise ValueError(f"Can't read the time “{t}” — use m:ss, h:mm:ss or seconds")
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs
