"""
Raga Dubbing Studio — the web interface.

    cd dubbing && python -m ui.app

This file is only the page layout and what each button does. The work is
done elsewhere:
  ui/runner.py     builds the pipeline state and streams a run
  ui/views.py      the project card, stage tracker and cast rows (HTML)
  ui/revise.py     natural-language edits (Revise tab)
  ui/exports.py    downloadable files (Export tab)
  ui/library.py    every finished dub (Library tab)
  ui/developer.py  timings, cache, per-line inspector (Developer mode)
  ui/options.py    languages, stages, voice choices
  ui/theme.py      colours and CSS
"""

from ui import startup  # noqa: F401, I001 — must run before pipeline modules load

import gradio as gr

from pipeline.cache import CACHE_ROOT, clip_id, video_id_from_url
from pipeline.edits.edit_planner import apply_plan
from pipeline.graph import EDIT_PIPELINE, FULL_PIPELINE, REVOICE_PIPELINE
from pipeline.state import Segment
from pipeline.voice.styles import BODHAN_STYLES, DEFAULT_STYLE_LABEL
from pipeline.voice.tts import preview_voice

from ui.developer import (
    cache_md,
    config_md,
    inspect_segment,
    rates_md,
    read_debug_log,
    segment_choices,
    speakers_md,
    timings_md,
)
from ui.library import library_choices, load_library_item
from ui.options import (
    AUTO_VOICE,
    DOMAINS,
    EDIT_STAGES,
    FULL_STAGES,
    HAS_SARVAM,
    MAX_CAST_ROWS,
    PREVIEW_TEXT,
    REVOICE_STAGES,
    SUPPORTED_LANGS,
    VOICE_CHOICES,
)
from ui.exports import collect_exports
from ui.revise import plan_changes, render_diff
from ui.runner import base_state, cached_state, parse_time, stream_run, voice_map
from ui.startup import LOG_FILE
from ui.theme import CSS, THEME
from ui.views import (
    cast_updates,
    idle_tracker,
    placeholder_rows,
    render_project_card,
    render_tracker,
    segments_to_df,
)

# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Raga Dubbing Studio") as demo:
    segments_state = gr.State([])
    edit_ops_state = gr.State([])
    video_id_state = gr.State("")
    lang_state = gr.State("")
    timings_state = gr.State({})
    final_state_box = gr.State({})
    before_edit_state = gr.State([])

    # ── Header ────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=True):
        gr.HTML(
            '<div id="brand"><div class="logo">R</div><div><div class="name">'
            'Raga Dubbing Studio</div><div class="tag">Lip-synced dubbing into '
            "Indian languages</div></div></div>"
        )
        dev_mode = gr.Checkbox(label="Developer mode", value=False, scale=0, min_width=170)

    # ── New project + project status ──────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=4, min_width=330, elem_classes="card"):
            gr.Markdown("#### New project")
            url_input = gr.Textbox(
                label="YouTube link", placeholder="https://www.youtube.com/watch?v=…"
            )
            with gr.Row():
                clip_start_in = gr.Textbox(
                    label="Start (optional)", placeholder="0:00", info="m:ss, h:mm:ss or seconds"
                )
                clip_end_in = gr.Textbox(
                    label="End (optional)",
                    placeholder="blank = to the end",
                    info="Only this part is downloaded and dubbed",
                )
            with gr.Row():
                target_lang = gr.Dropdown(
                    choices=list(SUPPORTED_LANGS.keys()), value="Hindi (हिन्दी)", label="Dub into"
                )
                num_speakers_dropdown = gr.Dropdown(
                    choices=["Auto"] + [str(n) for n in range(1, 13)],
                    value="Auto",
                    label="Speakers",
                    info="Set it if you know — most reliable",
                )
            lang_mix = gr.Radio(
                choices=[
                    ("Standard (as before)", "standard"),
                    ("Pure", "pure"),
                    ("Natural spoken", "natural"),
                    ("Heavy mix (Tanglish/Hinglish)", "heavy"),
                ],
                value="standard",
                label="English in the dub",
                info="How many English words stay, the way real speakers talk",
            )
            pipeline_mode = gr.Radio(
                choices=[
                    ("LLM translation (context-aware)", "llm"),
                    ("Indic: Bodhan translation, LLM only for lines it fails", "indic"),
                ],
                value="indic",
                label="Pipeline",
            )
            songs_mode = gr.Radio(
                choices=[
                    ("Keep the original singing", "keep"),
                    ("Re-sing in the dub language (experimental)", "resing"),
                ],
                value="keep",
                label="Songs",
                info="Re-singing composes a new tune (~$0.04 per phrase); the original melody is not kept yet",
            )
            tts_engine = gr.Radio(
                choices=[("Bodhan Indic-Speak", "bodhan"), ("Sarvam Bulbul v3", "sarvam")],
                value="bodhan",
                label="Voice engine",
                info=(
                    "Sarvam: 11 languages (hi bn ta te kn ml mr gu pa or en) — other "
                    "languages use Bodhan automatically. Any voice can still be picked "
                    "per speaker in Cast"
                    if HAS_SARVAM
                    else "Add SARVAM_API_KEY to .env to enable Sarvam"
                ),
                interactive=HAS_SARVAM,
            )
            with gr.Accordion("Advanced", open=False):
                domain = gr.Dropdown(
                    choices=DOMAINS,
                    value="education / lecture",
                    label="Content domain (keeps technical terms)",
                )
                voice_style_dropdown = gr.Dropdown(
                    choices=list(BODHAN_STYLES.keys()),
                    value=DEFAULT_STYLE_LABEL,
                    label="Speaking style",
                )
                voice_name_input = gr.Textbox(
                    label="One voice for everyone (optional)",
                    placeholder="Blank = cast each speaker (see Cast tab)",
                )
            process_btn = gr.Button("▶ Dub video", variant="primary", size="lg")
            gr.Markdown(
                "<small>Re-dubbing a video reuses every finished stage — only "
                "what changed is recomputed.</small>"
            )

        with gr.Column(scale=7, elem_classes="card"):
            project_html = gr.HTML(render_project_card("", None, []))
            tracker_html = gr.HTML(idle_tracker())
            status_md = gr.Markdown("", elem_id="status")

    # ── Workflow tabs ─────────────────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("🎬 Preview"):
            with gr.Row():
                original_video = gr.Video(label="Original", interactive=False)
                dubbed_video = gr.Video(label="Dubbed", interactive=False)
            dubbed_audio = gr.Audio(label="Dubbed soundtrack", interactive=False, type="filepath")

        with gr.Tab("📚 Library"):
            with gr.Row():
                library_pick = gr.Dropdown(
                    label="Dubbed videos — latest dub of each", choices=library_choices(), scale=6
                )
                library_refresh = gr.Button("↻ Refresh", scale=1)
            with gr.Row():
                library_dubbed = gr.Video(label="Dubbed", interactive=False)
                library_original = gr.Video(label="Original", interactive=False)

        with gr.Tab("🎙 Cast"):
            gr.Markdown(
                "Every speaker gets one consistent voice, cast automatically "
                "(gender from pitch, distinct voices). Choose any of the 45 "
                "voices, preview it, then **Re-voice** — the translation is kept."
            )
            cast_labels, cast_voices, cast_previews = [], [], []
            for k in range(MAX_CAST_ROWS):
                with gr.Row(equal_height=True, elem_classes="castrow"):
                    cast_labels.append(
                        gr.Markdown(
                            f"**Speaker {k + 1}** · detected after processing — more "
                            "appear once speakers are detected",
                            visible=k < 1,
                        )
                    )
                    cast_voices.append(
                        gr.Dropdown(
                            choices=VOICE_CHOICES,
                            value=AUTO_VOICE,
                            show_label=False,
                            visible=k < 1,
                            scale=3,
                        )
                    )
                    cast_previews.append(
                        gr.Button("▶ Preview", size="sm", visible=k < 1, scale=0, min_width=110)
                    )
            preview_audio = gr.Audio(label="Voice preview", interactive=False, autoplay=True)
            revoice_btn = gr.Button("🔁 Re-voice with this cast", variant="primary")

        with gr.Tab("📝 Script"):
            gr.Markdown(
                "Who says what, original and dubbed. Click a line to revise it.  \n"
                "<small>＋ recovered from speech the recogniser first skipped · "
                "⚠ transcript uncertain — worth a listen</small>"
            )
            transcript_table = gr.Dataframe(
                headers=["#", "Time", "Speaker", "Voice", "Original", "Dubbed"],
                datatype=["number", "str", "str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                column_widths=["5%", "8%", "10%", "9%", "34%", "34%"],
            )

        with gr.Tab("✏️ Revise"):
            gr.Markdown(
                "Ask for any number of changes in one message — wording, tone, "
                "voices, gender, who said a line, delivery, or leaving a line "
                "undubbed. Each request is listed so nothing is silently skipped.  \n"
                "*e.g. “make the intro more casual, speaker 2 should be an older man "
                "with a deeper voice, line 5 sad, and don’t dub the song at 1:20”*"
            )
            with gr.Row():
                edit_input = gr.Textbox(
                    label="Change request", lines=3, scale=5, placeholder="What should change?"
                )
                with gr.Column(scale=1, min_width=150):
                    parse_btn = gr.Button("Plan changes →")
                    apply_btn = gr.Button("Apply changes ✓", variant="primary")
            parse_status = gr.Markdown("")
            edit_queue_table = gr.Dataframe(
                headers=["Request", "Change", "Applies to", "Details"],
                datatype=["str", "str", "str", "str"],
                interactive=False,
                wrap=True,
                label="Planned changes",
            )
            diff_md = gr.Markdown("")

        with gr.Tab("⬇ Export"):
            gr.Markdown(
                "Lip-synced video (H.264/AAC), the dubbed soundtrack alone, and "
                "subtitles in the dubbed and original languages."
            )
            export_files = gr.File(label="Deliverables", file_count="multiple", interactive=False)

        # ── Developer ─────────────────────────────────────────────────────────
        with gr.Tab("🛠 Developer", visible=False) as dev_tab:
            dev_refresh = gr.Button("↻ Refresh", size="sm")
            with gr.Row():
                with gr.Column():
                    gr.Markdown("##### Stage timings (last run)")
                    dev_timings = gr.Markdown("_No run yet in this session._")
                with gr.Column():
                    gr.Markdown("##### Speakers")
                    dev_speakers = gr.Markdown("_No project loaded._")
            with gr.Accordion("Cache inspector", open=True):
                dev_cache = gr.Markdown("_No project loaded._")
            with gr.Accordion("Line inspector", open=True):
                dev_seg = gr.Dropdown(label="Line", choices=[], interactive=True)
                dev_seg_md = gr.Markdown("_Pick a line._")
                with gr.Row():
                    dev_tts = gr.Audio(label="Raw TTS clip", interactive=False)
                    dev_fit = gr.Audio(label="Fitted to slot", interactive=False)
                    dev_clip = gr.Video(label="First lip-synced shot", interactive=False)
                dev_seg_json = gr.JSON(label="Segment record")
            with gr.Accordion("Pipeline log (last run)", open=False):
                dev_log = gr.Textbox(
                    lines=14, max_lines=30, show_label=False, interactive=False, autoscroll=True
                )
            with gr.Accordion("Server output (live)", open=False):
                debug_live = gr.Checkbox(value=True, label="Auto-refresh every 2 s")
                debug_box = gr.Textbox(
                    lines=18,
                    max_lines=40,
                    interactive=False,
                    autoscroll=True,
                    show_label=False,
                    value=read_debug_log,
                )
                gr.Markdown(f"Full log on the server: `{LOG_FILE}`")
            with gr.Accordion("Measured TTS speaking rates", open=False):
                dev_rates = gr.Markdown(rates_md())
            with gr.Accordion("Configuration", open=False):
                dev_config = gr.Markdown(config_md())

    # ── Event wiring ──────────────────────────────────────────────────────────

    cast_outputs = [
        c for k in range(MAX_CAST_ROWS) for c in (cast_labels[k], cast_voices[k], cast_previews[k])
    ]
    # Every long-running action writes this same set.
    run_outputs = [
        status_md,
        tracker_html,
        project_html,
        dev_log,
        timings_state,
        final_state_box,
        original_video,
        dubbed_video,
        dubbed_audio,
        transcript_table,
        segments_state,
        video_id_state,
        lang_state,
        *cast_outputs,
    ]
    _N_FIXED = 6  # status … final_state_box

    def _queued():
        # Runs instantly, outside the queue: a click always gets an answer,
        # even while another job holds the GPU.
        return "⏳ Queued — starts as soon as the current job finishes (position shown here)."

    def _drive(events, stages, verb, on_done):
        for info in events:
            tracker = render_tracker(
                stages,
                info["timings"],
                info["current"],
                error=info["kind"] == "error",
                elapsed=info["elapsed"],
            )
            if info["kind"] == "error":
                yield (
                    f"**✗ {verb} failed — {info['message']}**  \n<small>Details in "
                    "Developer mode → Server output.</small>",
                    tracker,
                    gr.update(),
                    info["log"],
                    info["timings"],
                    info["final"],
                    *[gr.update()] * (len(run_outputs) - _N_FIXED),
                )
                return
            if info["kind"] == "progress":
                label = dict(stages).get(info["current"], "Finishing")
                yield (
                    f"⏳ {verb} — **{label}**…",
                    tracker,
                    gr.update(),
                    info["log"],
                    info["timings"],
                    gr.update(),
                    *[gr.update()] * (len(run_outputs) - _N_FIXED),
                )
            else:
                yield on_done(info, tracker)

    def _finished(info, tracker, msg, original=None, url=None):
        st = info["final"]
        segs = st.get("segments", [])
        lang = st.get("target_lang")
        seg_objs = [Segment.from_dict(s) for s in segs]
        return (
            f"✅ {msg} in {info['elapsed']:.0f}s — see **Preview**, **Script** and **Export**.",
            tracker,
            render_project_card(st.get("video_id"), lang, segs, url),
            info["log"],
            info["timings"],
            st,
            original if original is not None else gr.update(),
            st.get("final_video"),
            st.get("final_audio"),
            segments_to_df(seg_objs),
            segs,
            st.get("video_id"),
            lang,
            *cast_updates(segs),
        )

    def _on_process(
        cur_vid,
        url,
        c_start,
        c_end,
        lang,
        dom,
        nspk,
        vname,
        vstyle,
        mix,
        engine,
        mode,
        songs,
        *voices,
    ):
        if not url.strip():
            yield (
                "**⚠ Paste a YouTube link first.**",
                idle_tracker(),
                *[gr.update()] * (len(run_outputs) - 2),
            )
            return
        code = SUPPORTED_LANGS.get(lang, "hi")
        try:
            a, b = parse_time(c_start), parse_time(c_end)
        except ValueError as e:
            yield (f"**⚠ {e}**", idle_tracker(), *[gr.update()] * (len(run_outputs) - 2))
            return
        if a is not None and b is not None and b <= a:
            yield (
                "**⚠ End must be after Start.**",
                idle_tracker(),
                *[gr.update()] * (len(run_outputs) - 2),
            )
            return
        vid = clip_id(video_id_from_url(url.strip()), a, b)
        # Voice picks belong to the video they were made for; a new video
        # starts with automatic casting.
        vmap = voice_map(*voices) if vid == cur_vid else {}
        state = base_state(
            vid,
            code,
            dom,
            vname,
            BODHAN_STYLES.get(vstyle, "educational lecture"),
            vmap,
            mix,
            engine,
            mode,
            songs,
        )
        state["url"] = url.strip()
        state["clip_start"], state["clip_end"] = a, b
        state["num_speakers"] = int(nspk) if str(nspk).isdigit() else None
        print(f"\n=== Dub {url.strip()} → {code} (speakers: {nspk})", flush=True)
        yield (
            "⏳ Starting…",
            render_tracker(FULL_STAGES, {}, "download", elapsed=0),
            render_project_card(vid, code, [], url.strip()),
            *[gr.update()] * (len(run_outputs) - 3),
        )
        yield from _drive(
            stream_run(FULL_PIPELINE, state, FULL_STAGES),
            FULL_STAGES,
            "Dubbing",
            lambda info, tr: _finished(
                info, tr, "Dubbed", original=info["final"].get("video_path"), url=url.strip()
            ),
        )

    def _after_run(vid, lang, segs, final_st, timings):
        """Export list + developer panels, refreshed after any run."""
        return (
            collect_exports(vid, lang, segs, final_st),
            timings_md(timings, FULL_STAGES),
            speakers_md(segs),
            cache_md(vid),
            gr.update(choices=segment_choices(segs), value=None),
            rates_md(),
        )

    after_outputs = [export_files, dev_timings, dev_speakers, dev_cache, dev_seg, dev_rates]
    after_inputs = [video_id_state, lang_state, segments_state, final_state_box, timings_state]

    process_btn.click(
        _queued,
        outputs=[status_md],
        queue=False,
        show_progress="hidden",
    ).then(
        _on_process,
        inputs=[
            video_id_state,
            url_input,
            clip_start_in,
            clip_end_in,
            target_lang,
            domain,
            num_speakers_dropdown,
            voice_name_input,
            voice_style_dropdown,
            lang_mix,
            tts_engine,
            pipeline_mode,
            songs_mode,
            *cast_voices,
        ],
        outputs=run_outputs,
        show_progress="minimal",
        show_progress_on=[status_md],
    ).then(_after_run, inputs=after_inputs, outputs=after_outputs, show_progress="hidden")

    def _on_preview(voice, lang):
        if not voice or voice == AUTO_VOICE:
            raise gr.Error("Pick a voice first — 'Auto' has no single voice to preview.")
        code = SUPPORTED_LANGS.get(lang, "hi")
        text = PREVIEW_TEXT.get(code, PREVIEW_TEXT["en"])
        path = CACHE_ROOT / "_voice_previews" / f"{voice}_{code}.wav"
        try:
            return preview_voice(voice, code, text, path)
        except Exception as e:
            raise gr.Error(f"Preview failed: {e}")

    for k in range(MAX_CAST_ROWS):
        cast_previews[k].click(
            _on_preview,
            inputs=[cast_voices[k], target_lang],
            outputs=[preview_audio],
            show_progress="minimal",
        )

    def _on_revoice(segs, vid_id, lang, dom, vname, vstyle, mix, engine, *voices):
        if not segs:
            yield ("**⚠ Dub a video first.**", gr.update(), *[gr.update()] * (len(run_outputs) - 2))
            return
        state = cached_state(
            vid_id,
            SUPPORTED_LANGS.get(lang, "hi"),
            dom,
            vname,
            BODHAN_STYLES.get(vstyle, "educational lecture"),
            voice_map(*voices),
            segs,
            [],
            mix,
            engine,
        )
        if isinstance(state, str):
            yield (f"**{state}**", gr.update(), *[gr.update()] * (len(run_outputs) - 2))
            return
        print(f"\n=== Re-voice {vid_id}: {voice_map(*voices) or 'auto'}", flush=True)
        yield from _drive(
            stream_run(REVOICE_PIPELINE, state, REVOICE_STAGES),
            REVOICE_STAGES,
            "Re-voicing",
            lambda info, tr: _finished(info, tr, "Re-voiced"),
        )

    revoice_btn.click(
        _queued,
        outputs=[status_md],
        queue=False,
        show_progress="hidden",
    ).then(
        _on_revoice,
        inputs=[
            segments_state,
            video_id_state,
            target_lang,
            domain,
            voice_name_input,
            voice_style_dropdown,
            lang_mix,
            tts_engine,
            *cast_voices,
        ],
        outputs=run_outputs,
        show_progress="minimal",
        show_progress_on=[status_md],
    ).then(_after_run, inputs=after_inputs, outputs=after_outputs, show_progress="hidden")

    def _on_nspk(nspk, segs):
        return cast_updates(segs) if segs else placeholder_rows(nspk)

    num_speakers_dropdown.change(
        _on_nspk,
        inputs=[num_speakers_dropdown, segments_state],
        outputs=cast_outputs,
        show_progress="hidden",
    )

    parse_btn.click(
        lambda: "⏳ Reading your request…",
        outputs=[parse_status],
        queue=False,
        show_progress="hidden",
    ).then(
        plan_changes,
        inputs=[edit_input, segments_state, target_lang],
        outputs=[edit_ops_state, edit_queue_table, parse_status],
        show_progress="minimal",
        show_progress_on=[parse_status],
    )

    def _on_apply(plan, segs, vid_id, lang, dom, vname, vstyle, mix, engine, *voices):
        none = [gr.update()] * (len(run_outputs) - 2)
        if not plan or not any(r["actions"] for r in plan.get("requests", [])):
            yield ("**⚠ Nothing to apply — use Plan changes first.**", gr.update(), *none)
            return
        if not segs:
            yield ("**⚠ Dub a video first.**", gr.update(), *none)
            return
        new_segs, vmap, notes, needs_tr = apply_plan(segs, plan, voice_map(*voices))
        state = cached_state(
            vid_id,
            SUPPORTED_LANGS.get(lang, "hi"),
            dom,
            vname,
            BODHAN_STYLES.get(vstyle, "educational lecture"),
            vmap,
            new_segs,
            [],
            mix,
            engine,
        )
        if isinstance(state, str):
            yield (f"**{state}**", gr.update(), *none)
            return
        graph, stages = (
            (EDIT_PIPELINE, EDIT_STAGES) if needs_tr else (REVOICE_PIPELINE, REVOICE_STAGES)
        )
        print(
            f"\n=== Apply plan {vid_id}: {notes} → {'translate' if needs_tr else 'revoice'}",
            flush=True,
        )
        yield from _drive(
            stream_run(graph, state, stages),
            stages,
            "Applying changes",
            lambda info, tr: _finished(info, tr, f"Applied {len(notes)} change(s)"),
        )

    apply_btn.click(
        lambda segs: (_queued(), segs, ""),
        inputs=[segments_state],
        outputs=[status_md, before_edit_state, diff_md],
        queue=False,
        show_progress="hidden",
    ).then(
        _on_apply,
        inputs=[
            edit_ops_state,
            segments_state,
            video_id_state,
            target_lang,
            domain,
            voice_name_input,
            voice_style_dropdown,
            lang_mix,
            tts_engine,
            *cast_voices,
        ],
        outputs=run_outputs,
        show_progress="minimal",
        show_progress_on=[status_md],
    ).then(_after_run, inputs=after_inputs, outputs=after_outputs, show_progress="hidden").then(
        lambda b, a: ("#### What changed\n" + render_diff(b, a)) if b and a else "",
        inputs=[before_edit_state, segments_state],
        outputs=[diff_md],
        show_progress="hidden",
    )

    # Script row click → pre-fill a change request for that line
    def _on_row_select(evt: gr.SelectData, segs):
        if not segs or evt.index is None:
            return gr.update()
        ordered = sorted(segs, key=lambda d: d["start"])
        if evt.index[0] >= len(ordered):
            return gr.update()
        seg = Segment.from_dict(ordered[evt.index[0]])
        return f'Line {seg.id} ("{seg.text[:60]}"): '

    transcript_table.select(
        _on_row_select, inputs=[segments_state], outputs=[edit_input], show_progress="hidden"
    )

    # Developer mode
    dev_mode.change(
        lambda on: gr.update(visible=on),
        inputs=[dev_mode],
        outputs=[dev_tab],
        show_progress="hidden",
    )
    dev_refresh.click(
        lambda vid, segs, timings: (
            cache_md(vid),
            speakers_md(segs),
            timings_md(timings, FULL_STAGES),
            rates_md(),
            config_md(),
            read_debug_log(),
        ),
        inputs=[video_id_state, segments_state, timings_state],
        outputs=[dev_cache, dev_speakers, dev_timings, dev_rates, dev_config, debug_box],
        show_progress="hidden",
    )
    dev_seg.change(
        inspect_segment,
        inputs=[dev_seg, segments_state],
        outputs=[dev_seg_md, dev_tts, dev_fit, dev_clip, dev_seg_json],
        show_progress="hidden",
    )
    debug_timer = gr.Timer(2.0)
    debug_timer.tick(
        lambda on, live: read_debug_log() if on and live else gr.update(),
        inputs=[dev_mode, debug_live],
        outputs=[debug_box],
        show_progress="hidden",
    )

    library_pick.change(
        load_library_item,
        inputs=[library_pick],
        outputs=[library_dubbed, library_original],
        show_progress="hidden",
    )
    library_refresh.click(
        lambda: gr.update(choices=library_choices()),
        outputs=[library_pick],
        queue=False,
        show_progress="hidden",
    )
    # Fresh list on every page load, so new dubs show without a restart.
    demo.load(lambda: gr.update(choices=library_choices()), outputs=[library_pick])


if __name__ == "__main__":
    demo.launch(share=True, show_error=True, theme=THEME, css=CSS)
