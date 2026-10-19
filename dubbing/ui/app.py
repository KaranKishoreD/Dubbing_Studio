"""
YouTube Dubbing Studio — Gradio UI
Run: python -m ui.app   (from the dubbing/ directory)
"""

import sys
import os
from pathlib import Path

# Ensure project root is on path when run directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Load .env from project root (one level above dubbing/)
from dotenv import load_dotenv
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(dotenv_path=_env_path)

# Fail fast with a clear message if keys are missing
import os as _os
for _key in ("BODHAN_TRANSLATE_KEY", "BODHAN_SPEAK_KEY", "OPENROUTER_API_KEY"):
    if not _os.environ.get(_key):
        raise EnvironmentError(
            f"{_key} not set. Add it to your .env file at: {_env_path}"
        )

import gradio as gr
from pipeline.state import PipelineState, Segment
from pipeline.cache import video_id_from_url, get_cache_dir
from pipeline.graph import FULL_PIPELINE, EDIT_PIPELINE
from pipeline.edit_resolver import resolve_edits
from pipeline.nodes.synthesize import BODHAN_STYLES, DEFAULT_STYLE_LABEL


SUPPORTED_LANGS = {
    "Hindi (हिन्दी)": "hi",
    "Tamil (தமிழ்)": "ta",
    "Telugu (తెలుగు)": "te",
    "Kannada (ಕನ್ನಡ)": "kn",
    "Malayalam (മലയാളം)": "ml",
    "Bengali (বাংলা)": "bn",
    "Marathi (मराठी)": "mr",
    "Gujarati (ગુજરાતી)": "gu",
    "Punjabi (ਪੰਜਾਬੀ)": "pa",
    "Odia (ଓଡ଼ିଆ)": "or",
    "Assamese (অসমীয়া)": "as",
    "Urdu (اردو)": "ur",
    "English": "en",
}

DOMAINS = [
    "education / lecture",
    "technology / CS",
    "general",
    "medicine",
    "law",
    "finance",
]

# Bodhan credit estimate: ₹0.10/min ASR + ₹6/10K chars TTS ≈ ₹2.30 per 5-min video
BODHAN_CREDIT_WARNING = "⚠ Bodhan credit: ₹10 starting balance (~4 full 5-min videos). Cache reuse is free."


# ── Pipeline runners ──────────────────────────────────────────────────────────

def run_full_pipeline(
    url: str, lang_name: str, domain: str,
    voice_name: str = "", voice_style_label: str = DEFAULT_STYLE_LABEL,
    progress=gr.Progress(),
):
    if not url.strip():
        return None, None, _empty_df(), [], "", "⚠ Enter a YouTube URL first."

    video_id = video_id_from_url(url.strip())
    target_lang = SUPPORTED_LANGS.get(lang_name, "hi")
    voice_style = BODHAN_STYLES.get(voice_style_label, "educational lecture")

    state: PipelineState = {
        "url": url.strip(),
        "target_lang": target_lang,
        "domain": domain,
        "video_id": video_id,
        "video_path": None,
        "audio_path": None,
        "vocals_path": None,
        "background_path": None,
        "voice_ref_path": None,
        "segments": [],
        "final_video": None,
        "edit_ops": [],
        "log": [],
        "error": None,
        "voice_name": voice_name.strip() or None,
        "voice_style": voice_style,
    }

    stage_labels = [
        "download", "separate", "transcribe", "translate",
        "synthesize", "align", "lipsync", "assemble",
    ]
    final_state = state

    for i, event in enumerate(FULL_PIPELINE.stream(state)):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        progress((i + 1) / len(stage_labels), desc=f"[{i+1}/{len(stage_labels)}] {node_name}…")
        final_state = {**final_state, **node_output}

        if node_output.get("error"):
            return None, None, _empty_df(), [], "", f"✗ Error at [{node_name}]: {node_output['error']}"

    segs = [Segment.from_dict(s) for s in final_state.get("segments", [])]
    log = "\n".join(final_state.get("log", []))

    return (
        final_state.get("video_path"),
        final_state.get("final_video"),
        _segments_to_df(segs),
        final_state.get("segments", []),  # raw dicts for state
        video_id,
        log,
    )


def parse_edits(user_message: str, segments_state: list):
    if not segments_state:
        return [], _empty_queue(), "⚠ Process a video first."
    if not user_message.strip():
        return [], _empty_queue(), "⚠ Enter an edit instruction."

    segments = [Segment.from_dict(s) for s in segments_state]
    try:
        ops = resolve_edits(user_message.strip(), segments)
    except Exception as e:
        return [], _empty_queue(), f"✗ Parse error: {e}"

    queue_rows = [
        [
            ", ".join(f"S{sid}" for sid in op["segment_ids"]),
            op["instruction"],
            op.get("reason", ""),
        ]
        for op in ops
    ]
    return ops, queue_rows, f"✓ Parsed {len(ops)} edit operation(s). Review below, then Apply."


def apply_edits(
    edit_ops: list,
    segments_state: list,
    video_id: str,
    lang_name: str,
    domain: str,
    voice_name: str = "",
    voice_style_label: str = DEFAULT_STYLE_LABEL,
    progress=gr.Progress(),
):
    if not edit_ops:
        return None, segments_state, _empty_df(), "⚠ No edit operations to apply."
    if not segments_state:
        return None, segments_state, _empty_df(), "⚠ Process a video first."

    target_lang = SUPPORTED_LANGS.get(lang_name, "hi")
    voice_style = BODHAN_STYLES.get(voice_style_label, "educational lecture")
    cache_dir = get_cache_dir(video_id)

    # Reconstruct cached paths (separate_node writes to cache_dir/separated/)
    sep_dir = cache_dir / "separated"
    vocals = sep_dir / "vocals.wav"
    background = sep_dir / "no_vocals.wav"
    if not vocals.exists():
        return None, segments_state, _empty_df(), "✗ Cache missing — re-run full pipeline."

    state: PipelineState = {
        "url": "",
        "target_lang": target_lang,
        "domain": domain,
        "video_id": video_id,
        "video_path": str(cache_dir / "video.mp4"),
        "audio_path": str(cache_dir / "audio.wav"),
        "vocals_path": str(vocals.resolve()),
        "background_path": str(background.resolve()),
        "voice_ref_path": str(cache_dir / "voice_ref.wav"),
        "segments": segments_state,
        "final_video": None,
        "edit_ops": edit_ops,
        "log": [],
        "error": None,
        "voice_name": voice_name.strip() or None,
        "voice_style": voice_style,
    }

    edit_stages = ["translate", "synthesize", "align", "lipsync", "assemble"]
    final_state = state

    for i, event in enumerate(EDIT_PIPELINE.stream(state)):
        node_name = list(event.keys())[0]
        node_output = event[node_name]
        progress((i + 1) / len(edit_stages), desc=f"Re-processing: {node_name}…")
        final_state = {**final_state, **node_output}

        if node_output.get("error"):
            return None, segments_state, _empty_df(), f"✗ Error at [{node_name}]: {node_output['error']}"

    new_segs = final_state.get("segments", segments_state)
    transcript = _segments_to_df([Segment.from_dict(s) for s in new_segs])
    log = "\n".join(final_state.get("log", []))

    return final_state.get("final_video"), new_segs, transcript, f"✓ Edits applied!\n{log}"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _segments_to_df(segments: list[Segment]) -> list[list]:
    return [
        [s.id, f"{s.start:.1f}s", f"{s.end:.1f}s", s.lang,
         s.text[:80] + ("…" if len(s.text) > 80 else ""),
         (s.dubbed_text or "—")[:80]]
        for s in segments
    ]


def _empty_df() -> list[list]:
    return []


def _empty_queue() -> list[list]:
    return []


# ── Gradio Blocks ─────────────────────────────────────────────────────────────

with gr.Blocks(title="YT Dubbing Studio") as demo:

    # ── Hidden state ──────────────────────────────────────────────────────────
    segments_state = gr.State([])
    edit_ops_state = gr.State([])
    video_id_state = gr.State("")

    # ── Header ────────────────────────────────────────────────────────────────
    gr.Markdown("# YouTube Dubbing Studio — Indian Languages")
    gr.Markdown(
        "Paste any YouTube URL → get a lip-synced dubbed video in an Indian language. "
        "Handles Hinglish / code-switched lectures natively. "
        "Prompt-edit any portion in natural language."
    )
    gr.Markdown(BODHAN_CREDIT_WARNING)

    # ── Input row ─────────────────────────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=2):
            url_input = gr.Textbox(
                label="YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
            )
        with gr.Column(scale=1):
            target_lang = gr.Dropdown(
                choices=list(SUPPORTED_LANGS.keys()),
                value="Hindi (हिन्दी)",
                label="Target language",
            )
        with gr.Column(scale=1):
            domain = gr.Dropdown(
                choices=DOMAINS,
                value="education / lecture",
                label="Domain (preserves technical terms)",
            )
        with gr.Column(scale=0, min_width=160):
            process_btn = gr.Button("▶ Process Video", variant="primary", size="lg")

    # ── Voice controls ────────────────────────────────────────────────────────
    gr.Markdown(
        "**Voice:** Auto mode detects speaker pitch per segment and assigns male/female voice accordingly "
        "(handles multi-speaker videos). Override with a specific voice name to force one voice for all segments."
    )
    with gr.Row():
        with gr.Column(scale=1):
            voice_name_input = gr.Textbox(
                label="Voice name override (optional)",
                placeholder="Leave blank = auto pitch-detect  |  e.g. Kavya / Amit / Arun / Sravani",
            )
        with gr.Column(scale=2):
            voice_style_dropdown = gr.Dropdown(
                choices=list(BODHAN_STYLES.keys()),
                value=DEFAULT_STYLE_LABEL,
                label="Speaking style / mood",
            )

    # ── Log ───────────────────────────────────────────────────────────────────
    log_box = gr.Textbox(
        label="Pipeline log",
        lines=4,
        interactive=False,
    )

    # ── Video players ─────────────────────────────────────────────────────────
    with gr.Row():
        original_video = gr.Video(label="Original", interactive=False)
        dubbed_video = gr.Video(label="Dubbed", interactive=False)
    download_btn = gr.File(label="⬇ Download dubbed video", interactive=False, visible=False)

    # ── Transcript ────────────────────────────────────────────────────────────
    with gr.Accordion("Transcript (click a row to pre-fill the edit box)", open=True):
        transcript_table = gr.Dataframe(
            headers=["ID", "Start", "End", "Lang", "Original", "Dubbed"],
            datatype=["number", "str", "str", "str", "str", "str"],
            interactive=False,
            wrap=True,
        )

    gr.Markdown("---")

    # ── Prompt-edit section ───────────────────────────────────────────────────
    gr.Markdown("### ✏ Prompt-Edit")
    gr.Markdown(
        "Describe what to change in plain language. "
        "The resolver finds the right segments automatically — no timestamps needed.\n\n"
        "**Examples:**\n"
        "- *Make the intro more energetic*\n"
        "- *The joke around 45s — translate to English with a British accent*\n"
        "- *Slow down the explanation at 2 minutes, simpler words*\n"
        "- *Make segments 3 and 4 funnier, and translate the outro to French*"
    )

    with gr.Row():
        edit_input = gr.Textbox(
            label="Edit instruction",
            placeholder="Describe what to change…",
            lines=3,
            scale=4,
        )
        with gr.Column(scale=1, min_width=140):
            parse_btn = gr.Button("Parse Edits →", variant="secondary")
            apply_btn = gr.Button("Apply Edits ✓", variant="primary")

    edit_queue_table = gr.Dataframe(
        headers=["Segments", "Instruction", "Reason"],
        datatype=["str", "str", "str"],
        label="Edit queue — review before applying (edits below run in parallel on affected segments only)",
        interactive=True,
        wrap=True,
    )

    edit_status = gr.Textbox(label="Status", interactive=False, lines=3)

    # ── Event wiring ──────────────────────────────────────────────────────────

    def _on_process(url, lang, dom, vname, vstyle, progress=gr.Progress()):
        orig, dubbed, df, segs, vid_id, log = run_full_pipeline(
            url, lang, dom, vname, vstyle, progress
        )
        dl = gr.File(value=dubbed, visible=bool(dubbed))
        return orig, dubbed, df, segs, vid_id, log, dl

    process_btn.click(
        _on_process,
        inputs=[url_input, target_lang, domain, voice_name_input, voice_style_dropdown],
        outputs=[
            original_video, dubbed_video,
            transcript_table, segments_state, video_id_state,
            log_box, download_btn,
        ],
    )

    def _on_parse(msg, segs):
        ops, queue, status = parse_edits(msg, segs)
        return ops, queue, status

    parse_btn.click(
        _on_parse,
        inputs=[edit_input, segments_state],
        outputs=[edit_ops_state, edit_queue_table, edit_status],
    )

    def _on_apply(ops, segs, vid_id, lang, dom, vname, vstyle, progress=gr.Progress()):
        new_video, new_segs, df, status = apply_edits(
            ops, segs, vid_id, lang, dom, vname, vstyle, progress
        )
        dl = gr.File(value=new_video, visible=bool(new_video))
        return new_video, new_segs, df, status, dl

    apply_btn.click(
        _on_apply,
        inputs=[edit_ops_state, segments_state, video_id_state, target_lang, domain,
                voice_name_input, voice_style_dropdown],
        outputs=[dubbed_video, segments_state, transcript_table, edit_status, download_btn],
    )

    # Pre-fill edit box when a transcript row is clicked
    def _on_row_select(evt: gr.SelectData, segs):
        if not segs or evt.index is None:
            return ""
        row_idx = evt.index[0]
        if row_idx >= len(segs):
            return ""
        seg = Segment.from_dict(segs[row_idx])
        return f'Edit segment {seg.id} (~{seg.start:.0f}s): "{seg.text[:60]}…"'

    transcript_table.select(
        _on_row_select,
        inputs=[segments_state],
        outputs=[edit_input],
    )


if __name__ == "__main__":
    demo.launch(share=True, show_error=True, theme=gr.themes.Soft())
