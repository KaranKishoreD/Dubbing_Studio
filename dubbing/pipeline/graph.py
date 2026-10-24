from langgraph.graph import StateGraph, END
from .state import PipelineState
from .nodes.download import download_node
from .nodes.separate import separate_node
from .nodes.transcribe import transcribe_node
from .nodes.diarize import diarize_node
from .nodes.refine import refine_node
from .nodes.translate import translate_node
from .nodes.synthesize import synthesize_node
from .nodes.align import align_node
from .nodes.lipsync import lipsync_node
from .nodes.assemble import assemble_node


def _build(entry: str, edges: list[tuple[str, str]], nodes: dict) -> object:
    g = StateGraph(PipelineState)
    for name, fn in nodes.items():
        g.add_node(name, fn)
    g.set_entry_point(entry)
    for src, dst in edges:
        g.add_edge(src, dst if dst != "END" else END)
    return g.compile()


_SHARED_NODES = {
    "translate": translate_node,
    "synthesize": synthesize_node,
    "align": align_node,
    "lipsync": lipsync_node,
    "assemble": assemble_node,
}

_SHARED_EDGES = [
    ("translate", "synthesize"),
    ("synthesize", "align"),
    ("align", "lipsync"),
    ("lipsync", "assemble"),
    ("assemble", "END"),
]

# Full pipeline: download → separate → transcribe → diarize → translate → … → assemble
FULL_PIPELINE = _build(
    entry="download",
    edges=[
        ("download", "separate"),
        ("separate", "transcribe"),
        ("transcribe", "diarize"),
        ("diarize", "refine"),
        ("refine", "translate"),
        *_SHARED_EDGES,
    ],
    nodes={
        "download": download_node,
        "separate": separate_node,
        "transcribe": transcribe_node,
        "diarize": diarize_node,
        "refine": refine_node,
        **_SHARED_NODES,
    },
)

# Edit pipeline: skips download/separate/transcribe/diarize — speaker and gender
# are already persisted on each segment, so edits re-run from translate onward.
EDIT_PIPELINE = _build(
    entry="translate",
    edges=_SHARED_EDGES,
    nodes=_SHARED_NODES,
)

# Re-voice: new voice casting only — translations are kept, so it starts at
# synthesis and re-runs everything downstream of the audio.
REVOICE_PIPELINE = _build(
    entry="synthesize",
    edges=_SHARED_EDGES[1:],
    nodes={k: v for k, v in _SHARED_NODES.items() if k != "translate"},
)
