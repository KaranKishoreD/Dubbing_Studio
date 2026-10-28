"""
The studio's look: Gradio theme and custom CSS.
"""

import gradio as gr

THEME = gr.themes.Soft(
    primary_hue="indigo",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
)


CSS = """
.gradio-container {max-width: 1280px !important; margin: auto;}
#brand {display:flex; align-items:center; gap:14px; padding:6px 2px;}
#brand .logo {width:40px; height:40px; border-radius:10px; display:grid; place-items:center;
  background:linear-gradient(135deg,#4f46e5,#9333ea); color:#fff; font-weight:700; font-size:20px;}
#brand .name {font-size:20px; font-weight:700; letter-spacing:-0.01em;}
#brand .tag {font-size:13px; opacity:.65;}
.card {border:1px solid var(--border-color-primary); border-radius:14px; padding:14px !important;}
.project {display:flex; gap:14px; align-items:center; min-height:84px;}
.project img {width:150px; border-radius:10px; object-fit:cover;}
.project.empty {opacity:.6; font-size:15px;}
.ptitle {font-size:17px; font-weight:650; line-height:1.3;}
.pmeta {font-size:13px; opacity:.65; margin:2px 0 8px;}
.pfacts span {display:inline-block; font-size:12px; padding:3px 9px; margin:0 6px 4px 0;
  border-radius:999px; background:var(--background-fill-secondary);
  border:1px solid var(--border-color-primary);}
.tracker {display:flex; gap:6px; flex-wrap:wrap; margin-top:6px;}
.stage {flex:1 1 92px; min-width:86px; padding:8px 8px 7px; border-radius:10px;
  border:1px solid var(--border-color-primary); text-align:center;}
.stage .dot {width:10px; height:10px; border-radius:50%; margin:0 auto 5px; background:#cbd5e1;}
.stage .lbl {font-size:12.5px; font-weight:600;}
.stage .sub {font-size:11px; opacity:.65; min-height:14px;}
.stage.done {background:rgba(16,185,129,.08); border-color:rgba(16,185,129,.45);}
.stage.done .dot {background:#10b981;}
.stage.run {background:rgba(79,70,229,.08); border-color:#6366f1;}
.stage.run .dot {background:#6366f1; animation:pulse 1s infinite;}
.stage.err {background:rgba(239,68,68,.08); border-color:#ef4444;}
.stage.err .dot {background:#ef4444;}
.elapsed {font-size:12px; opacity:.6; text-align:right; margin-top:4px;}
@keyframes pulse {50% {opacity:.25}}
#status {min-height:24px;}
.castrow {align-items:center;}
"""
