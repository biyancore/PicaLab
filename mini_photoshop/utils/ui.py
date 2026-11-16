"""
Komponen UI & tema visual PicaLab (Streamlit + custom CSS).

Berisi: inject_css, hero header, metric cards, slider pembanding before/after,
histogram interaktif (Altair), empty state, dan chip/pill status.
Tidak ada logika pemrosesan citra di sini.
"""

import html

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from utils import helpers
from utils.pipeline import histogram_table

# Palet histogram (divalidasi terhadap surface gelap #12151C, lolos kontras 3:1)
CHANNEL_COLORS = {"Red": "#F04452", "Green": "#22A06B", "Blue": "#5B7CFA"}
NEUTRAL = "#8A91A5"

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@500;600&display=swap');

:root {
  --pl-bg: #0B0D12; --pl-panel: #12151C; --pl-panel-2: #171B24; --pl-border: #232836;
  --pl-text: #E8EAF0; --pl-muted: #8A91A5; --pl-accent: #FF6A4D; --pl-accent-2: #7C8CFF;
  --pl-ok: #3DDC97; --pl-warn: #FFB547; --pl-err: #FF5C7A; --pl-radius: 14px;
}
html, body, [class*="css"], .stMarkdown, .stText, button, input, textarea, select, label {
  font-family: 'Plus Jakarta Sans', system-ui, -apple-system, 'Segoe UI', sans-serif !important;
}
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(1200px 500px at 85% -10%, rgba(124,140,255,.10), transparent 60%),
    radial-gradient(900px 400px at -10% 0%, rgba(255,106,77,.08), transparent 55%),
    var(--pl-bg);
}
[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1500px; }

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] { background: var(--pl-panel); border-right: 1px solid var(--pl-border); }
[data-testid="stSidebar"] .block-container, [data-testid="stSidebarUserContent"] { padding-top: 1rem; }
.pl-brand { display:flex; gap:.75rem; align-items:center; margin:.2rem 0 1.1rem; }
.pl-brand b { display:block; font-size:1.15rem; letter-spacing:-.02em; color:var(--pl-text); }
.pl-brand span { font-size:.75rem; color:var(--pl-muted); }
.pl-logo { width:40px; height:40px; border-radius:12px; display:grid; place-items:center;
  font-weight:800; color:#fff; font-size:1.2rem;
  background: conic-gradient(from 210deg, var(--pl-accent), #FFB547, var(--pl-accent-2), var(--pl-accent));
  box-shadow: 0 6px 20px rgba(255,106,77,.35); }
.pl-side-label { font-size:.7rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase;
  color:var(--pl-muted); margin:.2rem 0 .45rem; }
.pl-status { display:flex; flex-wrap:wrap; align-items:center; gap:.45rem; padding:.55rem .7rem;
  border-radius:10px; border:1px solid var(--pl-border); background:var(--pl-panel-2);
  font-size:.8rem; font-weight:600; margin-bottom:.8rem; }
.pl-status small { flex-basis:100%; color:var(--pl-muted); font-weight:500; }
.pl-dot { width:8px; height:8px; border-radius:50%; background:var(--pl-ok);
  box-shadow:0 0 0 0 rgba(61,220,151,.6); animation: pl-pulse 2s infinite; }
.pl-status-err .pl-dot { background:var(--pl-err); animation:none; }
@keyframes pl-pulse { 0%{box-shadow:0 0 0 0 rgba(61,220,151,.55)} 70%{box-shadow:0 0 0 8px rgba(61,220,151,0)} 100%{box-shadow:0 0 0 0 rgba(61,220,151,0)} }

/* ---------- Hero ---------- */
.pl-hero { position:relative; overflow:hidden; border:1px solid var(--pl-border); border-radius:20px;
  padding:1.15rem 1.6rem 1.1rem; margin-bottom:1rem;
  background: linear-gradient(135deg, rgba(23,27,36,.95), rgba(18,21,28,.85)); animation: pl-rise .5s ease both; }
.pl-hero::after { content:""; position:absolute; inset:-40% -10% auto auto; width:420px; height:420px;
  background: radial-gradient(circle, rgba(255,106,77,.22), transparent 65%); filter: blur(10px);
  animation: pl-float 9s ease-in-out infinite; }
.pl-hero h1 { margin:0; padding:0; font-size:1.8rem; font-weight:800; letter-spacing:-.03em; line-height:1.15;
  background: linear-gradient(90deg, #fff 0%, #FFD1C4 45%, var(--pl-accent-2) 100%);
  -webkit-background-clip:text; background-clip:text; color:transparent; }
.pl-hero p { margin:.3rem 0 .7rem; color:var(--pl-muted); max-width:760px; font-size:.95rem; }
.pl-chips { display:flex; flex-wrap:wrap; gap:.45rem; position:relative; z-index:1; }
.pl-chip { font-size:.74rem; font-weight:600; padding:.28rem .65rem; border-radius:999px;
  border:1px solid var(--pl-border); background:rgba(255,255,255,.03); color:#C9CED9; }
.pl-chip.accent { border-color:rgba(255,106,77,.45); color:#FFB4A3; background:rgba(255,106,77,.08); }
@keyframes pl-rise { from{opacity:0; transform:translateY(8px)} to{opacity:1; transform:none} }
@keyframes pl-float { 0%,100%{transform:translate(0,0)} 50%{transform:translate(-30px,20px)} }

/* ---------- Section title ---------- */
.pl-section { display:flex; align-items:baseline; gap:.6rem; margin:1.3rem 0 .6rem; }
.pl-section .pl-sec-title { margin:0; padding:0; font-size:1.15rem; font-weight:700; letter-spacing:-.01em; color:var(--pl-text); }
.pl-section span { color:var(--pl-muted); font-size:.8rem; }

/* ---------- Metric cards ---------- */
.pl-metrics { display:grid; grid-template-columns: repeat(auto-fit, minmax(128px, 1fr)); gap:.7rem; margin:.4rem 0 1rem; }
.pl-metric { position:relative; border:1px solid var(--pl-border); border-radius:var(--pl-radius);
  padding:.8rem .95rem; background:var(--pl-panel); transition: transform .18s ease, border-color .18s ease, box-shadow .18s ease;
  animation: pl-rise .45s ease both; cursor:default; }
.pl-metric:hover { transform: translateY(-3px); border-color: rgba(124,140,255,.55);
  box-shadow: 0 10px 28px rgba(0,0,0,.35); }
.pl-metric .lbl { font-size:.72rem; font-weight:600; color:var(--pl-muted); text-transform:uppercase; letter-spacing:.07em; }
.pl-metric .val { font-family:'JetBrains Mono', ui-monospace, monospace; font-size:1.2rem; font-weight:600;
  color:var(--pl-text); margin-top:.25rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.pl-metric .dlt { font-size:.75rem; font-weight:600; margin-top:.15rem; color:var(--pl-muted); }
.pl-metric .dlt.up { color: var(--pl-ok); } .pl-metric .dlt.down { color: var(--pl-err); }
.pl-metric .ico { position:absolute; top:.7rem; right:.8rem; opacity:.55; font-size:1rem; }

/* ---------- Image frame ---------- */
.pl-frame-title { display:flex; justify-content:space-between; align-items:center; margin:0 0 .45rem; }
.pl-frame-title b { font-size:.85rem; color:var(--pl-text); }
.pl-tag { font-size:.7rem; font-weight:700; padding:.2rem .55rem; border-radius:6px;
  background:rgba(124,140,255,.12); color:#B9C2FF; border:1px solid rgba(124,140,255,.3); }
.pl-tag.accent { background:rgba(255,106,77,.12); color:#FFB4A3; border-color:rgba(255,106,77,.35); }
[data-testid="stImage"] img { border-radius:12px; border:1px solid var(--pl-border); }

/* ---------- Empty state ---------- */
.pl-empty { text-align:center; border:1.5px dashed var(--pl-border); border-radius:18px; padding:2.6rem 1.5rem;
  background:rgba(18,21,28,.6); animation: pl-rise .5s ease both; }
.pl-empty .big { font-size:2.6rem; }
.pl-empty h4 { margin:.5rem 0 .3rem; font-size:1.1rem; color:var(--pl-text); }
.pl-empty p { color:var(--pl-muted); margin:0 auto; max-width:520px; font-size:.9rem; }

/* ---------- Widget polish ---------- */
.stButton > button, .stDownloadButton > button { border-radius:10px; font-weight:600;
  transition: transform .12s ease, box-shadow .12s ease; }
.stButton > button:hover, .stDownloadButton > button:hover { transform: translateY(-1px); }
.stButton > button[kind="primary"] { background: linear-gradient(135deg, var(--pl-accent), #FF8A5C);
  border:none; box-shadow: 0 6px 18px rgba(255,106,77,.30); }
.stButton > button:disabled, .stDownloadButton > button:disabled { opacity:.38; box-shadow:none !important;
  transform:none !important; cursor:not-allowed; filter: grayscale(.4); }
[data-testid="stFileUploaderDropzone"] { border-radius:14px; border:1.5px dashed var(--pl-border);
  background:var(--pl-panel); transition:border-color .2s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: var(--pl-accent); }
[data-testid="stExpander"] details { border-radius:12px; border-color:var(--pl-border); background:var(--pl-panel); }
[data-testid="stTabs"] [data-baseweb="tab-list"] { gap:.3rem; }
[data-testid="stTabs"] [data-baseweb="tab"] { border-radius:8px 8px 0 0; padding:.45rem .9rem; font-weight:600; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:var(--pl-radius); }
[data-testid="stAlert"] { border-radius:12px; }
code { font-family:'JetBrains Mono', monospace !important; }
</style>
"""


def inject_css():
    st.markdown(_CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chips=()):
    chip_html = "".join(
        f'<span class="pl-chip{" accent" if i == 0 else ""}">{html.escape(c)}</span>' for i, c in enumerate(chips))
    st.markdown(
        f'<div class="pl-hero"><h1>{html.escape(title)}</h1><p>{html.escape(subtitle)}</p>'
        f'<div class="pl-chips">{chip_html}</div></div>', unsafe_allow_html=True)


def section_title(title: str, caption: str = ""):
    st.markdown(f'<div class="pl-section"><div class="pl-sec-title">{html.escape(title)}</div>'
                f'<span>{html.escape(caption)}</span></div>', unsafe_allow_html=True)


def frame_title(title: str, tag: str = "", accent: bool = False):
    tag_html = f'<span class="pl-tag{" accent" if accent else ""}">{html.escape(tag)}</span>' if tag else ""
    st.markdown(f'<div class="pl-frame-title"><b>{html.escape(title)}</b>{tag_html}</div>',
                unsafe_allow_html=True)


def empty_state(icon: str, title: str, text: str):
    st.markdown(f'<div class="pl-empty"><div class="big">{icon}</div><h4>{html.escape(title)}</h4>'
                f'<p>{html.escape(text)}</p></div>', unsafe_allow_html=True)


def metric_cards(items):
    """
    items: list of dict(label, value, delta=None, direction=None('up'|'down'), hint='', icon='').
    Kartu memakai tooltip native (atribut title) & animasi hover.
    """
    cards = []
    for i, it in enumerate(items):
        delta = it.get("delta")
        direction = it.get("direction") or ""
        dlt = f'<div class="dlt {direction}">{html.escape(str(delta))}</div>' if delta else '<div class="dlt">&nbsp;</div>'
        cards.append(
            f'<div class="pl-metric" title="{html.escape(it.get("hint", ""))}" style="animation-delay:{i * 40}ms">'
            f'<span class="ico">{it.get("icon", "")}</span>'
            f'<div class="lbl">{html.escape(it["label"])}</div>'
            f'<div class="val">{html.escape(str(it["value"]))}</div>{dlt}</div>')
    st.markdown(f'<div class="pl-metrics">{"".join(cards)}</div>', unsafe_allow_html=True)


def delta_text(before: float, after: float, fmt: str = "{:+.1f}", pct: bool = False):
    """Teks delta + arah ('up'/'down'/None) untuk metric card."""
    diff = after - before
    if abs(diff) < 1e-6:
        return "tidak berubah", None
    if pct and abs(before) > 1e-6:
        text = f"{diff / abs(before) * 100:+.0f}%"
    else:
        text = fmt.format(diff)
    return text, ("up" if diff > 0 else "down")


# ---------------------------------------------------------------------------
# Slider pembanding before/after
# ---------------------------------------------------------------------------
def compare_slider(before_bgr: np.ndarray, after_bgr: np.ndarray, height: int = 560,
                   before_label: str = "SEBELUM", after_label: str = "SESUDAH"):
    """Slider geser interaktif (drag/klik/sentuh/keyboard). Kedua citra harus berasio sama."""
    before_uri = helpers.image_to_data_uri(before_bgr)
    after_uri = helpers.image_to_data_uri(after_bgr)
    h, w = after_bgr.shape[:2]
    doc = f"""
<!doctype html><html><head><style>
  html,body {{ margin:0; background:transparent; font-family:system-ui,'Segoe UI',sans-serif; }}
  .wrap {{ position:relative; width:100%; height:{height}px; display:flex; justify-content:center; }}
  .stage {{ position:relative; height:100%; aspect-ratio:{w}/{h}; max-width:100%; border-radius:12px;
           overflow:hidden; border:1px solid #232836; user-select:none; touch-action:none; cursor:ew-resize; }}
  .stage img {{ position:absolute; inset:0; width:100%; height:100%; object-fit:contain; pointer-events:none; }}
  .top {{ clip-path: inset(0 50% 0 0); }}
  .bar {{ position:absolute; top:0; bottom:0; left:50%; width:2px; background:#fff;
         box-shadow:0 0 12px rgba(0,0,0,.6); transform:translateX(-1px); }}
  .knob {{ position:absolute; top:50%; left:50%; width:42px; height:42px; border-radius:50%;
          transform:translate(-50%,-50%); background:#FF6A4D; border:3px solid #fff;
          box-shadow:0 6px 18px rgba(0,0,0,.45); display:grid; place-items:center; color:#fff;
          font-weight:800; font-size:15px; outline:none; }}
  .lbl {{ position:absolute; top:12px; padding:4px 10px; border-radius:6px; font-size:11px; font-weight:700;
         letter-spacing:.08em; color:#fff; background:rgba(11,13,18,.72); backdrop-filter:blur(4px); }}
  .l {{ left:12px; }} .r {{ right:12px; }}
</style></head><body>
<div class="wrap"><div class="stage" id="s">
  <img src="{after_uri}" alt="sesudah"><img class="top" id="t" src="{before_uri}" alt="sebelum">
  <div class="bar" id="b"></div>
  <div class="knob" id="k" tabindex="0" role="slider" aria-label="Pembanding" aria-valuemin="0" aria-valuemax="100" aria-valuenow="50">⇆</div>
  <span class="lbl l">{html.escape(before_label)}</span><span class="lbl r">{html.escape(after_label)}</span>
</div></div>
<script>
  const s=document.getElementById('s'), t=document.getElementById('t'), b=document.getElementById('b'), k=document.getElementById('k');
  let pos=50, drag=false;
  function set(p){{ pos=Math.max(0,Math.min(100,p)); t.style.clipPath=`inset(0 ${{100-pos}}% 0 0)`;
    b.style.left=pos+'%'; k.style.left=pos+'%'; k.setAttribute('aria-valuenow', Math.round(pos)); }}
  function fromEvt(e){{ const r=s.getBoundingClientRect(); set((e.clientX-r.left)/r.width*100); }}
  s.addEventListener('pointerdown', e=>{{ drag=true; s.setPointerCapture(e.pointerId); fromEvt(e); }});
  s.addEventListener('pointermove', e=>{{ if(drag) fromEvt(e); }});
  s.addEventListener('pointerup', ()=>drag=false);
  k.addEventListener('keydown', e=>{{ if(e.key==='ArrowLeft') set(pos-2); if(e.key==='ArrowRight') set(pos+2); }});
</script></body></html>"""
    components.html(doc, height=height + 6)


# ---------------------------------------------------------------------------
# Histogram interaktif
# ---------------------------------------------------------------------------
def _hist_long(image_bgr, version):
    data = histogram_table(image_bgr)
    df = pd.DataFrame(data)
    long = df.melt(id_vars="Intensitas", var_name="Channel", value_name="Jumlah piksel")
    long["Versi"] = version
    return long


def histogram_chart(before_bgr: np.ndarray, after_bgr: np.ndarray, mode: str = "luma"):
    """
    mode='luma' : area luminansi SESUDAH + garis putus-putus SEBELUM (overlay).
    mode='rgb'  : small multiples per channel (baris berlabel), masing-masing
                  area SESUDAH berwarna + garis SEBELUM abu-abu putus-putus.
    Tooltip & zoom/pan sumbu X aktif.
    """
    df = pd.concat([_hist_long(before_bgr, "Sebelum"), _hist_long(after_bgr, "Sesudah")], ignore_index=True)
    x = alt.X("Intensitas:Q", scale=alt.Scale(domain=[0, 255], nice=False, clamp=True),
              title="Intensitas piksel (0–255)")
    tooltip = [alt.Tooltip("Versi:N"), alt.Tooltip("Channel:N"), alt.Tooltip("Intensitas:Q"),
               alt.Tooltip("Jumlah piksel:Q", format=",.0f")]

    def _style(chart):
        return chart.configure(background="transparent").configure_view(strokeWidth=0).configure_axis(
            gridColor="#232836", domainColor="#2c3242", labelColor="#8A91A5", titleColor="#8A91A5"
        ).configure_legend(labelColor="#C9CED9").configure_title(color="#E8EAF0", anchor="start", fontSize=13)

    if mode == "rgb":
        # Small multiples: satu grafik penuh-lebar per channel (judul = identitas, bukan warna saja)
        for ch, color in CHANNEL_COLORS.items():
            d = df[df["Channel"] == ch]
            y = alt.Y("Jumlah piksel:Q", title=None, axis=alt.Axis(format="~s", tickCount=3))
            area = alt.Chart(d[d["Versi"] == "Sesudah"]).mark_area(
                color=color, opacity=0.35, line={"color": color, "strokeWidth": 2}).encode(x=x, y=y, tooltip=tooltip)
            before = alt.Chart(d[d["Versi"] == "Sebelum"]).mark_line(
                color=NEUTRAL, strokeDash=[5, 4], strokeWidth=1.5).encode(x=x, y=y, tooltip=tooltip)
            chart = (before + area).properties(height=130, title=f"Channel {ch}").interactive(bind_y=False)
            st.altair_chart(_style(chart), width="stretch", theme=None)
        st.caption("Garis putus-putus abu-abu = sebelum filter · area berwarna = sesudah filter. "
                   "Scroll/drag pada grafik untuk zoom sumbu X.")
        return df

    # mode == "luma": overlay sebelum (garis putus-putus) vs sesudah (area)
    d = df[df["Channel"] == "Luminansi"]
    y = alt.Y("Jumlah piksel:Q", title="Jumlah piksel", axis=alt.Axis(format="~s"))
    area = alt.Chart(d[d["Versi"] == "Sesudah"]).mark_area(
        opacity=0.35, color="#FF6A4D", line={"color": "#FF6A4D", "strokeWidth": 2}).encode(x=x, y=y, tooltip=tooltip)
    before = alt.Chart(d[d["Versi"] == "Sebelum"]).mark_line(
        color=NEUTRAL, strokeDash=[5, 4], strokeWidth=2).encode(x=x, y=y, tooltip=tooltip)
    legend = alt.Chart(pd.DataFrame({"Versi": ["Sebelum", "Sesudah"]})).mark_point(opacity=0).encode(
        color=alt.Color("Versi:N", scale=alt.Scale(domain=["Sebelum", "Sesudah"], range=[NEUTRAL, "#FF6A4D"]),
                        legend=alt.Legend(orient="top", title=None, symbolOpacity=1)))
    chart = (before + area + legend).properties(height=260).interactive(bind_y=False)
    st.altair_chart(_style(chart), width="stretch", theme=None)
    st.caption("Garis putus-putus abu-abu = sebelum filter · area berwarna = sesudah filter. "
               "Scroll/drag pada grafik untuk zoom sumbu X.")
    return df
