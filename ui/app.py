"""ChestAI - Streamlit UI for the chest X-ray API.

Run:  streamlit run app.py
Config (environment): API_URL (default http://localhost:8000)
The UI only talks to the FastAPI backend; it never calls Groq directly.
"""
import base64
import hashlib
import html
import io
import os

import requests
import streamlit as st
from PIL import Image


API_URL = os.getenv("API_URL", "http://localhost:8000").rstrip("/")
REQUEST_TIMEOUT = float(os.getenv("API_TIMEOUT", "120"))

st.set_page_config(
    page_title="ChestAI",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="collapsed",
)

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&display=swap');
:root{
  --bg:#0a1217; --surface:#101c24; --line:#1d303b; --text:#e7f0f4; --muted:#8ba3b0;
  --teal:#2dd4bf; --blue:#5ab0ff; --amber:#e6b455;
}
html, body, [class*="css"], .stApp { font-family:'IBM Plex Sans', system-ui, sans-serif; }
.stApp { background:var(--bg); color:var(--text); }

/* no sidebar, no Streamlit chrome */
[data-testid="stSidebar"], [data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"], [data-testid="stSidebarNav"] { display:none !important; }
#MainMenu, footer, [data-testid="stToolbar"], [data-testid="stDecoration"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; }
.block-container { max-width:1120px; padding:2.2rem 1.4rem 3rem; }

h1.cx-title { font-size:clamp(1.55rem, 3.2vw, 2.15rem); font-weight:600; letter-spacing:-0.01em;
  margin:.9rem 0 .35rem; line-height:1.2; color:var(--text); }
p.cx-lead { color:var(--muted); font-size:1.02rem; line-height:1.6; max-width:68ch; margin:0 0 1.6rem; }
.cx-note { display:inline-block; border:1px solid rgba(230,180,85,.45); color:var(--amber);
  background:rgba(230,180,85,.07); border-radius:6px; padding:.28rem .65rem; font-size:.82rem; }

h3.cx-sec { font-size:1.12rem; font-weight:600; margin:2.1rem 0 .8rem; color:var(--text);
  padding-left:.7rem; border-left:3px solid var(--teal); }
.cx-sub { color:var(--muted); font-size:.9rem; margin:-.45rem 0 .9rem .7rem; }

/* uploader */
[data-testid="stFileUploader"] section { background:var(--surface); border:1px dashed #2a4553;
  border-radius:10px; padding:1.1rem; }
[data-testid="stFileUploader"] section:hover { border-color:var(--teal); }
[data-testid="stFileUploader"] small, [data-testid="stFileUploader"] span { color:var(--muted); }

/* buttons */
.stButton > button[kind="primary"] { background:linear-gradient(135deg,#14b8a6,#3b82f6); color:#041016;
  border:0; font-weight:600; padding:.55rem 1.5rem; border-radius:8px; }
.stButton > button[kind="primary"]:hover { filter:brightness(1.08); color:#041016; }
.stButton > button[kind="primary"]:focus-visible { outline:2px solid var(--blue); outline-offset:2px; }
.stButton > button:not([kind="primary"]) { background:transparent; border:1px solid var(--line); color:var(--text); }

/* select */
[data-baseweb="select"] > div { background:var(--surface); border-color:var(--line); }

/* images */
.cx-fig { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.7rem; margin:0; }
.cx-fig img { width:100%; border-radius:6px; display:block; background:#000; }
.cx-fig figcaption { color:var(--muted); font-size:.86rem; padding:.55rem .2rem 0; }
.cx-empty { display:flex; align-items:center; justify-content:center; min-height:260px; text-align:center;
  color:var(--muted); border:1px dashed var(--line); border-radius:10px; padding:1rem; }

/* findings chips */
.cx-chips { display:flex; flex-wrap:wrap; gap:.45rem; margin:0 0 1rem; align-items:center; }
.cx-chip { border:1px solid rgba(45,212,191,.5); background:rgba(45,212,191,.1); color:var(--teal);
  border-radius:999px; padding:.22rem .75rem; font-size:.88rem; }
.cx-none { color:var(--muted); font-size:.95rem; }

/* score bars */
.cx-rows { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:.5rem 1rem; }
.cx-row { display:grid; grid-template-columns:minmax(120px,190px) 1fr 3.2rem; align-items:center;
  gap:.9rem; padding:.45rem 0; border-bottom:1px solid rgba(29,48,59,.7); font-size:.92rem; }
.cx-row:last-child { border-bottom:0; }
.cx-row .nm { color:var(--muted); }
.cx-row.pos .nm { color:var(--text); font-weight:500; }
.cx-bar { height:8px; background:#16262f; border-radius:4px; position:relative; }
.cx-bar b.th { position:absolute; top:-3px; bottom:-3px; width:2px; background:var(--amber); border-radius:1px; }
.cx-prio { display:inline-block; border-radius:6px; padding:.28rem .7rem; font-size:.88rem; font-weight:600; margin-right:.6rem; }
.cx-prio.High { background:rgba(248,113,113,.12); color:#f87171; border:1px solid rgba(248,113,113,.5); }
.cx-prio.Medium { background:rgba(230,180,85,.1); color:var(--amber); border:1px solid rgba(230,180,85,.5); }
.cx-prio.Low { background:rgba(45,212,191,.1); color:var(--teal); border:1px solid rgba(45,212,191,.5); }
.cx-bar i { display:block; height:100%; background:#3d6f8a; border-radius:4px; }
.cx-row.pos .cx-bar i { background:linear-gradient(90deg,var(--teal),var(--blue)); }
.cx-row .sc { text-align:right; font-variant-numeric:tabular-nums; color:var(--text); }
.cx-foot { color:var(--muted); font-size:.82rem; margin-top:.6rem; }

/* text cards */
.cx-card { background:var(--surface); border:1px solid var(--line); border-radius:10px; padding:1.1rem 1.3rem; margin:0; }
.cx-card dt { color:var(--teal); font-weight:600; font-size:.95rem; margin:1rem 0 .2rem; }
.cx-card dt:first-child { margin-top:0; }
.cx-card dd { margin:0; line-height:1.65; max-width:78ch; }
.cx-warn { color:var(--muted); }
.cx-spec { padding:.7rem 0; border-bottom:1px solid rgba(29,48,59,.7); }
.cx-spec:first-child { padding-top:0; }
.cx-spec b { color:var(--blue); font-weight:600; }
.cx-spec p { margin:.15rem 0 0; color:var(--muted); line-height:1.55; }
.cx-clin { margin-top:.9rem; color:var(--muted); font-size:.92rem; line-height:1.55; }
.cx-clin.rule { padding-top:.9rem; border-top:1px solid var(--line); }

.cx-footer { margin-top:2.6rem; padding-top:1rem; border-top:1px solid var(--line); color:var(--muted);
  font-size:.82rem; line-height:1.55; }
@media (max-width:640px){ .cx-row{ grid-template-columns:1fr 3rem; } .cx-row .cx-bar{ grid-column:1 / -1; order:3; } }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

REASON_TEXT = {
    "not_configured": "The explanation service is not set up on this server.",
    "rate_limited": "The explanation service is busy right now.",
    "timeout": "The explanation service took too long to respond.",
    "service_error": "The explanation service could not be reached.",
    "invalid_response": "The explanation could not be verified, so it was not shown.",
}


# ------------------------------------------------------------------ helpers
def esc(x) -> str:
    return html.escape(str(x))


def label(name: str) -> str:
    return name.replace("_", " ")


@st.cache_data(show_spinner=False, max_entries=8)
def preview_data_uri(img_bytes: bytes) -> str:
    """Downscaled PNG data-URI of the uploaded image, for display only."""
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img.thumbnail((1024, 1024))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def figure(src: str, caption: str, alt: str) -> str:
    return (
        f'<figure class="cx-fig"><img src="{src}" alt="{esc(alt)}">'
        f"<figcaption>{esc(caption)}</figcaption></figure>"
    )


def request_analysis(img_bytes: bytes, name: str, mime: str, cam_class: str):
    """Call the FastAPI backend. Returns (result | None, error message | None)."""
    try:
        r = requests.post(
            f"{API_URL}/predict",
            params={"explain": "true", "llm_explain": "true",
                    **({} if cam_class == "auto" else {"explain_class": cam_class})},
            files={"file": (name, img_bytes, mime or "image/png")},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException:
        return None, "Could not reach the analysis service. Please try again shortly."
    if r.status_code != 200:
        try:
            detail = r.json().get("detail", "")
        except ValueError:
            detail = ""
        return None, f"The analysis could not be completed. {detail}".strip()
    try:
        return r.json(), None
    except ValueError:
        return None, "The analysis service returned an unexpected response."


def render_scores(preds: list) -> str:
    rows = []
    for p in preds:
        pos = " pos" if p["detected"] else ""
        score = float(p["probability"])
        thr = float(p["threshold"])
        rows.append(
            f'<div class="cx-row{pos}"><span class="nm">{esc(label(p["disease"]))}</span>'
            f'<div class="cx-bar"><i style="width:{score * 100:.1f}%"></i>'
            f'<b class="th" style="left:{thr * 100:.1f}%" title="threshold {thr:.2f}"></b></div>'
            f'<span class="sc">{score:.2f}</span></div>'
        )
    return '<div class="cx-rows">' + "".join(rows) + "</div>"


def section(title: str, sub: str | None = None, first: bool = False):
    style = ' style="margin-top:.4rem"' if first else ""
    st.markdown(f'<h3 class="cx-sec"{style}>{esc(title)}</h3>', unsafe_allow_html=True)
    if sub:
        st.markdown(f'<div class="cx-sub">{sub}</div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ header

st.markdown('<h1 class="cx-title">ChestAI — Intelligent Chest X-ray Analysis</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="cx-lead">Upload a chest X-ray to see how an AI model scores 14 findings, which image '
    "regions influenced a chosen score, and a plain-language explanation of the result.</p>",
    unsafe_allow_html=True,
)

# ------------------------------------------------------------------ upload
section("Upload X-ray", first=True)
file = st.file_uploader(
    "Chest X-ray image (PNG or JPG)", type=["png", "jpg", "jpeg"], label_visibility="collapsed"
)
state = st.session_state

if file is None:
    st.markdown('<div class="cx-empty">Choose a PNG or JPG chest X-ray to begin.</div>', unsafe_allow_html=True)
else:
    img_bytes = file.getvalue()
    digest = hashlib.sha256(img_bytes).hexdigest()
    if state.get("digest") != digest:  # new image: drop previous results
        state["digest"] = digest
        state["results"] = {}
        state["pred_order"] = None
        state.pop("cam_class", None)

    if st.button("Analyze X-ray", type="primary"):
        state["results"] = {}
        with st.spinner("Analyzing image…"):
            res, err = request_analysis(img_bytes, file.name, file.type, "auto")
        if err:
            st.error(err)
        else:
            cls = res["explained_class"]
            state["results"][cls] = res
            state["pred_order"] = [p["disease"] for p in res["findings"]]
            state["scores"] = {p["disease"]: p["probability"] for p in res["findings"]}
            state["cam_class"] = cls

    analyzed = bool(state.get("pred_order"))
    res = None
    selected = None

    # ------------------------------------------------------------- visualization
    section(
        "Original X-ray and Grad-CAM",
        "Choose a condition to see which image regions influenced its score." if analyzed else None,
    )
    if analyzed:
        scores = dict(state["scores"])
        selected = st.selectbox(
            "Grad-CAM class",
            state["pred_order"],
            key="cam_class",
            format_func=lambda c: f"{label(c)} · score {scores[c]:.2f}",
            label_visibility="collapsed",
        )
        if selected not in state["results"]:
            with st.spinner("Updating heatmap and explanation…"):
                new_res, err = request_analysis(img_bytes, file.name, file.type, selected)
            if err:
                st.error(err)
            else:
                state["results"][selected] = new_res
        res = state["results"].get(selected)

    left, right = st.columns(2, gap="medium")
    with left:
        st.markdown(figure(preview_data_uri(img_bytes), "Original X-ray", "Uploaded chest X-ray"), unsafe_allow_html=True)
    with right:
        if res and res.get("heatmap_base64"):
            cam_src = "data:image/png;base64," + res["heatmap_base64"]
            st.markdown(
                figure(cam_src, f"Grad-CAM · {label(res['explained_class'])}", "Grad-CAM heatmap"),
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="cx-empty">Select Analyze X-ray to generate the heatmap.</div>',
                unsafe_allow_html=True,
            )

    if res:
        # --------------------------------------------------------- predictions
        section("AI prediction results")
        prio = res.get("review_priority", "Medium")
        st.markdown(
            f'<div class="cx-chips"><span class="cx-prio {esc(prio)}">Review priority: {esc(prio)}</span></div>',
            unsafe_allow_html=True,
        )
        findings = res["detected"]
        if findings:
            chips = "".join(f'<span class="cx-chip">{esc(label(f))}</span>' for f in findings)
            st.markdown(
                f'<div class="cx-chips"><span class="cx-none">Above their decision threshold:</span>{chips}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="cx-chips"><span class="cx-none">No finding scored above its decision threshold.</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown(render_scores(res["findings"]), unsafe_allow_html=True)
        st.markdown(
            '<div class="cx-foot">Scores are model outputs on a 0–1 scale, not calibrated probabilities of disease. The amber tick on each bar marks that finding’s decision threshold.</div>',
            unsafe_allow_html=True,
        )

        # --------------------------------------------------------- explanation
        exp = res.get("explanation") or {"status": "unavailable", "reason": "service_error"}
        exp_ok = exp.get("status") == "ok" and bool(exp.get("data"))
        section(
            "AI Explanation",
            f"About {esc(label(exp.get('class', selected)))}. Written from the model scores only; "
            "the AI did not examine your image.",
        )
        if exp_ok:
            d = exp["data"]
            items = [
                ("About this condition", d["condition_overview"]),
                ("What the score means", d["score_interpretation"]),
                ("About the heatmap", d["gradcam_note"]),
                ("Limits and uncertainty", d["uncertainty_note"]),
            ]
            body = "".join(f"<dt>{esc(t)}</dt><dd>{esc(v)}</dd>" for t, v in items)
            st.markdown(f'<dl class="cx-card">{body}</dl>', unsafe_allow_html=True)
        else:
            msg = REASON_TEXT.get(exp.get("reason"), REASON_TEXT["service_error"])
            st.markdown(
                f'<div class="cx-card cx-warn">AI explanation unavailable. {esc(msg)} '
                "Your predictions and heatmap above are not affected.</div>",
                unsafe_allow_html=True,
            )
            if st.button("Try again", key="retry_explanation"):
                with st.spinner("Retrying…"):
                    new_res, err = request_analysis(img_bytes, file.name, file.type, selected)
                if err:
                    st.error(err)
                else:
                    state["results"][selected] = new_res
                    st.rerun()

        # --------------------------------------------------------- follow-up
        section("Suggested Medical Follow-up")
        fu = res.get("follow_up") or {}
        parts = []
        if fu.get("specialties"):
            for s in fu["specialties"]:
                parts.append(f'<div class="cx-spec"><b>{esc(s["name"])}</b><p>{esc(s["reason"])}</p></div>')
        else:
            generic = fu.get("generic_message") or "Please have this result reviewed by a qualified clinician."
            parts.append(f'<div class="cx-spec"><p style="margin:0">{esc(generic)}</p></div>')
        if exp_ok:
            parts.append(f'<div class="cx-clin">{esc(exp["data"]["follow_up_note"])}</div>')
        note = fu.get("note", "A qualified clinician must interpret these results.")
        parts.append(f'<div class="cx-clin rule">{esc(note)}</div>')
        st.markdown(f'<div class="cx-card">{"".join(parts)}</div>', unsafe_allow_html=True)

st.markdown(
    '<div class="cx-footer">ChestAI is a research prototype. It is not a medical device, does not provide a diagnosis, '
    "and must not be used for clinical decisions. Always consult a qualified healthcare professional.</div>",
    unsafe_allow_html=True,
)
