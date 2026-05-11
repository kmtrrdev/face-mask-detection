"""
Face Mask Detection — Futuristic Streamlit UI
"""

import streamlit as st
import requests
from PIL import Image
import io
import cv2
import numpy as np
import urllib.request
import os
import base64

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="MASK.AI — Face Mask Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

API_URL = "http://localhost:8000/predict"

# ─── Futuristic CSS ───────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

:root {
    --neon-cyan:   #00f5ff;
    --neon-green:  #00ff88;
    --neon-red:    #ff003c;
    --neon-gold:   #ffd700;
    --bg-dark:     #020408;
    --bg-card:     rgba(0, 245, 255, 0.03);
    --border-glow: rgba(0, 245, 255, 0.2);
}

/* Global */
html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg-dark) !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: #c0e8f0 !important;
}
[data-testid="stAppViewContainer"]::before {
    content: '';
    position: fixed;
    inset: 0;
    background:
        radial-gradient(ellipse 80% 50% at 20% 10%, rgba(0,245,255,0.06) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 90%, rgba(0,255,136,0.04) 0%, transparent 60%);
    pointer-events: none;
    z-index: 0;
}

/* Grid scanlines overlay */
[data-testid="stAppViewContainer"]::after {
    content: '';
    position: fixed;
    inset: 0;
    background-image: repeating-linear-gradient(
        0deg, transparent, transparent 2px,
        rgba(0,245,255,0.015) 2px, rgba(0,245,255,0.015) 4px
    );
    pointer-events: none;
    z-index: 0;
}

[data-testid="stMain"] { background: transparent !important; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 2rem 3rem !important; }

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }

/* ── Hero Title ── */
.hero-title {
    font-family: 'Orbitron', monospace;
    font-size: clamp(2.5rem, 5vw, 4.5rem);
    font-weight: 900;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    background: linear-gradient(135deg, var(--neon-cyan) 0%, #ffffff 50%, var(--neon-green) 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    margin-bottom: 0;
    filter: drop-shadow(0 0 30px rgba(0,245,255,0.4));
    animation: titlePulse 4s ease-in-out infinite;
}
@keyframes titlePulse {
    0%, 100% { filter: drop-shadow(0 0 30px rgba(0,245,255,0.4)); }
    50%       { filter: drop-shadow(0 0 60px rgba(0,245,255,0.8)); }
}
.hero-sub {
    font-family: 'Orbitron', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.5em;
    color: rgba(0,245,255,0.5);
    text-align: center;
    margin-top: 0.3rem;
    text-transform: uppercase;
}
.hero-line {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
    margin: 1.5rem auto;
    max-width: 600px;
    box-shadow: 0 0 10px var(--neon-cyan);
}

/* ── Cards ── */
.glass-card {
    background: var(--bg-card);
    border: 1px solid var(--border-glow);
    border-radius: 4px;
    padding: 1.5rem;
    position: relative;
    backdrop-filter: blur(10px);
    box-shadow: 0 0 30px rgba(0,245,255,0.05), inset 0 1px 0 rgba(255,255,255,0.05);
}
.glass-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), transparent);
    border-radius: 4px 4px 0 0;
}

/* ── Result Cards ── */
.result-allow {
    background: rgba(0,255,136,0.06);
    border: 1px solid rgba(0,255,136,0.4);
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    animation: allowPulse 2s ease-in-out infinite;
    box-shadow: 0 0 40px rgba(0,255,136,0.15), inset 0 0 40px rgba(0,255,136,0.03);
}
@keyframes allowPulse {
    0%, 100% { box-shadow: 0 0 40px rgba(0,255,136,0.15), inset 0 0 40px rgba(0,255,136,0.03); }
    50%       { box-shadow: 0 0 80px rgba(0,255,136,0.3),  inset 0 0 60px rgba(0,255,136,0.06); }
}
.result-deny {
    background: rgba(255,0,60,0.06);
    border: 1px solid rgba(255,0,60,0.4);
    border-radius: 4px;
    padding: 2rem;
    text-align: center;
    animation: denyPulse 1.5s ease-in-out infinite;
    box-shadow: 0 0 40px rgba(255,0,60,0.15), inset 0 0 40px rgba(255,0,60,0.03);
}
@keyframes denyPulse {
    0%, 100% { box-shadow: 0 0 40px rgba(255,0,60,0.15), inset 0 0 40px rgba(255,0,60,0.03); }
    50%       { box-shadow: 0 0 80px rgba(255,0,60,0.35), inset 0 0 60px rgba(255,0,60,0.06); }
}
.result-icon { font-size: 4rem; line-height: 1; margin-bottom: 0.5rem; }
.result-label {
    font-family: 'Orbitron', monospace;
    font-size: 1.6rem;
    font-weight: 900;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}
.result-label-allow { color: var(--neon-green); text-shadow: 0 0 20px var(--neon-green); }
.result-label-deny  { color: var(--neon-red);   text-shadow: 0 0 20px var(--neon-red); }
.result-action {
    font-family: 'Rajdhani', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    opacity: 0.7;
}

/* ── Confidence Bar ── */
.conf-bar-container {
    margin: 1.2rem 0;
}
.conf-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.3em;
    color: rgba(0,245,255,0.6);
    text-transform: uppercase;
    margin-bottom: 0.4rem;
}
.conf-bar-track {
    height: 6px;
    background: rgba(0,245,255,0.08);
    border-radius: 3px;
    overflow: hidden;
    border: 1px solid rgba(0,245,255,0.1);
}
.conf-bar-fill-green {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #00cc66, var(--neon-green));
    box-shadow: 0 0 10px var(--neon-green);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.conf-bar-fill-red {
    height: 100%;
    border-radius: 3px;
    background: linear-gradient(90deg, #cc0022, var(--neon-red));
    box-shadow: 0 0 10px var(--neon-red);
    transition: width 0.8s cubic-bezier(0.4,0,0.2,1);
}
.conf-value {
    font-family: 'Orbitron', monospace;
    font-size: 0.8rem;
    text-align: right;
    margin-top: 0.2rem;
}

/* ── Section Labels ── */
.section-label {
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.4em;
    color: rgba(0,245,255,0.4);
    text-transform: uppercase;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.8rem;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(0,245,255,0.2), transparent);
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'Orbitron', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.2em;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
    text-transform: uppercase;
}
.status-online {
    background: rgba(0,255,136,0.1);
    border: 1px solid rgba(0,255,136,0.3);
    color: var(--neon-green);
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--neon-green);
    box-shadow: 0 0 6px var(--neon-green);
    animation: blink 1.5s infinite;
}
@keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }

/* ── Toggle ── */
[data-testid="stToggle"] label {
    font-family: 'Rajdhani', sans-serif !important;
    letter-spacing: 0.1em !important;
    color: rgba(0,245,255,0.7) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    border-bottom: 1px solid rgba(0,245,255,0.15) !important;
    gap: 0 !important;
}
[data-testid="stTabs"] [role="tab"] {
    font-family: 'Orbitron', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.2em !important;
    color: rgba(0,245,255,0.4) !important;
    border-radius: 0 !important;
    padding: 0.6rem 1.5rem !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    transition: all 0.2s !important;
    text-transform: uppercase !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: var(--neon-cyan) !important;
    border-bottom: 2px solid var(--neon-cyan) !important;
    background: rgba(0,245,255,0.04) !important;
    text-shadow: 0 0 10px var(--neon-cyan) !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: rgba(0,245,255,0.02) !important;
    border: 1px dashed rgba(0,245,255,0.2) !important;
    border-radius: 4px !important;
}

/* ── Camera ── */
[data-testid="stCameraInput"] video,
[data-testid="stCameraInput"] img {
    border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 4px !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: var(--neon-cyan) !important; }

/* ── Warning / Info ── */
[data-testid="stAlert"] {
    background: rgba(0,245,255,0.04) !important;
    border-color: rgba(0,245,255,0.2) !important;
    font-family: 'Rajdhani', sans-serif !important;
}
</style>
""", unsafe_allow_html=True)

# ─── Face Detector ────────────────────────────────────────────────
CASCADE_PATH = "haarcascade_frontalface_default.xml"
if not os.path.exists(CASCADE_PATH):
    try:
        urllib.request.urlretrieve(
            "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
            CASCADE_PATH
        )
    except:
        CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)


def detect_and_crop(pil_img: Image.Image, padding: float = 0.3):
    arr  = np.array(pil_img.convert("RGB"))
    gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60,60))
    if len(faces) == 0:
        return pil_img, False
    x, y, w, h = max(faces, key=lambda f: f[2]*f[3])
    H, W = arr.shape[:2]
    px, py = int(w*padding), int(h*padding)
    crop = arr[max(0,y-py):min(H,y+h+py), max(0,x-px):min(W,x+w+px)]
    return Image.fromarray(crop), True


def render_result(result: dict):
    status = result["status"]
    cls    = result["class"]
    conf   = result["confidence"]
    probs  = result["probabilities"]
    action = result["action"]

    allow = status == "mask_on"
    card_cls   = "result-allow" if allow else "result-deny"
    label_cls  = "result-label-allow" if allow else "result-label-deny"
    icon       = "🛡️" if allow else "⚠️"
    fill_cls   = "conf-bar-fill-green" if allow else "conf-bar-fill-red"
    conf_color = "#00ff88" if allow else "#ff003c"

    st.markdown(f"""
    <div class="{card_cls}">
        <div class="result-icon">{icon}</div>
        <div class="result-label {label_cls}">{cls}</div>
        <div class="result-action">{action}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Confidence bar
    pct = conf * 100
    st.markdown(f"""
    <div class="conf-bar-container">
        <div class="conf-label">Confidence Level</div>
        <div class="conf-bar-track">
            <div class="{fill_cls}" style="width:{pct:.1f}%"></div>
        </div>
        <div class="conf-value" style="color:{conf_color};">{pct:.1f}%</div>
    </div>
    """, unsafe_allow_html=True)

    # Prob bars
    st.markdown('<div class="section-label">Class Probabilities</div>', unsafe_allow_html=True)
    for label, prob in probs.items():
        is_mask = label == "WithMask"
        bar_cls = "conf-bar-fill-green" if is_mask else "conf-bar-fill-red"
        col_hex = "#00ff88" if is_mask else "#ff003c"
        st.markdown(f"""
        <div class="conf-bar-container">
            <div class="conf-label">{label}</div>
            <div class="conf-bar-track">
                <div class="{bar_cls}" style="width:{prob*100:.1f}%"></div>
            </div>
            <div class="conf-value" style="color:{col_hex};">{prob*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)


def run_prediction(img: Image.Image, face_detect: bool):
    cropped, found = (detect_and_crop(img) if face_detect else (img, True))
    if face_detect and not found:
        st.warning("No face detected — using full image.")
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG")

    col_img, col_res = st.columns([1, 1], gap="large")
    with col_img:
        st.markdown('<div class="section-label">Captured Frame</div>', unsafe_allow_html=True)
        st.image(img, use_container_width=True)
        if face_detect and found:
            st.markdown('<div class="section-label">Cropped Face</div>', unsafe_allow_html=True)
            st.image(cropped, use_container_width=True)

    with col_res:
        st.markdown('<div class="section-label">Analysis Result</div>', unsafe_allow_html=True)
        try:
            resp = requests.post(
                API_URL,
                files={"file": ("face.jpg", buf.getvalue(), "image/jpeg")},
                timeout=15,
            )
            render_result(resp.json()) if resp.status_code == 200 else st.error(resp.json().get("detail"))
        except requests.exceptions.ConnectionError:
            st.error("⚠️  API offline — run `python app.py` first.")
        except Exception as e:
            st.error(str(e))


# ─── HERO ────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 2rem 0 1rem;">
    <div class="hero-title">MASK · AI</div>
    <div class="hero-sub">Neural Face Mask Detection System · MobileNetV2</div>
    <div class="hero-line"></div>
</div>
""", unsafe_allow_html=True)

# Status + toggle row
c1, c2, c3 = st.columns([1, 2, 1])
with c1:
    st.markdown('<div class="status-badge status-online"><div class="status-dot"></div>SYSTEM ONLINE</div>', unsafe_allow_html=True)
with c3:
    face_detect = st.toggle("Auto Face Crop", value=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─── TABS ────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["▶  Upload Image", "◉  Live Webcam"])

with tab1:
    uploaded = st.file_uploader("", type=["jpg","jpeg","png"], label_visibility="collapsed")
    if uploaded:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("ANALYZING..."):
            run_prediction(Image.open(uploaded).convert("RGB"), face_detect)

with tab2:
    st.markdown('<p style="font-family:\'Rajdhani\',sans-serif;letter-spacing:0.1em;color:rgba(0,245,255,0.5);font-size:0.9rem;">Position your face in the frame, then click Take Photo</p>', unsafe_allow_html=True)
    cam = st.camera_input("", label_visibility="collapsed")
    if cam:
        st.markdown("<br>", unsafe_allow_html=True)
        with st.spinner("ANALYZING..."):
            run_prediction(Image.open(cam).convert("RGB"), face_detect)

# ─── FOOTER ──────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:4rem; border-top:1px solid rgba(0,245,255,0.08); padding-top:1rem; display:flex; justify-content:space-between; align-items:center;">
    <span style="font-family:'Orbitron',monospace;font-size:0.55rem;letter-spacing:0.3em;color:rgba(0,245,255,0.2);">MASK.AI · v1.0</span>
    <span style="font-family:'Rajdhani',sans-serif;font-size:0.8rem;color:rgba(0,245,255,0.25);letter-spacing:0.1em;">MobileNetV2 · OpenCV · FastAPI · Streamlit</span>
    <span style="font-family:'Orbitron',monospace;font-size:0.55rem;letter-spacing:0.3em;color:rgba(0,245,255,0.2);">TARGET &gt;90% ACC</span>
</div>
""", unsafe_allow_html=True)