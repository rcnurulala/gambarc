# dashboard.py
import os
import time
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image

# ML
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

# OpenAI v1 client & exceptions
from openai import OpenAI, RateLimitError, APIConnectionError, OpenAIError

# -------------------------
# PAGE + THEME
# -------------------------
st.set_page_config(page_title="AI Vision Studio", page_icon="🪶", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
    <style>
    /* PAGE BACKGROUND */
    .stApp {
        background: linear-gradient(180deg,#E6F6FF 0%, #FFF5F8 100%);
        font-family: "Poppins", sans-serif;
    }

    /* TOP HEADER */
    .topbar {
        background: linear-gradient(90deg,#2E86AB,#9BC1FF);
        color: white;
        padding: 18px 28px;
        border-radius: 12px;
        box-shadow: 0 8px 24px rgba(34,49,63,0.10);
        display:flex;
        align-items:center;
        justify-content:space-between;
        margin-bottom:18px;
    }
    .topbar h1 { margin:0; font-size:20px; font-weight:700; color: #ffffff; }
    .topbar .subtitle { color: rgba(255,255,255,0.92); font-size:13px; opacity:0.95; }

    /* LAYOUT CARDS */
    .card {
        background: white;
        border-radius: 12px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(16,24,40,0.06);
    }
    .stat-card { background: linear-gradient(180deg,#F7FBFF,#F0F8FF); border-radius:12px; padding:12px; }
    .small-muted { color:#6B7280; font-size:0.9rem; }

    /* SIDEBAR LOOK */
    .sidebar .stButton>button { border-radius:10px; }
    .left-nav { color:#ffffff; padding:12px 10px; display:block; border-radius:8px; margin-bottom:6px; }
    .left-nav:hover { background: rgba(255,255,255,0.06); }

    /* DETECT IMAGE FRAME */
    .img-frame { border-radius:12px; overflow:hidden; border:1px solid rgba(16,24,40,0.04); }

    /* AI box */
    .ai-box { background: linear-gradient(180deg,#FFF8F6,#FFF5F5); padding:14px; border-radius:12px; }

    </style>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# Helper: OpenAI client + safe call
# -------------------------
def get_openai_client():
    key_name = "OPENAI_API_KEY_GAMBARC"
    # prefer st.secrets
    key = None
    if key_name in st.secrets:
        key = st.secrets[key_name]
    else:
        key = os.environ.get(key_name)
    if not key:
        return None, f"OPENAI API key not found. Add `{key_name}` to Streamlit Secrets or set env var."
    try:
        client = OpenAI(api_key=key)
        return client, None
    except Exception as e:
        return None, f"Failed to init OpenAI client: {e}"

def call_chat_completion(client, messages, primary="gpt-4o-mini", fallback="gpt-3.5-turbo"):
    if client is None:
        return "🔒 OpenAI client not configured."
    attempts = 3
    for i in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=primary,
                messages=messages,
                temperature=0.2,
                max_tokens=350
            )
            return resp.choices[0].message.content
        except RateLimitError:
            if i < attempts - 1:
                time.sleep(2 + i*2)
                continue
            # try fallback
            try:
                resp = client.chat.completions.create(
                    model=fallback,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=350
                )
                return resp.choices[0].message.content
            except Exception as e:
                return f"Rate limit + fallback failed: {e}"
        except APIConnectionError:
            return "🚫 Connection error to OpenAI API."
        except OpenAIError as e:
            return f"OpenAI error: {e}"
        except Exception as e:
            return f"Unexpected error calling OpenAI: {e}"
    return "Unknown error."

# -------------------------
# Load models (cached)
# -------------------------
@st.cache_resource
def load_models():
    yolo_m = None
    clf = None
    try:
        yolo_m = YOLO("model/model pt.pt")
    except Exception as e:
        st.warning(f"⚠️ YOLO model load failed: {e}")
    try:
        clf = tf.keras.models.load_model("model/model h5.h5")
    except Exception as e:
        st.warning(f"⚠️ Classifier load failed: {e}")
    return yolo_m, clf

yolo_model, classifier = load_models()

# -------------------------
# App header (styled)
# -------------------------
st.markdown(f"""
<div class="topbar">
  <div>
    <h1>AI Vision Studio</h1>
    <div class="subtitle">Klasifikasi & Deteksi Objek • Interpretasi AI otomatis</div>
  </div>
  <div style="display:flex;align-items:center;gap:12px">
    <div style="background:#ffffff1f;padding:8px 12px;border-radius:999px;color:white;font-weight:600">Signed-in</div>
    <div style="width:40px;height:40px;border-radius:999px;background:#fff;display:flex;align-items:center;justify-content:center;font-weight:700;color:#2E86AB">RC</div>
  </div>
</div>
""", unsafe_allow_html=True)

# -------------------------
# Sidebar (styled minimal)
# -------------------------
st.sidebar.markdown("<div style='padding:10px 8px;background:linear-gradient(180deg,#2E86AB,#9BC1FF);border-radius:10px;color:white;text-align:center'><strong>Circle</strong></div>", unsafe_allow_html=True)
st.sidebar.markdown("")
st.sidebar.markdown("<div class='left-nav'>🏠 Overview</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='left-nav'>📊 Insights</div>", unsafe_allow_html=True)
st.sidebar.markdown("<div class='left-nav'>📁 Channels</div>", unsafe_allow_html=True)
st.sidebar.markdown("<hr/>", unsafe_allow_html=True)

st.sidebar.header("Analisis")
mode = st.sidebar.radio("", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("Unggah Gambar (JPG/PNG)", type=["jpg","jpeg","png"])
st.sidebar.caption("Pilih mode, lalu unggah gambar. Interpretasi akan otomatis ditampilkan di kanan.")

# -------------------------
# Top stat cards (visual)
# -------------------------
col1, col2 = st.columns([3,1])
with col1:
    st.markdown(
        """
        <div class="card" style="display:flex;gap:18px;align-items:center">
          <div style="flex:1">
            <h2 style="margin:0;color:#114B5F">Visits for today</h2>
            <div style="font-size:36px;font-weight:700;color:#114B5F">824</div>
            <div class="small-muted">Overview</div>
          </div>
          <div style="width:260px;height:120px;background:linear-gradient(90deg,#EBF8FF,#FFF7F6);border-radius:10px;display:flex;align-items:center;justify-content:center">
            <div style="text-align:center">
              <div style="font-weight:700;font-size:22px;color:#FF8A65">Popularity rate</div>
              <div style="font-size:28px;font-weight:800">87</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        """
        <div class="card" style="text-align:center">
          <h4 style="margin:6px 0">Finance</h4>
          <div style="font-size:22px;font-weight:700">$12,841</div>
          <div class="small-muted">Monthly income</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("<br/>")

# -------------------------
# Main content: image left, results right
# -------------------------
left_col, right_col = st.columns([1.2, 1])

with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📸 Gambar")
    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")
        st.markdown('<div class="img-frame">', unsafe_allow_html=True)
        st.image(img, use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown("<div class='small-muted'>File: {}</div>".format(getattr(uploaded_file, "name", "uploaded_image")), unsafe_allow_html=True)
    else:
        st.info("Unggah gambar dari sidebar untuk mulai analisis.")
    st.markdown('</div>', unsafe_allow_html=True)

with right_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("🧾 Hasil & Interpretasi")
    placeholder_results = st.empty()
    placeholder_metrics = st.empty()
    placeholder_ai = st.empty()
    st.markdown('</div>', unsafe_allow_html=True)

# -------------------------
# If uploaded, run models and show results
# -------------------------
if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(img)

    with st.spinner("Memproses gambar..."):
        prompt = "Tidak ada hasil model."
        # YOLO path
        if mode.startswith("Deteksi") and yolo_model is not None:
            try:
                results = yolo_model(img_np)
                # visualize detection overlay
                try:
                    vis = results[0].plot()
                    vis_pil = Image.fromarray(vis)
                    placeholder_results.image(vis_pil, use_column_width=True, caption="Hasil Deteksi (overlay)")
                except Exception:
                    placeholder_results.image(img, use_column_width=True, caption="Gambar (visualisasi overlay gagal)")

                # gather detected labels & confidences
                boxes = getattr(results[0], "boxes", [])
                names = getattr(results[0], "names", {})
                detected_objects = []
                confs = []
                for b in boxes:
                    try:
                        cls = int(b.cls)
                        conf = float(b.conf)
                        label = names.get(cls, str(cls))
                        detected_objects.append((label, conf))
                        confs.append(conf)
                    except Exception:
                        # try fallback array layout
                        try:
                            arr = np.array(b)
                            cls = int(arr[5])
                            conf = float(arr[4])
                            label = names.get(cls, str(cls))
                            detected_objects.append((label, conf))
                            confs.append(conf)
                        except Exception:
                            pass

                # metrics
                detected_count = len(detected_objects)
                avg_conf = float(np.mean(confs)) if confs else 0.0

                placeholder_metrics.markdown(
                    f"""
                    <div style='display:flex;gap:12px'>
                      <div style='flex:1' class='stat-card'>
                        <div style='font-size:14px;color:#6B7280'>Objek terdeteksi</div>
                        <div style='font-weight:800;font-size:28px;color:#114B5F'>{detected_count}</div>
                      </div>
                      <div style='flex:1' class='stat-card'>
                        <div style='font-size:14px;color:#6B7280'>Confidence rata-rata</div>
                        <div style='font-weight:800;font-size:28px;color:#114B5F'>{avg_conf*100:.1f}%</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                # build prompt with labels + confidences
                if detected_objects:
                    detected_text = ", ".join([f"{lbl} ({conf*100:.1f}%)" for lbl, conf in detected_objects])
                    prompt = f"Model YOLO mendeteksi objek berikut: {detected_text}. Jelaskan hasil deteksi ini secara singkat dan fokus pada objek yang terdeteksi, jangan berspekulasi di luar daftar tersebut."
                else:
                    prompt = "Model YOLO tidak mendeteksi objek. Jelaskan kemungkinan penyebab singkat."

            except Exception as e:
                placeholder_results.error(f"Gagal menjalankan YOLO: {e}")
                prompt = "Gagal menjalankan YOLO."

        # Classification path
        elif mode.startswith("Klasifikasi") and classifier is not None:
            try:
                img_resized = img.resize((224,224))
                arr = keras_image.img_to_array(img_resized)
                arr = np.expand_dims(arr, axis=0)/255.0
                pred = classifier.predict(arr)[0]
                class_names = ['Kucing','Anjing','Burung']  # change if needed
                idx = int(np.argmax(pred))
                conf = float(np.max(pred))
                label = class_names[idx]

                placeholder_results.markdown(f"### 🏷️ Prediksi: **{label}**")
                placeholder_results.progress(conf)
                placeholder_results.caption(f"Confidence: {conf:.2%}")
                dfp = pd.DataFrame({'Kelas': class_names, 'Prob': pred})
                placeholder_results.bar_chart(dfp.set_index('Kelas'))

                prompt = f"Model mengklasifikasikan gambar sebagai {label} dengan tingkat keyakinan {conf:.2%}. Jelaskan interpretasi singkat dan apa artinya."

                placeholder_metrics.markdown(
                    f"""
                    <div style='display:flex;gap:12px'>
                      <div style='flex:1' class='stat-card'>
                        <div style='font-size:14px;color:#6B7280'>Label</div>
                        <div style='font-weight:800;font-size:22px;color:#114B5F'>{label}</div>
                      </div>
                      <div style='flex:1' class='stat-card'>
                        <div style='font-size:14px;color:#6B7280'>Confidence</div>
                        <div style='font-weight:800;font-size:22px;color:#114B5F'>{conf:.2%}</div>
                      </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            except Exception as e:
                placeholder_results.error(f"Gagal menjalankan classifier: {e}")
                prompt = "Gagal menjalankan classifier."

        else:
            placeholder_results.info("Model yang dipilih tidak tersedia pada server.")
            prompt = "Model tidak tersedia."

    # -------------------------
    # AI Interpretation (automatic)
    # -------------------------
    placeholder_ai.markdown("### 💬 Interpretasi AI Terintegrasi")
    client, client_err = get_openai_client()
    if client_err:
        placeholder_ai.error(client_err)
    else:
        with st.spinner("ChatGPT sedang menafsirkan..."):
            messages = [
                {"role":"system","content":"Kamu adalah asisten yang menjelaskan hasil deteksi secara singkat, langsung, dan fokus pada daftar objek yang diberikan. Jangan menambahkan objek yang tidak terdeteksi."},
                {"role":"user","content":prompt}
            ]
            ai_text = call_chat_completion(client, messages, primary="gpt-4o-mini", fallback="gpt-3.5-turbo")
            placeholder_ai.markdown(f"<div class='ai-box'>{ai_text}</div>", unsafe_allow_html=True)

# -------------------------
# Footer
# -------------------------
st.markdown("<br/><div style='text-align:center; color:#7B8794'>© 2025 AI Vision Studio • Designed by RC</div>", unsafe_allow_html=True)
