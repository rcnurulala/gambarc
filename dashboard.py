import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd
from openai import OpenAI

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Vision Studio",
    page_icon="🪶",
    layout="wide"
)

# ==========================
# CUSTOM STYLE
# ==========================
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #E0E7FF, #FDF2F8);
        font-family: "Poppins", sans-serif;
    }
    .block-container {
        background-color: white;
        padding: 2rem 3rem;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
    }
    h1, h2, h3 {
        color: #4C1D95;
        font-weight: 700;
    }
    .stButton>button {
        background-color: #7C3AED;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1rem;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #6D28D9;
        transform: scale(1.03);
    }
    .stProgress > div > div {
        background-color: #8B5CF6 !important;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODELS
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/model pt.pt")
    classifier = tf.keras.models.load_model("model/model h5.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# OPENAI CONFIG (new API)
# ==========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY_GAMBARC"])

# ==========================
# HEADER
# ==========================
st.markdown("<h1 style='text-align:center;'>🪶 AI Vision Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6B21A8;'>Klasifikasi & Deteksi Objek dengan Interpretasi AI Terintegrasi</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Mode Analisis")
mode = st.sidebar.radio("", ["🎯 Deteksi Objek (YOLO)", "🧠 Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.caption("💡 Gunakan YOLO untuk deteksi banyak objek, atau mode klasifikasi untuk satu jenis gambar.")

# ==========================
# PIPELINE
# ==========================
if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, caption="📸 Gambar Diupload", use_container_width=True)

    with st.spinner("🤖 Menganalisis gambar..."):
        if mode == "🎯 Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()
            detected = len(results[0].boxes)

            with col2:
                st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)
                st.markdown(f"### 🟣 Jumlah Objek Terdeteksi: **{detected}**")

            prompt = f"Gambar ini terdeteksi {detected} objek menggunakan YOLO. Jelaskan kemungkinan isi gambar tersebut secara informatif."

        else:
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(image.img_to_array(img_resized), axis=0) / 255.0
            prediction = classifier.predict(img_array)[0]

            class_names = ['Kucing', 'Anjing', 'Burung']  # ganti sesuai model kamu
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)
            pred_label = class_names[class_index]

            with col2:
                st.markdown(f"### 🏷️ Prediksi: **{pred_label}**")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence:.2%}")
                df = pd.DataFrame({'Kelas': class_names, 'Probabilitas': prediction})
                st.bar_chart(df.set_index('Kelas'))

            prompt = f"Model memprediksi gambar ini sebagai {pred_label} dengan keyakinan {confidence:.2%}. Jelaskan makna hasil ini dengan bahasa sederhana."

    # ==========================
    # INTERPRETASI CHATGPT
    # ==========================
    st.markdown("---")
    st.subheader("💬 Interpretasi AI Terintegrasi")
    with st.spinner("ChatGPT sedang menafsirkan hasil..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah AI yang menjelaskan hasil analisis gambar secara ramah dan mudah dipahami."},
                {"role": "user", "content": prompt}
            ]
        )
    st.markdown(f"<div style='background:#F5F3FF; padding:1rem; border-radius:10px;'>{response.choices[0].message.content}</div>", unsafe_allow_html=True)

    # ==========================
    # Q&A CHAT
    # ==========================
    st.markdown("### 🗨️ Tanya AI tentang hasil ini")
    user_q = st.text_input("Tulis pertanyaanmu:")
    if user_q:
        with st.spinner("ChatGPT sedang menjawab..."):
            q_response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Kamu adalah AI yang menjawab pertanyaan tentang hasil analisis gambar."},
                    {"role": "user", "content": f"Hasil model: {prompt}. Pertanyaan: {user_q}"}
                ]
            )
        st.markdown(f"**🤖 ChatGPT:** {q_response.choices[0].message.content}")

else:
    st.markdown("### 📥 Silakan unggah gambar di sidebar untuk memulai analisis.")
    st.image("https://cdn-icons-png.flaticon.com/512/4792/4792929.png", width=300)
    st.markdown("<p style='text-align:center; color:#6B7280;'>Belum ada gambar yang diunggah</p>", unsafe_allow_html=True)
