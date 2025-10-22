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
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================
# CUSTOM STYLE (COLORFUL)
# ==========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #FDE68A, #FBCFE8, #A5B4FC);
    font-family: 'Poppins', sans-serif;
}
.block-container {
    background-color: rgba(255,255,255,0.85);
    border-radius: 18px;
    padding: 2.2rem 3rem;
    box-shadow: 0 8px 25px rgba(0,0,0,0.12);
    backdrop-filter: blur(10px);
}
h1, h2, h3 { color: #4C1D95; font-weight: 700; }
.stButton>button {
    background: linear-gradient(90deg, #7C3AED, #EC4899);
    color: white; border-radius: 8px; border: none;
    padding: 0.6rem 1.2rem; font-weight: 600;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 0 15px rgba(236,72,153,0.5);
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #DDD6FE, #FBCFE8);
    color: #4C1D95;
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #4C1D95; }
.stProgress > div > div { background-color: #A78BFA !important; }
.interpret-box {
    background-color: #F5F3FF;
    border-left: 5px solid #7C3AED;
    padding: 1rem; border-radius: 10px; margin-top: 1rem;
}
.caption { color: #6B7280; font-size: 0.9rem; text-align: center; }
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
# OPENAI CONFIG
# ==========================
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY_GAMBARC"])

# ==========================
# HEADER
# ==========================
st.markdown("<h1 style='text-align:center;'>🪶 AI Vision Studio</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#6B21A8;'>Klasifikasi & Deteksi Objek dengan Sentuhan AI</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙ Mode Analisis")
mode = st.sidebar.radio("", ["🎯 Deteksi Objek (YOLO)", "🧠 Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.caption("💡 Pilih mode analisis, lalu unggah gambar untuk mulai.")

# ==========================
# PIPELINE
# ==========================
if uploaded_file:
    img = Image.open(uploaded_file)
    col1, col2 = st.columns([1, 1], gap="large")

    with col1:
        st.image(img, caption="📸 Gambar Diupload", use_container_width=True)

    with st.spinner("🤖 Menganalisis gambar..."):
        # 🎯 DETEKSI OBJEK (YOLO)
        if mode == "🎯 Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()
            boxes = results[0].boxes
            names = results[0].names

            detected_objects = []
            for box in boxes:
                cls = int(box.cls)
                conf = float(box.conf)
                label = names[cls]
                detected_objects.append(f"{label} ({conf*100:.1f}%)")

            with col2:
                st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)
                st.markdown("### 🟣 Objek Terdeteksi:")
                if detected_objects:
                    for obj in detected_objects:
                        st.write(f"- {obj}")
                else:
                    st.info("Tidak ada objek terdeteksi.")

        # 🧠 KLASIFIKASI GAMBAR
        else:
            img_proc = img.convert("RGB").resize((224, 224))
            img_array = np.expand_dims(np.array(img_proc) / 255.0, axis=0)

            prediction = classifier.predict(img_array)[0]
            class_names = ['Kucing', 'Anjing']
            pred_label = class_names[int(np.argmax(prediction))]
            confidence = float(np.max(prediction))

            with col2:
                st.markdown(f"### 🏷 Prediksi: *{pred_label}*")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence:.2%}")

                df = pd.DataFrame({'Kelas': class_names, 'Probabilitas': prediction})
                st.bar_chart(df.set_index('Kelas'))

# ==========================
# 💬 INTERPRETASI GAMBAR
# ==========================
    st.markdown("---")
    st.subheader("💬 Interpretasi Gambar oleh AI")

    # Buat prompt berdasar hasil
    if mode == "🎯 Deteksi Objek (YOLO)":
        if detected_objects:
            prompt = (
                f"Gambar ini menampilkan {', '.join(detected_objects)}. "
                "Jelaskan isi dan konteks visual gambar ini secara alami dan edukatif, tanpa memberikan saran atau pertanyaan."
            )
        else:
            prompt = (
                "Tidak ada objek yang terdeteksi pada gambar. "
                "Jelaskan kemungkinan isi visual gambar ini secara alami dan edukatif tanpa saran atau pertanyaan."
            )
    else:
        prompt = (
            f"Gambar ini diprediksi sebagai {pred_label} dengan tingkat keyakinan {confidence:.2%}. "
            "Jelaskan isi dan ciri visual gambar ini secara alami dan edukatif tanpa saran atau pertanyaan."
        )

    # Hasil interpretasi AI
    with st.spinner("🧠 Menghasilkan interpretasi..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu adalah AI yang hanya menjelaskan isi gambar secara deskriptif dan edukatif. "
                        "Tidak boleh memberikan saran, opini, atau pertanyaan lanjutan di luar isi gambar."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
        )

    interpretasi = response.choices[0].message.content
    st.markdown(f"<div class='interpret-box'>{interpretasi}</div>", unsafe_allow_html=True)

else:
    st.markdown("### 📥 Silakan unggah gambar di sidebar untuk memulai analisis.")
    st.image("https://cdn-icons-png.flaticon.com/512/4792/4792929.png", width=300)
    st.markdown("<p class='caption'>Belum ada gambar yang diunggah</p>", unsafe_allow_html=True)
