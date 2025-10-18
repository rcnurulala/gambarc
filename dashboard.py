import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd

# ==========================
# PAGE CONFIG
# ==========================
st.set_page_config(
    page_title="AI Vision Dashboard",
    page_icon="🧠",
    layout="wide"
)

# ==========================
# CUSTOM STYLE
# ==========================
st.markdown("""
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .stImage {
        border-radius: 12px;
        transition: transform 0.2s ease;
    }
    .stImage:hover {
        transform: scale(1.02);
    }
    .css-1v3fvcr {
        background-color: #EEF2FF !important;
        border-radius: 10px;
        padding: 0.5rem 1rem;
        color: #312E81;
    }
    h1 {
        text-align: center;
        color: #312E81;
        font-weight: 800;
    }
    h3 {
        color: #4338CA;
        font-weight: 700;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================
# LOAD MODEL
# ==========================
@st.cache_resource
def load_models():
    yolo_model = YOLO("model/model pt.pt")
    classifier = tf.keras.models.load_model("model/model h5.h5")
    return yolo_model, classifier

yolo_model, classifier = load_models()

# ==========================
# HEADER
# ==========================
st.markdown("## 🧠 Image Classification & Object Detection Dashboard")
st.markdown("<p style='text-align:center; color:#6366F1;'>Aplikasi AI sederhana untuk mendeteksi dan mengklasifikasi gambar secara real-time</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Pengaturan")
menu = st.sidebar.radio("Pilih Mode:", ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"])
uploaded_file = st.sidebar.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])
st.sidebar.markdown("---")
st.sidebar.info("Gunakan mode *Deteksi Objek* untuk mendeteksi banyak objek, atau *Klasifikasi* untuk mengenali jenis gambar tunggal.")

# ==========================
# MAIN DASHBOARD
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.image(img, caption="📸 Gambar Diupload", use_container_width=True)

    with st.spinner("🔍 Sedang menganalisis gambar..."):
        if menu == "Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()

            with col2:
                st.image(result_img, caption="🎯 Hasil Deteksi", use_container_width=True)
                st.success(f"Jumlah objek terdeteksi: {len(results[0].boxes)}")

        elif menu == "Klasifikasi Gambar":
            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)[0]
            class_names = ['Kucing', 'Anjing', 'Burung']  # ganti sesuai model kamu
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            with col2:
                st.markdown(f"### 🏷️ Prediksi: **{class_names[class_index]}**")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence:.2%}")

                df_pred = pd.DataFrame({
                    'Kelas': class_names,
                    'Probabilitas': prediction
                })
                st.bar_chart(df_pred.set_index('Kelas'))

    # ==========================
    # RINGKASAN HASIL
    # ==========================
    st.markdown("---")
    st.markdown("### 📊 Ringkasan Analisis")
    c1, c2, c3 = st.columns(3)
    c1.metric("Mode", menu)
    c2.metric("Nama File", uploaded_file.name)
    c3.metric("Ukuran", f"{img.size[0]} × {img.size[1]} px")

else:
    st.markdown("### 📥 Silakan unggah gambar di sidebar untuk mulai analisis.")
    st.image("https://cdn-icons-png.flaticon.com/512/4792/4792929.png", width=250)
    st.markdown("<p style='text-align:center; color:#6B7280;'>Belum ada gambar yang diunggah</p>", unsafe_allow_html=True)
