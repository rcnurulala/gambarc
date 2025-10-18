import streamlit as st
from ultralytics import YOLO
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from PIL import Image
import numpy as np
import pandas as pd

# ==========================
# CONFIGURASI DASBOR
# ==========================
st.set_page_config(
    page_title="AI Vision App",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <style>
    .main {
        background-color: #F8FAFC;
        border-radius: 12px;
        padding: 1rem;
    }
    .stImage {
        border-radius: 12px;
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
st.title("🧠 Image Classification & Object Detection App")
st.markdown("### 🚀 Pilih mode analisis di sidebar dan unggah gambar untuk mulai.")

# ==========================
# PILIH MODE
# ==========================
menu = st.sidebar.radio(
    "Pilih Mode:",
    ["Deteksi Objek (YOLO)", "Klasifikasi Gambar"],
    index=0
)

uploaded_file = st.file_uploader("📤 Unggah Gambar", type=["jpg", "jpeg", "png"])

# ==========================
# PIPELINE PROSES
# ==========================
if uploaded_file is not None:
    img = Image.open(uploaded_file)

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="📸 Gambar yang Diupload", use_container_width=True)

    with st.spinner("🔍 Sedang menganalisis gambar..."):
        if menu == "Deteksi Objek (YOLO)":
            results = yolo_model(img)
            result_img = results[0].plot()
            with col2:
                st.image(result_img, caption="🎯 Hasil Deteksi Objek", use_container_width=True)
                st.success(f"Jumlah objek terdeteksi: {len(results[0].boxes)}")

        elif menu == "Klasifikasi Gambar":
            # Preprocessing
            img_resized = img.resize((224, 224))
            img_array = image.img_to_array(img_resized)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            # Prediksi
            prediction = classifier.predict(img_array)[0]
            class_names = ['Kucing', 'Anjing', 'Burung']  # ubah sesuai model kamu
            class_index = np.argmax(prediction)
            confidence = np.max(prediction)

            with col2:
                st.markdown(f"### 🏷️ Prediksi: **{class_names[class_index]}**")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence:.2%}")

                # Grafik probabilitas
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
    st.write("🧩 Mode:", menu)
    st.write("📁 Nama file:", uploaded_file.name)
    st.write("🌈 Ukuran gambar:", f"{img.size[0]} x {img.size[1]} px")

else:
    st.info("📥 Silakan unggah gambar terlebih dahulu untuk memulai analisis.")

