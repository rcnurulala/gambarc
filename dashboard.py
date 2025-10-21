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
st.markdown("<p style='text-align:center; color:#6B21A8;'>Klasifikasi & Deteksi Objek Kucing 🐱 dan Anjing 🐶</p>", unsafe_allow_html=True)
st.markdown("---")

# ==========================
# SIDEBAR
# ==========================
st.sidebar.header("⚙️ Mode Analisis")

# 🐾 Tambahkan ilustrasi kucing dan anjing di sidebar
col_sb1, col_sb2 = st.sidebar.columns(2)
with col_sb1:
    st.image("https://cdn-icons-png.flaticon.com/512/616/616408.png", width=80, caption="Kucing 🐱")
with col_sb2:
    st.image("https://cdn-icons-png.flaticon.com/512/616/6164082.png", width=80, caption="Anjing 🐶")

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
        # ==========================
        # 🎯 DETEKSI OBJEK (YOLO)
        # ==========================
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

            prompt = (
                f"Model YOLO mendeteksi objek berikut di dalam gambar: {', '.join(detected_objects)}."
                if detected_objects else
                "Model YOLO tidak mendeteksi objek apapun. Jelaskan kemungkinan penyebabnya secara singkat."
            )

        # ==========================
        # 🧠 KLASIFIKASI GAMBAR
        # ==========================
        else:
            img_resized = img.resize((224, 224))
            img_array = np.expand_dims(image.img_to_array(img_resized), axis=0) / 255.0
            prediction = classifier.predict(img_array).flatten()

            class_names = ['Kucing', 'Anjing']
            num_classes = classifier.output_shape[-1]

            # Binary (Sigmoid)
            if num_classes == 1:
                prob_dog = float(prediction[0])
                prob_cat = 1 - prob_dog
                probs = [prob_cat, prob_dog]
                pred_label = 'Anjing' if prob_dog >= 0.5 else 'Kucing'
                confidence = max(probs)

            # Softmax (2 neuron)
            elif num_classes == 2:
                probs = prediction
                pred_label = class_names[np.argmax(probs)]
                confidence = float(np.max(probs))

            # Error handling
            else:
                st.warning(
                    f"⚠️ Model memiliki {num_classes} output neuron, "
                    "pastikan hanya dua kelas: Kucing & Anjing."
                )
                class_names = [f"Kelas_{i+1}" for i in range(num_classes)]
                probs = prediction
                pred_label = class_names[np.argmax(probs)]
                confidence = float(np.max(probs))

            with col2:
                st.markdown(f"### 🏷️ Prediksi: **{pred_label}**")
                st.progress(float(confidence))
                st.caption(f"Confidence: {confidence:.2%}")

                df = pd.DataFrame({'Kelas': class_names, 'Probabilitas': probs})
                st.bar_chart(df.set_index('Kelas'))

            prompt = (
                f"Model memprediksi gambar ini sebagai {pred_label} "
                f"dengan tingkat keyakinan {confidence:.2%}. "
                "Jelaskan hasil ini secara sederhana."
            )

    # ==========================
    # 💬 INTERPRETASI CHATGPT
    # ==========================
    st.markdown("---")
    st.subheader("💬 Interpretasi AI Terintegrasi")
    with st.spinner("🧠 Menghasilkan interpretasi..."):
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Kamu adalah AI yang menjelaskan hasil analisis gambar dengan cara alami dan edukatif."},
                {"role": "user", "content": prompt}
            ]
        )
    st.markdown(f"<div class='interpret-box'>{response.choices[0].message.content}</div>", unsafe_allow_html=True)

else:
    st.markdown("### 📥 Silakan unggah gambar di sidebar untuk memulai analisis.")
    st.image("https://cdn-icons-png.flaticon.com/512/4792/4792929.png", width=300)
    st.markdown("<p class='caption'>Belum ada gambar yang diunggah</p>", unsafe_allow_html=True)
