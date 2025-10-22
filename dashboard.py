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
st.sidebar.header("⚙️ Mode Analisis")
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
        # 🧠 KLASIFIKASI GAMBAR (robust preprocessing)
        # ==========================
        else:
            # --- Ambil input shape model (fallback ke 224x224)
            input_shape = classifier.input_shape  # biasanya (None, H, W, C) atau (None, C, H, W)
            # Default target size
            if input_shape is not None and len(input_shape) >= 3:
                # Cari H & W di input_shape
                if input_shape[1] is None or input_shape[2] is None:
                    target_h, target_w = 224, 224
                else:
                    # jika channels_last: (None, H, W, C)
                    # jika channels_first: (None, C, H, W)
                    if len(input_shape) == 4 and input_shape[-1] in (1, 3):
                        target_h, target_w = int(input_shape[1]), int(input_shape[2])
                    elif len(input_shape) == 4 and input_shape[1] in (1, 3):
                        target_h, target_w = int(input_shape[2]), int(input_shape[3])
                    else:
                        target_h, target_w = 224, 224
            else:
                target_h, target_w = 224, 224

            # --- Baca & normalisasi gambar sesuai kebutuhan model
            # Pastikan mode warna
            # Jika model ingin 1 channel -> grayscale, else RGB
            expected_channels = None
            if len(input_shape) == 4:
                # cek channels_last atau channels_first
                if input_shape[-1] in (1, 3):
                    expected_channels = int(input_shape[-1])  # channels_last
                    channels_last = True
                elif input_shape[1] in (1, 3):
                    expected_channels = int(input_shape[1])   # channels_first
                    channels_last = False
            # fallback: default 3 channel RGB
            if expected_channels is None:
                expected_channels = 3
                channels_last = True

            # Convert mode sesuai expected_channels
            if expected_channels == 1:
                img_proc = img.convert("L")  # grayscale
            else:
                img_proc = img.convert("RGB")

            img_resized = img_proc.resize((target_w, target_h))
            # Konversi ke numpy array dengan dtype float32
            img_np = np.array(img_resized).astype(np.float32) / 255.0

            # Jika model expects channels_first, transpose
            if not channels_last:
                # img_np shape now (H, W, C) or (H, W) if grayscale
                if img_np.ndim == 2:
                    img_np = np.expand_dims(img_np, axis=2)  # (H, W, 1)
                img_np = np.transpose(img_np, (2, 0, 1))  # -> (C, H, W)

            # Pastikan ada batch dimension
            if img_np.ndim == 3:
                img_array = np.expand_dims(img_np, axis=0)  # (1, H, W, C) atau (1, C, H, W)
            else:
                # unexpected shape
                img_array = img_np.reshape((1,) + img_np.shape)

            # Debug kecil (tampilkan bentuk model & input untuk membantu)
            st.write("Model input_shape:", classifier.input_shape)
            st.write("Prepared img_array.shape:", img_array.shape, "dtype:", img_array.dtype)

            # Prediksi dengan try/except untuk menangkap dan tampilkan error lebih informatif
            try:
                prediction_raw = classifier.predict(img_array)
            except Exception as e:
                st.error("Gagal melakukan prediksi. Cek model.input_shape dan img_array.shape di atas.")
                st.error(f"Error detail: {e}")
                # hentikan proses lanjutan
                prediction_raw = None

            if prediction_raw is not None:
                prediction = np.asarray(prediction_raw).flatten()

                class_names = ['Kucing', 'Anjing']
                num_classes = classifier.output_shape[-1] if classifier.output_shape is not None else None

                # Binary (sigmoid, 1 neuron)
                if num_classes == 1:
                    prob_dog = float(prediction[0])
                    probs = [1 - prob_dog, prob_dog]
                    pred_label = 'Anjing' if prob_dog >= 0.5 else 'Kucing'
                    confidence = max(probs)

                # Softmax (2 neuron)
                elif num_classes == 2:
                    probs = prediction
                    pred_label = class_names[int(np.argmax(probs))]
                    confidence = float(np.max(probs))

                # Fallback jika bentuk output tidak sesuai
                else:
                    probs = prediction.tolist()
                    # jika jumlah probs <> 2, buat label otomatis
                    if len(probs) != 2:
                        class_names = [f"Kelas_{i+1}" for i in range(len(probs))]
                    pred_label = class_names[int(np.argmax(probs))]
                    confidence = float(np.max(probs))

                # Tampilkan hasil
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
