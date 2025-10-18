import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile

# Konfigurasi tampilan halaman
st.set_page_config(page_title="Deteksi Objek YOLOv8", page_icon="🐾", layout="wide")

# Gaya CSS lembut pastel
st.markdown("""
    <style>
        body {
            background-color: #FDF6F0;
        }
        .title {
            text-align: center;
            font-size: 36px;
            color: #2B2D42;
            font-weight: bold;
        }
        .subtitle {
            text-align: center;
            color: #8D99AE;
            font-size: 18px;
        }
        .stButton>button {
            background-color: #A2D2FF;
            color: #000;
            border-radius: 12px;
            padding: 8px 24px;
            font-weight: 600;
        }
        .stButton>button:hover {
            background-color: #FFAFCC;
            color: white;
        }
    </style>
""", unsafe_allow_html=True)

# Judul halaman
st.markdown("<div class='title'>🐾 Deteksi Objek dengan YOLOv8</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Unggah gambar dan lihat hasil deteksi objeknya!</div>", unsafe_allow_html=True)
st.markdown("---")

# Input gambar
uploaded_file = st.file_uploader("Unggah gambar", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns([1.2, 0.8])

    with col1:
        # Simpan sementara gambar
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_file.read())

        # Load model YOLOv8
        model = YOLO("yolov8n.pt")

        # Prediksi
        results = model(temp_file.name)
        res_img = results[0].plot()  # hasil visualisasi
        img = Image.fromarray(res_img)

        st.image(img, caption="Hasil Deteksi", use_column_width=True)

    with col2:
        st.subheader("📊 Hasil Interpretasi")
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                label = model.names[cls]
                st.markdown(f"- **{label.capitalize()}** terdeteksi dengan keyakinan **{conf:.2%}**")

        if not boxes:
            st.info("Tidak ada objek terdeteksi.")

st.markdown("---")
st.caption("✨ Dibuat oleh RC | YOLOv8 Object Detection Dashboard ✨")
