import streamlit as st
from PIL import Image
from model import load_model, predict_face_shape
import os

# Load model
MODEL_PATH = "face_shape_model.pth"
model = load_model(MODEL_PATH)

# Mapping bentuk wajah ke frame kacamata
face_shape_to_frames = {
    "Heart": ["frame1.jpg", "frame2.jpg"],
    "Oval": ["frame3.jpg", "frame4.jpg"],
    "Round": ["frame5.jpg", "frame6.jpg"],
    "Square": ["frame7.jpg", "frame8.jpg"],
    "Oblong": ["frame9.jpg", "frame10.jpg"]
}

st.title("Sistem Rekomendasi Kacamata Berdasarkan Bentuk Wajah")

uploaded_file = st.file_uploader("Upload foto wajah kamu", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Foto Wajah", use_column_width=True)

    if st.button("Deteksi Bentuk Wajah & Rekomendasi"):
        predicted_shape = predict_face_shape(model, image)
        st.success(f"Bentuk wajah terdeteksi: **{predicted_shape}**")

        recommended = face_shape_to_frames.get(predicted_shape, [])
        st.markdown("### Rekomendasi Frame Kacamata:")
        for frame in recommended:
            frame_path = os.path.join("images", frame)
            if os.path.exists(frame_path):
                st.image(frame_path, width=200)
            else:
                st.write(f"- {frame} (gambar tidak ditemukan)")
