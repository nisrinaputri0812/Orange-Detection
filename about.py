import streamlit as st
from PIL import Image

st.image("https://raw.githubusercontent.com/nisrinaputri0812/Orange-Detection/main/LogoBaru.jpg", width=200)

def run():
    st.subheader("ℹ️ Tentang Aplikasi")
    st.write("Aplikasi ini dikembangkan untuk mendeteksi jeruk secara otomatis pada gambar atau video. Dengan memanfaatkan teknologi kecerdasan buatan dan computer vision, proses identifikasi jeruk menjadi lebih cepat, akurat, dan efisien.")

    st.header("ℹ️ Tentang YOLOv8")
    st.write("""YOLOv8 (You Only Look Once versi 8) adalah model deteksi objek yang dikembangkan oleh Ultralytics. Model ini menggunakan pendekatan **real-time object detection** yang mampu mengenali objek dengan cepat dan akurat. YOLOv8 mendukung berbagai format input, termasuk gambar maupun video sehingga cocok untuk implementasi praktis di berbagai bidang.""")

    st.header("ℹ️ Manfaat Aplikasi")
    st.write("""
    - **Efisiensi**: Mempercepat proses identifikasi jeruk.
    - **Otomatisasi**: Membantu penyortiran buah jeruk secara otomatis.
    - **Akurasi Tinggi**: Menggunakan algoritma modern yang handal.
    - **Fleksibel**: Dapat digunakan untuk gambar maupun video. """)
