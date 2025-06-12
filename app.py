import streamlit as st
import cv2
from PIL import Image
from ultralytics import YOLO
import numpy as np
import tempfile

# Load model YOLOv8
model = YOLO("best.pt")  # Ganti dengan path model kamu jika berbeda

# Tampilkan daftar nama kelas
print(model.names)

# Sidebar: Pilih Mode
st.sidebar.title("Mode Deteksi")
mode = st.sidebar.selectbox("Pilih mode:", ["Gambar", "Video", "Webcam"])

st.title("Deteksi Kematangan Buah Jeruk - YOLOv8")

# Mode Gambar
if mode == "Gambar":
    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Gambar yang diunggah", use_column_width=True)

        # Konversi ke array dan deteksi
        img_np = np.array(image)
        results = model(img_np)

        # Tampilkan gambar hasil deteksi 
        results = model(image, conf=0.25)
        st.write(results[0].boxes)
        annotated_image  = results[0].plot()
        annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB) 
        st.image(annotated_image, caption="Hasil Deteksi", use_column_width=True)

# Mode Video
elif mode == "Video":
    video_file = st.file_uploader("Upload file video", type=["mp4", "avi", "mov"])
    if video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model(frame, conf=0.25)
            st.write(results[0].boxes)
            annotated_frame = results[0].plot()
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB) 
            stframe.image(annotated_frame, channels="BGR", use_column_width=True)

        cap.release()

# Mode Webcam (lokal saja)
elif mode == "Webcam":
    st.warning("Webcam hanya bisa digunakan saat dijalankan secara lokal.")
    st.write("Menyalakan webcam... Tekan 'q' di jendela untuk berhenti.")
    
    cap = cv2.VideoCapture(0)  # 0 = default webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            st.write("Gagal membaca frame dari webcam.")
            break

        results = model(frame, conf=0.25)
        st.write(results[0].boxes)
        annotated_frame = results[0].plot()

        cv2.imshow("Deteksi Webcam - Tekan q untuk keluar", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()