import os
import requests
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1) ชื่อไฟล์โมเดลและ URL สำหรับดาวน์โหลดจาก GitHub
MODEL_PATH = "best.pt"
RAW_URL = (
    "https://raw.githubusercontent.com/"
    "todsawatkmutt/deployshortpaperyolov11/main/best%20(2).pt"
)

@st.cache_resource(show_spinner=False)
def load_model():
    # ถ้ายังไม่มีไฟล์โมเดล ให้ดาวน์โหลดเก็บไว้
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading model weights…"):
            resp = requests.get(RAW_URL, stream=True)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
    # โหลดโมเดลด้วย Ultralytics YOLO
    return YOLO(MODEL_PATH)

# โหลดโมเดล (จะ cache ไว้ครั้งแรก)
model = load_model()

st.title("🕵️ Crack Detection with YOLO11 + Streamlit")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    # อ่านภาพที่ผู้ใช้ส่งมา
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    # รัน inference
    results = model.predict(source=np.array(img), conf=0.25, imgsz=1088)
    annotated = results[0].plot()

    # แสดงผลลัพธ์
    st.image(annotated, caption="Detection Result", use_container_width=True)
