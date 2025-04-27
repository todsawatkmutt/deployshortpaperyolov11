import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# 1) โหลดโมเดลจาก path ของคุณ
MODEL_PATH = "/content/ultralytics11/runs/detect/train/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.title("🕵️ Crack Detection with YOLO11 + Streamlit")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # 2) รัน inference
    img_array = np.array(img)
    results = model.predict(source=img_array, conf=0.25, imgsz=1088)

    # 3) วาดกรอบบนภาพ
    annotated = results[0].plot()

    # 4) แสดงผล
    st.image(annotated, caption="Detection Result", use_column_width=True)
