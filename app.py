import os
import requests
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# --- โหลด .env ถ้ามี
from dotenv import load_dotenv
load_dotenv()

# --- ดึงค่าจาก environment หรือจาก st.secrets
MODEL_PATH = os.getenv("MODEL_PATH", st.secrets.get("MODEL_PATH", "best.pt"))
RAW_URL    = os.getenv("RAW_URL",    st.secrets.get("RAW_URL",    ""))

@st.cache_resource(show_spinner=False)
def load_model():
    # ดาวน์โหลดถ้ายังไม่มี
    if not os.path.exists(MODEL_PATH):
        if not RAW_URL:
            st.error("❌ RAW_URL ไม่ได้ตั้งค่าไว้ใน environment หรือ st.secrets")
            st.stop()
        with st.spinner("Downloading model weights…"):
            resp = requests.get(RAW_URL, stream=True)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
    return YOLO(MODEL_PATH)

model = load_model()

st.title("🕵️ Crack Detection with YOLO11 + Streamlit")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    results = model.predict(source=np.array(img), conf=0.25, imgsz=1088)
    annotated = results[0].plot()
    st.image(annotated, caption="Detection Result", use_container_width=True)
