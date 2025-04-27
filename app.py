import os
import requests
import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# where we'll store the pt locally
MODEL_PATH = "best.pt"
# your raw GitHub URL (note the %20 for the space)
RAW_URL    = (
    "https://raw.githubusercontent.com/"
    "todsawatkmutt/deployshortpaperyolov11/main/best%20(2).pt"
)

@st.cache_resource(show_spinner=False)
def load_model():
    # download if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading weights…"):
            r = requests.get(RAW_URL, stream=True)
            r.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
    # now load with ultralytics
    return YOLO(MODEL_PATH)

model = load_model()

st.title("🕵️ Crack Detection with YOLO11 + Streamlit")

uploaded = st.file_uploader("Upload an image", type=["jpg","jpeg","png"])
if uploaded:
    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_column_width=True)

    results = model.predict(source=np.array(img), conf=0.25, imgsz=1088)
    annotated = results[0].plot()
    st.image(annotated, caption="Detection Result", use_column_width=True)
